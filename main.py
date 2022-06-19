#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import numpy as np
from tqdm import tqdm
import threading 

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset

from options import args_parser

from trainers import Trainer
from update import LocalModel
from models import ResNet_model
from utils import get_dataset, average_weights, exp_details, compute_similarity
from CKA import CKA, CudaCKA

if __name__ == '__main__':
    args = args_parser()
    
    # Tensorboard writer
    tb_writer = SummaryWriter(args.log_path)
    
    # stdout experiment details
    exp_details(args, tb_writer)

    device = torch.device(f'cuda:{args.train_device}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
   

    # BUILD MODEL
    global_model = ResNet_model(args)
    
   
    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.set_mode("train")
    
    # load dataset and user train indices
    train_dataset, test_dataset, warmup_dataset, user_train_idxs = get_dataset(args)
    
    if args.exp != "FL":
        # only used smaller fraction of the test data 
        len_data = len(test_dataset)
        idxs = [i for i in range(len_data)]
        np.random.shuffle(idxs)
        len_data = int(len_data * args.server_data_frac)
        idxs = idxs[:len_data]
        test_dataset = Subset(test_dataset, idxs)
    
    if args.warmup:
        # Only test, warmup are used
        test_loader  = DataLoader(
            test_dataset, 
            batch_size=args.local_bs, 
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        warmup_loader = DataLoader(
            test_dataset if args.sup_warmup else warmup_dataset, # if sup_warmup, use 1 view sup dataset; else 2 view simclr dataset
            batch_size=args.warmup_bs, 
            shuffle=True, 
            num_workers=8,
            pin_memory=True
        )

        server_model = Trainer(
            args = args,
            model = copy.deepcopy(global_model), 
            train_loader = None, 
            test_loader = test_loader,
            warmup_loader = warmup_loader,
            device = device, 
            client_id = -1
        )

        # Start warmup
        warmup_state = server_model.warmup(args.warmup_epochs, args.sup_warmup)

        # Get model_state_dict
        warmup_model_state = warmup_state["model"]

        # Initializes the current global model with the state_dict 
        global_model.load_state_dict(warmup_model_state)

    
    num_clients_part = int(args.frac * args.num_users)
    assert num_clients_part > 0
    # Training 
    for epoch in range(args.epochs):
        local_weights, local_losses, local_top1s, local_top5s = {}, {}, {}, {}
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        # Select clients for training in this round
        part_users_ids = np.random.choice(range(args.num_users), num_clients_part, replace=False)
        
        threads = []
        for i, client_id in enumerate(part_users_ids):
            trainset = Subset(
                train_dataset, 
                user_train_idxs[client_id]
            ) 
            # instantiate local model
            local_model = LocalModel(
                args=args,
                model=copy.deepcopy(global_model), 
                trainset=trainset,
                test_dataset=test_dataset, 
                client_id = i
            )
            
            if args.parallel:
                thread = threading.Thread(target=local_model.update_weights_parallel, args=(local_weights, local_losses, local_top1s, local_top5s,))
                threads.append(thread)
                thread.start()
                
            else:
                model_state_dict, loss, top1, top5 = local_model.update_weights()

                print(f"client {i} updated")
                # collect weights, metrics
                local_weights[i] = model_state_dict
                local_losses[i] = loss
                local_top1s[i] = top1
                local_top5s[i] = top5
        
        if args.parallel:
            for thread in threads:
                thread.join()
            
            
        for i, client_id in enumerate(part_users_ids):
            loss, top1, top5 = local_losses[i], local_top1s[i], local_top5s[i]
            
            tb_writer.add_scalar(f"val_loss_client_{i}", loss, epoch)
            tb_writer.add_scalar(f"val_top1_acc_client_{i}", top1, epoch)
            tb_writer.add_scalar(f"val_top5_acc_client_{i}", top5, epoch)
        
        
        # simclr 나 simsiam의 경우 finetune이 무조건 필요 (classifier 가 없으므로)
        # 1. aggregation 전에 finetune, args.finetune_before_agg = True
        # 2. finetune 전에 aggregation, args.finetune_before_agg = False
        if args.exp != "FL":
            # Finetune each client's model before aggregation
            if args.finetune_before_agg:
                for i, client_id in enumerate(part_users_ids):
                    local_weight = local_weights[i]
                    global_model.load_state_dict(local_weight)
                     # test loader for linear eval 
                    test_loader  = DataLoader(
                        test_dataset, 
                        batch_size=args.local_bs, 
                        shuffle=False, 
                        num_workers=4, 
                        pin_memory=True
                    )

                    server_model = Trainer(
                        args = args,
                        model = copy.deepcopy(global_model), 
                        train_loader = None, 
                        test_loader = test_loader,
                        warmup_loader = None,
                        device = device, 
                        client_id = -1
                    )

                    state_dict, loss_avg, top1_avg, top5_avg = server_model.test(
                        finetune=args.finetune, 
                        epochs=args.finetune_epoch
                    )
                         
                    local_weights[i] = state_dict

                # aggregate weights
                global_weights = average_weights(local_weights)
                
                global_model.load_state_dict(global_weights)
                
            # Finetune each client's model after aggregation
            if not args.finetune_before_agg:
                # aggregate weights
                global_weights = average_weights(local_weights)

                global_model.load_state_dict(global_weights)
                 # test loader for linear eval 
                test_loader  = DataLoader(
                    test_dataset, 
                    batch_size=args.local_bs, 
                    shuffle=False, 
                    num_workers=4, 
                    pin_memory=True
                )

                server_model = Trainer(
                    args = args,
                    model = copy.deepcopy(global_model), 
                    train_loader = None, 
                    test_loader = test_loader,
                    warmup_loader = None,
                    device = device, 
                    client_id = -1
                )

                state_dict, loss_avg, top1_avg, top5_avg = server_model.test(
                    finetune=args.finetune, 
                    epochs=args.finetune_epoch
                )

                if args.finetune:
                    missing_keys, unexpected_keys = global_model.load_state_dict(state_dict)
                    print(f"missing keys {missing_keys}")
                    print(f"unexp keys {unexpected_keys}")
                    
                    
        # FL일 경우 finetune 하지 않고 aggregate된 weight로만 성능 평가 
        else:
            # aggregate weights only
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)
            # test loader for linear eval 
            test_loader  = DataLoader(
                test_dataset, 
                batch_size=args.local_bs, 
                shuffle=False, 
                num_workers=4, 
                pin_memory=True
            )

            server_model = Trainer(
                args = args,
                model = copy.deepcopy(global_model), 
                train_loader = None, 
                test_loader = test_loader,
                warmup_loader = None,
                device = device, 
                client_id = -1
            )

            state_dict, loss_avg, top1_avg, top5_avg = server_model.test(
                finetune=False, 
                epochs=1
            )
            
        print("#######################################################")
        print(f' \nAvg Validation Stats after {epoch+1} global rounds')
        print(f'Validation Loss     : {loss_avg:.2f}')
        print(f'Validation Accuracy : top1/top5 {top1_avg:.2f}%/{top5_avg:.2f}%\n')
        print("#######################################################")
        
        tb_writer.add_scalar("Server_loss", loss_avg, epoch)
        tb_writer.add_scalar("Server_top1_acc", top1_avg, epoch)
        tb_writer.add_scalar("Server_top5_acc", top5_avg, epoch)
    
    tb_writer.close()