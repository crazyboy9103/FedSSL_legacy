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
from models import ResNet50, ResNet18, VGG16
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
    np.random.seed(args.seed)
    
    models_dict = {"resnet18": ResNet18, "resnet50": ResNet50, "vgg16": VGG16}
    if args.exp == "simclr":
        # BUILD MODEL
        global_model = models_dict[args.model](
            pretrained = args.pretrained, 
            out_dim = args.out_dim, 
            exp = "simclr", 
            mode = "train", 
            freeze = args.freeze,
            pred_dim = None, 
            num_classes = 10
        )
    
    elif args.exp == "simsiam":
         global_model = models_dict[args.model](
            pretrained = args.pretrained, 
            out_dim = args.out_dim, 
            exp = "simsiam", 
            mode = "train", 
            freeze = args.freeze,
            pred_dim = args.pred_dim, 
            num_classes = 10
        )
                
    elif args.exp == "FL":
        global_model = models_dict[args.model](
            pretrained = args.pretrained, 
            out_dim = args.out_dim, 
            exp = "FL", 
            mode = "train", 
            freeze = args.freeze,
            pred_dim = None, 
            num_classes = 10
        )
        
    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    
    # load dataset and user train indices
    train_dataset, test_dataset, warmup_dataset, user_train_idxs = get_dataset(args)
    
    # only used small fraction of the test data 
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
            test_dataset if args.sup_warmup else warmup_dataset,
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
        
    
    
    # Training
    valid_loss, valid_top1, valid_top5 = [],  [], []

    for epoch in range(args.epochs):
        local_weights, local_losses, local_top1s, local_top5s = {}, {}, {}, {}
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        # Select clients for training in this round
        num_clients_part = int(args.frac * args.num_users)
        assert num_clients_part > 0
        part_users_ids = np.random.choice(range(args.num_users), num_clients_part, replace=False)
        
        threads = []
        for i, client_id in enumerate(part_users_ids):
            trainset = Subset(train_dataset, user_train_idxs[client_id]) 
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
        
        
        # Compute cosine similarity between model weights
        cos_sim = compute_similarity(torch.device(f"cuda:{args.train_device}"), local_weights)
        
        for layer_idx, cos in enumerate(cos_sim):
            tb_writer.add_scalar(f"cos_sim_round_{epoch}", cos, layer_idx)
            
        # aggregate weights
        global_weights = average_weights(local_weights)

        # update global weights
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
        
        state_dict, _, loss_avg, top1_avg, top5_avg = server_model.test(finetune=True)
        global_model.load_state_dict(state_dict)
        
        valid_loss.append(loss_avg)
        valid_top1.append(top1_avg)
        valid_top5.append(top5_avg)

        # print global training loss after every 'i' rounds
        #if (epoch+1) % print_every == 0:
        print(f' \nAvg Validation Stats after {epoch+1} global rounds:')
        print(f'Validation Loss : {loss_avg:.2f}')
        print(f'Validation Accuracy: top1/top5 {top1_avg:.2f}%/{top5_avg:.2f}%\n')
        
        tb_writer.add_scalar("Server_loss", loss_avg, epoch)
        tb_writer.add_scalar("Server_top1_acc", top1_avg, epoch)
        tb_writer.add_scalar("Server_top5_acc", top5_avg, epoch)
    
    tb_writer.close()