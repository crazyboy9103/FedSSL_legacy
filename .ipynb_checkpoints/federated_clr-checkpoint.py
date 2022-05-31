#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset

from options import args_parser

from trainers import NormalFL, SimCLR, SimSiam
from update import LocalModel
from models import LinearEvalModel, ResNet50, ResNet18, SimSiamResNet18, SimSiamResNet50
from utils import get_dataset, average_weights, exp_details, compute_similarity
from CKA import CKA, CudaCKA
# import threading 
# import GPUtil
# GPUtil.showUtilization()

if __name__ == '__main__':
    args = args_parser()
    
    # Tensorboard writer
    tb_writer = SummaryWriter(args.log_path)
    
    # stdout experiment details
    exp_details(args, tb_writer)

    device = torch.device(f'cuda:{args.train_device}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    

    if args.exp == "simclr":
        # BUILD MODEL
        if args.model == "resnet18":
            global_model = ResNet18(
                pretrained = args.pretrained, 
                out_dim = args.out_dim, 
                simclr = True
            )
        elif args.model == "resnet50":
            global_model = ResNet50(
                pretrained = args.pretrained, 
                out_dim = args.out_dim, 
                simclr = True
            )
    
    elif args.exp == "simsiam":
        if args.model == "resnet18":
            global_model = SimSiamResNet18(
                pretrained = args.pretrained, 
                out_dim = args.out_dim, 
                pred_dim = args.pred_dim
            )
        elif args.model == "resnet50":
            global_model = SimSiamResNet50(
                pretrained = args.pretrained, 
                out_dim = args.out_dim, 
                pred_dim = args.pred_dim
            )
                
    elif args.exp == "FL":
        if args.model == "resnet18":
            global_model = ResNet18(
                pretrained = args.pretrained, 
                out_dim = args.out_dim, 
                simclr = False
            )
        elif args.model == "resnet50":
            global_model = ResNet50(
                pretrained = args.pretrained, 
                out_dim = args.out_dim, 
                simclr = False
            )

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    
    # load dataset and user train indices
    train_dataset, test_dataset, warmup_dataset, user_train_idxs = get_dataset(args)
    
    
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
            warmup_dataset, 
            batch_size=args.warmup_bs, 
            shuffle=True, 
            num_workers=8,
            pin_memory=True
        )
        
        if args.exp == "simclr":
            server_model = SimCLR(
                args = args,
                model = copy.deepcopy(global_model), 
                train_loader = None, 
                test_loader = test_loader,
                warmup_loader = warmup_loader,
                device = device, 
                client_id = -1
            )
        elif args.exp == "simsiam":
            server_model = SimSiam(
                args = args,
                model = copy.deepcopy(global_model), 
                train_loader = None, 
                test_loader = test_loader,
                warmup_loader = warmup_loader,
                device = device, 
                client_id = -1
            )
        elif args.exp == "FL":
            server_model = NormalFL(
                args = args,
                model = copy.deepcopy(global_model), 
                train_loader = None, 
                test_loader = test_loader,
                warmup_loader = warmup_loader,
                device = device, 
                client_id = -1
            )
        # Start warmup
        warmup_state = server_model.warmup(args.warmup_epochs)
        
        # Get model_state_dict
        warmup_model_state = warmup_state["model"]
        
        # Initializes the current global model with the state_dict 
        global_model.load_state_dict(warmup_model_state)
        
    
    
    # Training
    valid_loss, valid_top1, valid_top5 = [],  [], []

    for epoch in range(args.epochs):
        local_weights, local_losses, local_top1s, local_top5s = {}, {}, {}, {}
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        
        # Select clients for training in this round
        num_clients_part = int(args.frac * args.num_users)
        assert num_clients_part > 0
        part_users_ids = np.random.choice(range(args.num_users), num_clients_part, replace=False)
        
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
            
            model_state_dict, loss, top1, top5 = local_model.update_weights()
            
            print(f"client {i} updated")
            # collect weights, metrics
            local_weights[i] = copy.deepcopy(model_state_dict)
            local_losses[i] = copy.deepcopy(loss)
            local_top1s[i] = top1
            local_top5s[i] = top5
        
        
            
            
        for i, client_id in enumerate(part_users_ids):
            loss, top1, top5 = local_losses[i], local_top1s[i], local_top5s[i]
            
            tb_writer.add_scalar(f"val_loss_client_{i}", loss, epoch)
            tb_writer.add_scalar(f"val_top1_acc_client_{i}", top1, epoch)
            tb_writer.add_scalar(f"val_top5_acc_client_{i}", top5, epoch)
        
        
        # Convert dictionary into list
        # local_weights = list(local_weights.values())
        
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
        
        if args.exp == "simclr":
            server_model = SimCLR(
                args = args,
                model = copy.deepcopy(global_model), 
                train_loader = None, 
                test_loader = test_loader,
                warmup_loader = None,
                device = device, 
                client_id = -1
            )
            
        elif args.exp == "FL":
            server_model = NormalFL(
                args = args,
                model = copy.deepcopy(global_model), 
                train_loader = None, 
                test_loader = test_loader,
                device = device, 
                client_id = -1
            )
        
        elif args.exp == "simsiam":
            server_model = SimSiam(
                args = args,
                model = copy.deepcopy(global_model), 
                train_loader = None, 
                test_loader = test_loader,
                warmup_loader = None,
                device = device, 
                client_id = -1
            )
        
        loss_avg, top1_avg, top5_avg = server_model.test()
        
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