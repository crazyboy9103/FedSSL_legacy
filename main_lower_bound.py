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
from sampling import get_train_idxs

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
    global_model.set_mode("linear")
    
    # load dataset and user train indices
    train_dataset, test_dataset, warmup_dataset, user_train_idxs = get_dataset(args)
    
    
    server_data_iid = args.server_data_iid
    
    len_data = len(test_dataset)
    len_data = int(len_data * args.server_data_frac)
    if server_data_iid:
        # only used smaller fraction of the test data 
        train_idxs = get_train_idxs(
            test_dataset,
            num_users = 1, 
            num_items = len_data, 
            alpha=100000
        )
        
    else: 
        train_idxs = get_train_idxs(
            test_dataset,
            num_users = 1, 
            num_items = len_data, 
            alpha=0.1
        )
    
    test_idxs = list(set([i for i in range(len_data)])- set(train_idxs[0]))
    finetune_dataset = Subset(test_dataset, train_idxs[0])
    test_dataset = Subset(test_dataset, test_idxs)
    
    # Training 
    for epoch in range(args.epochs):
        finetune_loader =  DataLoader(
            finetune_dataset, 
            batch_size=args.local_bs, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True,
            drop_last=True
        )

         # test loader for linear eval 
        test_loader  = DataLoader(
            test_dataset, 
            batch_size=args.local_bs, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True,
            drop_last=True
        )

        server_model = Trainer(
            args = args,
            model = copy.deepcopy(global_model), 
            train_loader = None, 
            test_loader = finetune_loader,
            warmup_loader = None,
            device = device, 
            client_id = -1
        )

        # Finetune and save model
        state_dict, _, _, _ = server_model.test(
            finetune=True, 
            epochs=args.finetune_epoch
        )
        

        # Then test
        server_model.test_loader = test_loader
        _, loss_avg, top1_avg, top5_avg = server_model.test(
            finetune=False, 
            epochs=1
        )
        
        global_model.load_state_dict(state_dict)
            
        print("#######################################################")
        print(f' \nAvg Validation Stats after {epoch+1} global rounds')
        print(f'Validation Loss     : {loss_avg:.2f}')
        print(f'Validation Accuracy : top1/top5 {top1_avg:.2f}%/{top5_avg:.2f}%\n')
        print("#######################################################")
        
        tb_writer.add_scalar("Server_loss", loss_avg, epoch)
        tb_writer.add_scalar("Server_top1_acc", top1_avg, epoch)
        tb_writer.add_scalar("Server_top5_acc", top5_avg, epoch)
    
    tb_writer.close()