#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from trainers import Trainer
import copy 

class LocalModel():
    def __init__(self, args, client_id, model, trainset, test_dataset):
        self.args = args
        self.client_id = client_id
        self.model = model
        self.model.train()
        
        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=args.lr, 
                momentum=args.momentum)
            
        elif args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=args.lr, 
                weight_decay=args.weight_decay
            )
        
        
        self.device = torch.device(f"cuda:{args.train_device}") if torch.cuda.is_available() else torch.device("cpu")
        
        
        self.train_loader = DataLoader(
            trainset, 
            batch_size=args.local_bs, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True
        )
        self.test_loader  = DataLoader(
            test_dataset, 
            batch_size=args.local_bs, 
            shuffle=False, 
            num_workers=8, 
            pin_memory=True
        )
        

        self.trainer = Trainer(
            args = self.args,
            model = self.model, 
            train_loader = self.train_loader,
            test_loader = self.test_loader,
            warmup_loader = None,
            device = self.device, 
            client_id = client_id
        )

        
        

    def update_weights(self):
        best_model_state = self.trainer.train()
        model_state_dict, loss, top1, top5 = best_model_state["model"], best_model_state["loss"], best_model_state["top1"], best_model_state["top5"]
        return model_state_dict, loss, top1, top5