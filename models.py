#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
from torchvision import models
import copy 

# VGG
class VGG16(nn.Module):
    def __init__(self, pretrained, out_dim, exp, mode, freeze, pred_dim = None, num_classes = 10):
        super(VGG16, self).__init__()
        self.backbone = models.vgg16(pretrained=pretrained)
        mlp_dim = self.backbone.classifier[0].in_features
        
        self.exp = exp
        self.mode = mode
        self.freeze = freeze
        
        if self.exp == "simclr":
            self.backbone.classifer = nn.Sequential(
                nn.Linear(mlp_dim, out_dim), 
                nn.ReLU(), 
                nn.Linear(out_dim, out_dim), 
            )
            
            self.predictor = nn.Sequential(
                nn.Linear(out_dim, num_classes)
            )
        elif self.exp == "simsiam":
            assert pred_dim != None
            self.backbone.classifier = nn.Sequential(
                nn.Linear(mlp_dim, out_dim, bias=False), 
                nn.BatchNorm1d(out_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(out_dim, mlp_dim, bias=False), 
                nn.BatchNorm1d(mlp_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(mlp_dim, out_dim, bias=False), 
                nn.BatchNorm1d(out_dim, affine=False)
            )

            # Remove bias as it's followed by BN
            #self.backbone.classifier[6].bias.requires_grad = False

            self.predictor = nn.Sequential(
                nn.Linear(out_dim, pred_dim), 
                nn.BatchNorm1d(pred_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(pred_dim, out_dim)
            )
            
        elif self.exp == "FL":
            self.backbone.classifier = nn.Identity()
            
            self.predictor = nn.Sequential(
                nn.Linear(mlp_dim, num_classes)
            )
        
        elif self.exp == "orchestra":
            self.backbone.classifier = nn.Linear(mlp_dim, out_dim)
            
            hidden_dim = 512
            self.predictor = nn.Sequential(
                nn.Linear(mlp_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim, affine=False), 
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim, bias=False), 
                nn.BatchNorm1d(out_dim, affine=False), 
            )
            
            self.target_backbone = copy.deepcopy(self.backbone)
            for target_param in self.target_backbone.parameters():
                target_param.requires_grad = False
            
            self.target_predictor = copy.deepcopy(self.predictor)
            for target_pred_param in self.target_predictor.parameters():
                target_pred_param.requires_grad = False
            
            #Rotation
            self.deg_layer = nn.Linear(hidden_dim, 4)
            
            self.mem_projections = nn.Linear(self.m_size, 512, bias=False)
            self.centroids = nn.Linear(512, self.N_centroids, bias=False) # must be defined second last
            self.local_centroids = nn.Linear(512, self.N_local, bias=False) # must be defined last
            
            
            
            
        if self.mode == "linear":
            requires_grad = False if freeze else True
            for param in self.backbone.parameters():
                param.requires_grad = requires_grad
            
            # Train last linear layer is must
            for param in self.predictor.parameters():
                param.requires_grad = True
                
    def forward(self, x1, x2=None, x3=None, deg_labels=None):
        if self.mode == "linear":
            return self.predictor(self.backbone(x1))
        
        elif self.mode == "train":
            if self.exp == "simclr":
                return self.backbone(x1)
            
            elif self.exp == "simsiam":
                z1 = self.backbone(x1)
                z2 = self.backbone(x2)

                p1 = self.predictor(z1)
                p2 = self.predictor(z2)
        
                return p1, p2, z1.detach(), z2.detach()
        
            elif self.exp == "FL":
                return self.predictor(self.backbone(x1))
            
            elif self.exp == "orchestra":
                N = x1.shape[0]
                C = self.centroids.weight.data.detach().T
                
                Z1 = F.normalize(self.predictor(self.backbone(x1)), dim=1)
                Z2 = F.normalize(self.predictor(self.backbone(x2)), dim=1)
                
                cZ2 = Z2 @ C
                
                logpZ2 = torch.log(F.softmax(cZ2 / self.T, dim=1))
                
                with torch.no_grad():
                    tau = self.ema
                    
                    for target, online in zip(self.target_backbone.parameters(), self.backbone.parameters()):
                        target.data = (tau) * target.data + (1 - tau) * online.data
                    for target, online in zip(self.target_projector.parameters(), self.projector.parameters()):
                        target.data = (tau) * target.data + (1 - tau) * online.data
                        
                    tZ1 = F.normalize(self.target_predictor(self.target_backbone(x1)), dim=1)
                    
                    cP1 = tZ1 @ C
                    tP1 = F.softmax(cP1 / self.T, dim=1)
                
                L_cluster = -torch.sum(tP1 * logpZ2, dim=1).mean()
                
                deg_preds = self.deg_layer(self.predictor(self.backbone(x3)))
                L_deg = F.cross_entropy(deg_preds, deg_labels)
                L = L_cluster + L_deg
                
                with torch.no_grad():
                    N = tZ1.shape[0]
                    self.mem_projections.weight.data[:, :-N] = self.mem_projections.weight.data[:, N:].detach().clone()
                    self.mem_projections.weight.data[:, -N:] = tZ1.T.detach().clone()
                
                return L

# ResNet

class ResNet50(nn.Module):
    def __init__(self, pretrained, out_dim, exp, mode, freeze, pred_dim = None, num_classes = 10):
        super(ResNet50, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        mlp_dim = self.backbone.fc.in_features
        
        self.exp = exp
        self.mode = mode
        self.freeze = freeze
        
        if self.exp == "simclr":
            self.backbone.fc = nn.Sequential(
                nn.Linear(mlp_dim, out_dim), 
                nn.ReLU(), 
                nn.Linear(out_dim, out_dim), 
            )
            
            self.predictor = nn.Sequential(
                nn.Linear(out_dim, num_classes)
            )
        elif self.exp == "simsiam":
            assert pred_dim != None
            self.backbone.fc = nn.Sequential(
                nn.Linear(mlp_dim, out_dim, bias=False), 
                nn.BatchNorm1d(out_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(out_dim, mlp_dim, bias=False), 
                nn.BatchNorm1d(mlp_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(mlp_dim, out_dim, bias=False), 
                nn.BatchNorm1d(out_dim, affine=False)
            )

            # Remove bias as it's followed by BN
            #self.backbone.classifier[6].bias.requires_grad = False

            self.predictor = nn.Sequential(
                nn.Linear(out_dim, pred_dim), 
                nn.BatchNorm1d(pred_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(pred_dim, out_dim)
            )
            
        elif self.exp == "FL":
            self.backbone.fc = nn.Identity()
            
            self.predictor = nn.Sequential(
                nn.Linear(mlp_dim, num_classes)
            )
        
        if self.mode == "linear":
            requires_grad = False if freeze else True
            for param in self.backbone.parameters():
                param.requires_grad = requires_grad
            
            # Train last linear layer is must
            for param in self.predictor.parameters():
                param.requires_grad = True
                
    def forward(self, x1, x2 = None):
        if self.mode == "linear":
            return self.predictor(self.backbone(x1))
        
        elif self.mode == "train":
            if self.exp == "simclr":
                return self.backbone(x1)
            
            elif self.exp == "simsiam":
                z1 = self.backbone(x1)
                z2 = self.backbone(x2)

                p1 = self.predictor(z1)
                p2 = self.predictor(z2)
        
                return p1, p2, z1.detach(), z2.detach()
        
            elif self.exp == "FL":
                return self.predictor(self.backbone(x1))
        
class ResNet18(nn.Module):
    def __init__(self, pretrained, out_dim, exp, mode, freeze, pred_dim = None, num_classes = 10):
        super(ResNet18, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        mlp_dim = self.backbone.fc.in_features
        
        self.exp = exp
        self.mode = mode
        self.freeze = freeze
        
        
        if self.exp == "simclr":
            self.backbone.fc = nn.Sequential(
                nn.Linear(mlp_dim, out_dim), 
                nn.ReLU(), 
                nn.Linear(out_dim, out_dim), 
            )
            
            self.predictor = nn.Sequential(
                nn.Linear(out_dim, num_classes)
            )
        elif self.exp == "simsiam":
            assert pred_dim != None
            self.backbone.fc = nn.Sequential(
                nn.Linear(mlp_dim, out_dim, bias=False), 
                nn.BatchNorm1d(out_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(out_dim, mlp_dim, bias=False), 
                nn.BatchNorm1d(mlp_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(mlp_dim, out_dim, bias=False), 
                nn.BatchNorm1d(out_dim, affine=False)
            )

            # Remove bias as it's followed by BN
            #self.backbone.classifier[6].bias.requires_grad = False

            self.predictor = nn.Sequential(
                nn.Linear(out_dim, pred_dim), 
                nn.BatchNorm1d(pred_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(pred_dim, out_dim)
            )
            
        elif self.exp == "FL":
            self.backbone.fc = nn.Identity()
            
            self.predictor = nn.Sequential(
                nn.Linear(mlp_dim, num_classes)
            )
        
        if self.mode == "linear":
            requires_grad = False if freeze else True
            for param in self.backbone.parameters():
                param.requires_grad = requires_grad
            
            # Train last linear layer is must
            for param in self.predictor.parameters():
                param.requires_grad = True
                
    def forward(self, x1, x2 = None):
        if self.mode == "linear":
            return self.predictor(self.backbone(x1))
        
        elif self.mode == "train":
            if self.exp == "simclr":
                return self.backbone(x1)
            
            elif self.exp == "simsiam":
                z1 = self.backbone(x1)
                z2 = self.backbone(x2)

                p1 = self.predictor(z1)
                p2 = self.predictor(z2)
        
                return p1, p2, z1.detach(), z2.detach()
        
            elif self.exp == "FL":
                return self.predictor(self.backbone(x1))
