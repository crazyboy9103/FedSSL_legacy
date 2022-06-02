#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
from torchvision import models
import copy 



class LinearEvalModel(nn.Module):
    def __init__(self, linear_eval_model):
        super(LinearEvalModel, self).__init__()
        self.model = linear_eval_model
        
    def forward(self, x):
        return self.model(x)
    
class ResNet50(nn.Module):
    def __init__(self, pretrained, out_dim, simclr):
        super(ResNet50, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        mlp_dim = self.backbone.fc.in_features
        if simclr:
            self.backbone.fc = nn.Sequential(
                nn.Linear(mlp_dim, out_dim), 
                nn.ReLU(), 
                nn.Linear(out_dim, out_dim),
                #nn.Softmax(dim=1)
            )
        
        else:
            self.backbone.fc = nn.Sequential(
                nn.Linear(mlp_dim, out_dim),
                #nn.Softmax(dim=1)
            )
            
    
    def linear_eval_model(self, freeze, num_classes):
        backbone_copy = copy.deepcopy(self.backbone)
        in_dim = backbone_copy.fc[0].in_features
        
        linear_layer = nn.Sequential(
            nn.Linear(in_dim, num_classes), 
            nn.Softmax(dim=1)
        )
        backbone_copy.fc = linear_layer
        
        requires_grad = False if freeze else True
        for param in backbone_copy.parameters():
            param.requires_grad = requires_grad
        
        for param in backbone_copy.fc.parameters():
            param.requires_grad = True
        
        return backbone_copy
            
    def forward(self, x):
        return self.backbone(x)
    
class ResNet18(nn.Module):
    def __init__(self, pretrained, out_dim, simclr):
        super(ResNet18, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        mlp_dim = self.backbone.fc.in_features
        if simclr:
            self.backbone.fc = nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim), 
                nn.ReLU(), 
                nn.Linear(mlp_dim, out_dim) 
                #nn.Softmax(dim=1)
            )
        else:
            self.backbone.fc = nn.Sequential(
                nn.Linear(mlp_dim, out_dim),
                #nn.Softmax(dim=1)
            )
    
    def linear_eval_model(self, freeze, num_classes):
        backbone_copy = copy.deepcopy(self.backbone)
        in_dim = backbone_copy.fc[0].in_features
        
        linear_layer = nn.Sequential(
            nn.Linear(in_dim, num_classes), 
            nn.Softmax(dim=1)
        )
        backbone_copy.fc = linear_layer
        
        requires_grad = False if freeze else True
        for param in backbone_copy.parameters():
            param.requires_grad = requires_grad
        
        for param in backbone_copy.fc.parameters():
            param.requires_grad = True
        
        return backbone_copy
    
    def forward(self, x):
        return self.backbone(x)
    

class SimSiamResNet18(nn.Module):
    def __init__(self, pretrained, out_dim=2048, pred_dim=512):
        super(SimSiamResNet18, self).__init__()
        
        self.encoder = models.resnet18(zero_init_residual=True, pretrained=pretrained)
        in_features = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Linear(in_features, out_dim)
    
        self.encoder.fc = nn.Sequential(
            nn.Linear(in_features, out_dim, bias=False), 
            nn.BatchNorm1d(out_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(out_dim, in_features, bias=False), 
            nn.BatchNorm1d(in_features), 
            nn.ReLU(inplace=True), 
            self.encoder.fc, 
            nn.BatchNorm1d(out_dim, affine=False)
        )
        
        # Remove bias as it's followed by BN
        self.encoder.fc[6].bias.requires_grad = False
        
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, pred_dim, bias=False), 
            nn.BatchNorm1d(pred_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(pred_dim, out_dim)
        )
    
    def forward(self, x1, x2):
        # x1, x2 : two views
        
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        return p1, p2, z1.detach(), z2.detach()
    
    
class SimSiamResNet50(nn.Module):
    def __init__(self, pretrained, out_dim=2048, pred_dim=512):
        super(SimSiamResNet50, self).__init__()
        
        self.encoder = models.resnet50( zero_init_residual=True, pretrained=pretrained)
        in_features = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Linear(in_features, out_dim)

        self.encoder.fc = nn.Sequential(
            nn.Linear(in_features, out_dim, bias=False), 
            nn.BatchNorm1d(out_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(out_dim, in_features, bias=False), 
            nn.BatchNorm1d(in_features), 
            nn.ReLU(inplace=True), 
            self.encoder.fc, 
            nn.BatchNorm1d(out_dim, affine=False)
        )
        
        # Remove bias as it's followed by BN
        self.encoder.fc[6].bias.requires_grad = False
        
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, pred_dim), 
            nn.BatchNorm1d(pred_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(pred_dim, out_dim)
        )
    
    def forward(self, x1, x2):
        # x1, x2 : two views
        
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        return p1, p2, z1.detach(), z2.detach()
    

    
class SimSiamLinear(nn.Module):
    def __init__(self, trained_encoder, freeze, num_classes):
        super(SimSiamLinear, self).__init__()
        self.encoder_copy = copy.deepcopy(trained_encoder)
        in_features = self.encoder_copy.fc[0].in_features
        
        linear_layer = nn.Sequential(
            nn.Linear(in_features, num_classes), 
            nn.Softmax(dim=1)
        )
        self.encoder_copy.fc = linear_layer
        
        requires_grad = False if freeze else True
        for param in self.encoder_copy.parameters():
            param.requires_grad = requires_grad
        
        for param in self.encoder_copy.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.encoder_copy(x)
        