#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
from torchvision import models
import copy 
import torch

            
class ResNet50(nn.Module):
    def __init__(self, pretrained, out_dim, exp, mode, freeze, pred_dim = None, num_classes = 10):
        super(ResNet50, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        mlp_dim = self.backbone.fc.in_features
        
        self.exp = exp
        self.freeze = freeze
        
        if self.exp == "simclr":
            self.backbone.fc = nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim), 
                nn.ReLU(), 
                nn.Linear(mlp_dim, out_dim), 
            )
            
            self.predictor = nn.Sequential(
                nn.Linear(out_dim, num_classes), 
                nn.Softmax(dim=-1)
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
                self.backbone.fc,
                #nn.Linear(mlp_dim, out_dim, bias=False), 
                nn.BatchNorm1d(out_dim, affine=False)
            )

            # Remove bias as it's followed by BN
            #self.backbone.classifier[6].bias.requires_grad = False

            self.projector = nn.Sequential(
                nn.Linear(out_dim, pred_dim, bias=False), 
                nn.BatchNorm1d(pred_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(pred_dim, out_dim)
            )
            
            self.predictor = nn.Sequential(
                nn.Linear(out_dim, num_classes),
                nn.Softmax(dim=-1)
            )
            
            
        elif self.exp == "FL":
            self.backbone.fc = nn.Identity()
            
            self.predictor = nn.Sequential(
                nn.Linear(mlp_dim, num_classes), 
                nn.Softmax(dim=-1)
            )
        
        self.set_mode(mode)
            
    def set_mode(self, mode):
        self.mode = mode
        
        if mode == "linear":
            if self.freeze:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.backbone.eval()
            else:
                for param in self.backbone.parameters():
                    param.requires_grad = True
                self.backbone.train()
            
                    
        elif mode == "train":
            for param in self.backbone.parameters():
                param.requires_grad = True
            
            self.backbone.train()
            
        for param in self.predictor.parameters():
            param.requires_grad = True
        
        self.predictor.train()

            
    def forward(self, x1, x2 = None):
        if self.mode == "linear": 
            if self.freeze:
                with torch.no_grad():
                    z1 = self.backbone(x1)
                return self.predictor(z1)
            
            else:
                return self.predictor(self.backbone(x1))
        
        elif self.mode == "train":
            # self.backbone.train()
            
            if self.exp == "simclr":
                return self.backbone(x1)
            
            elif self.exp == "simsiam":
                z1 = self.backbone(x1)
                z2 = self.backbone(x2)

                p1 = self.projector(z1)
                p2 = self.projector(z2)
        
                return p1, p2, z1.detach(), z2.detach()
        
            elif self.exp == "FL":
                return self.predictor(self.backbone(x1))
        
class ResNet18(nn.Module):
    def __init__(self, pretrained, out_dim, exp, mode, freeze, pred_dim = None, num_classes = 10):
        super(ResNet18, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        mlp_dim = self.backbone.fc.in_features
        
        self.exp = exp
        self.freeze = freeze
        
        
        if self.exp == "simclr":
            self.backbone.fc = nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim), 
                nn.ReLU(), 
                nn.Linear(mlp_dim, out_dim), 
            )
            
            self.predictor = nn.Sequential(
                nn.Linear(out_dim, num_classes),
                nn.Softmax(dim=-1)
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

            self.projector = nn.Sequential(
                nn.Linear(out_dim, pred_dim), 
                nn.BatchNorm1d(pred_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(pred_dim, out_dim)
            )
            
            self.predictor = nn.Sequential(
                nn.Linear(out_dim, num_classes),
                nn.Softmax(dim=-1)
            )
            
        elif self.exp == "FL":
            self.backbone.fc = nn.Identity()
            
            self.predictor = nn.Sequential(
                nn.Linear(mlp_dim, num_classes),
                nn.Softmax(dim=-1)
            )
            
        self.set_mode(mode)
        
    def set_mode(self, mode):
        self.mode = mode
        if mode == "linear":
            if self.freeze:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                
                self.backbone.eval()
            
            else:
                for param in self.backbone.parameters():
                    param.requires_grad = True
                
                self.backbone.train()
            
            
        elif mode == "train":
            for param in self.backbone.parameters():
                param.requires_grad = True
            
            self.backbone.train()
        
        for param in self.predictor.parameters():
            param.requires_grad = True
        
        self.predictor.train()
            
    
    def forward(self, x1, x2 = None):
        if self.mode == "linear":
            if self.freeze:
                with torch.no_grad():
                    z1 = self.backbone(x1)
                return self.predictor(z1)
            
            else:
                return self.predictor(self.backbone(x1))
        
        elif self.mode == "train":
            # self.backbone.train()
            if self.exp == "simclr":
                return self.backbone(x1)
            
            elif self.exp == "simsiam":
                z1 = self.backbone(x1)
                z2 = self.backbone(x2)

                p1 = self.projector(z1)
                p2 = self.projector(z2)
        
                return p1, p2, z1.detach(), z2.detach()
        
            elif self.exp == "FL":
                return self.predictor(self.backbone(x1))


# Orchestra Clustering function
def sknopp(cZ, lamb=25, max_iters=100):
    with torch.no_grad():
        N_samples, N_centroids = cZ.shape # cZ is [N_samples, N_centroids]
        probs = F.softmax(cZ * lamd, dim=1).T # probs should be [N_centroids, N_samples]

        r = torch.ones((N_centroids, 1), device=probs.device) / N_centroids # desired row sum vector
        c = torch.ones((N_samples, 1), device=probs.device) / N_samples # desired col sum vector

        inv_N_centroids = 1. / N_centroids
        inv_N_samples = 1. / N_samples

        err = 1e3
        for it in range(max_iters):
            r = inv_N_centroids / (probs @ c)  # (N_centroids x N_samples) @ (N_samples, 1) = N_centroids x 1
            c_new = inv_N_samples / (r.T @ probs).T  # ((1, N_centroids) @ (N_centroids x N_samples)).t() = N_samples x 1
            if it % 10 == 0:
                err = torch.nansum(torch.abs(c / c_new - 1))
            c = c_new
            if (err < 1e-2):
                break

        # inplace calculations. 
        probs *= c.squeeze()
        probs = probs.T # [N_samples, N_centroids]
        probs *= r.squeeze()

        return probs * N_samples # Soft assignments
# VGG
class VGG16(nn.Module):
    def __init__(self, pretrained, out_dim, exp, mode, freeze, pred_dim = None, num_classes = 10, m_size=None, N_centroids=None, ema=None, T=None):
        super(VGG16, self).__init__()
        self.backbone = models.vgg16(pretrained=pretrained)
        mlp_dim = self.backbone.classifier[0].in_features
        
        self.exp = exp
        self.mode = mode
        self.freeze = freeze
        
        if self.exp == "simclr":
            self.backbone.classifier = nn.Sequential(
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
            assert m_size != None, "memory buffer size = None"
            assert N_centroids != None, "N centroids = None"
            assert N_local_centroids != None, "N local centroids = None"
            assert ema != None, "ema = None"
            assert T != None, "T = None"
            self.m_size = m_size
            self.N_centroids = N_centroids
            self.N_local_centroids = N_local_centroids
            self.ema = ema
            self.T = T
            
            # used as feature extractor
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
            self.mem_proj = nn.Linear(m_size, hidden_dim, bias=False)
            self.centroids = nn.Linear(hidden_dim, N_centroids, bias=False)
            self.local_centroids = nn.Linear(hidden_dim, N_local_centroids, bias=False)
    
    @torch.no_grad()
    def reset_memory(self, data):
        proj_bank = []
        n_samples = 0
        for x, _ in data:
            if n_samples >= self.m_size:
                break
            z = F.normalize(self.target_predictor(self.target_backbone(x[0])), dim=1)
            proj_bank.append(z)
            n_samples += x[0].shape[0]
        
        proj_bank = torch.cat(proj_bank, dim=0).contiguous()
        if n_samples > self.m_size:
            proj_bank = proj_bank[:self.m_size]
            
        self.mem_proj.weight.data.copy_(proj_bank.T)
    
    @torch.no_grad()
    def update_target(self):
        tau = self.ema
        for target, online in zip(self.target_backbone.parameters(), self.backbone.parameters()):
            target.data = (tau) * target.data + (1-tau) * online.data
        for target, online in zip(self.target_predictor.parameters(), self.predictor.parameters()):
            target.data = (tau) * target.data + (1 - tau) * online.data
    
    @torch.no_grad()
    def local_clustering(self):
        Z = self.mem_proj.weight.data.T.detach().clone()
        centroids = Z[np.random.choice(Z.shape[0], self.N_local_centroids, replace=False)]
        local_iters = 5
        
        for i in range(local_iters):
            assigns = sknopp(Z @ centroids.T, max_iters=10)
            choice_cluster = torch.argmax(assigns, dim=1)
            
            for idx in range(self.N_local_centroids):
                selected = torch.nonzero(choice_cluster == idx).squeeze()
                selected = torch.index_select(Z, 0, selected)
                if selected.shape[0] == 0:
                    selected = Z[torch.randint(len(Z), (1,))]
                centroids[idx] = F.normalize(selected.mean(dim=0), dim=0)
        
        self.local_centroids.weight.data.copy_(centroids)
      
    def global_clustering(self, Z1, nG=1., nL=1.):
        N = Z1.shape[0]
        optimizer = torch.optim.SGD(self.centroids.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        train_loss = 0
        total_rounds = 500
        for round_idx in range(total_rounds):
            with torch.no_grad():
                SK_assigns = sknopp(self.centroids(Z1))
            
            
            
            probs1 = F.softmax(self.centroids(F.normalize(Z1, dim=1)) / self.T, dim=1)
            loss = -F.cosine_similarity(SK_assigns, probs1, dim=-1).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                self.centroids.weight.copy_(F.normalize(self.centroids.weight.data.clone(), dim=1))
                train_loss += loss.item()
        
            

    def forward(self, x1, x2=None, x3=None, deg_labels=None):
        if self.mode == "linear":
            if self.freeze:
                self.backbone.eval()
            else:
                self.backbone.train()  
            return self.predictor(self.backbone(x1))
        
        elif self.mode == "train":
            self.backbone.train()
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
                C = self.centroids.weight.data.detach().clone().T
                Z1 = F.normalize(self.predictor(self.backbone(x1)), dim=1)
                Z2 = F.normalize(self.predictor(self.backbone(x2)), dim=1)
                
                cZ2 = Z2 @ C
                
                logpZ2 = torch.log(F.softmax(cZ2 / self.T, dim=1))
                
                with torch.no_grad():
                    self.update_target()
                    tZ1 = F.normalize(self.target_predictor(self.target_backbone(x1)), dim=1)
                    
                    cP1 = tZ1 @ C
                    tP1 = F.softmax(cP1 / self.T, dim=1)
                
                L_cluster = -torch.sum(tP1 * logpZ2, dim=1).mean()
                
                deg_preds = self.deg_layer(self.predictor(self.backbone(x3)))
                L_deg = F.cross_entropy(deg_preds, deg_labels)
                L = L_cluster + L_deg
                with torch.no_grad():
                    self.update_memory(tZ1)
                return L