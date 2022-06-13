
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import random
import torch
import torch.nn as nn
import os
from torchvision import datasets, transforms
from torchvision.transforms.functional import rotate

from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid, cifar_noniid_unequal

from CKA import CKA, CudaCKA

def compute_similarity(device, client_weights):
    with torch.no_grad():
        # layer by layer similarity

        # Take only conv layers
        names = [name for name in client_weights[0].keys() if "conv" in name and "weight" in name]
        N = len(names)
        #cka_sim = torch.zeros(N, device=device)
        cos_sim = torch.zeros(N)

        # CKA, Cosine
        #cuda_cka = CudaCKA(device)
        cos = nn.CosineSimilarity(dim=0, eps=1e-6).to(device)

        N_weights = len(client_weights)
        for i in range(N_weights):
            for j in range(i+1, N_weights):
                ith_state_dict = client_weights[i]
                jth_state_dict = client_weights[j]

                for sim_idx, name in enumerate(names):
                    i_layer_weight = ith_state_dict[name]
                    j_layer_weight = jth_state_dict[name]
                    i_layer_vec = i_layer_weight.view(-1).unsqueeze(-1).to(device)
                    j_layer_vec = j_layer_weight.view(-1).unsqueeze(-1).to(device)

                    #cka_value = cuda_cka.linear_CKA(i_layer_vec, j_layer_vec).cpu().item()
                    #cka_sim[sim_idx] += cka_value
                    
                    cos_value = cos(i_layer_vec, j_layer_vec).cpu().item()
                    cos_sim[sim_idx] += cos_value


        total = N_weights * (N_weights - 1) / 2
        #cka_sim /= total
        cos_sim /= total

    return cos_sim
            
class SimCLRTransformWrapper(object):
    def __init__(self, base_transform, args):
        self.base_transform = base_transform
        self.n_views = args.n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)] # two views by default

class OrchestraTransformWrapper(object):
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, x):
        rand = random.random()
        if rand < 0.25:
            angle = 0
        elif 0.25 < rand < 0.5:
            angle = 1
        elif 0.5 < rand < 0.75:
            angle = 2
        else:
            angle = 3
            
        x1 = self.base_transform(x)
        x2 = self.base_transform(x)
        x3 = rotate(self.base_transform(x), 90 * angle)
        return [x1, x2, x3], angle
    
def get_dataset(args):
    cifar_data_path = os.path.join(args.data_path, "cifar")
    mnist_data_path = os.path.join(args.data_path, "mnist")
    
    if args.exp == "simclr" or args.exp == "simsiam":
        s = args.strength
        target_size = args.target_size
        
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            #transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * target_size)),
            transforms.ToTensor()
        ])
        
        test_transforms = transforms.Compose([
            transforms.Resize(size=target_size), 
            transforms.ToTensor()
        ])

        warmup_transforms = train_transforms
        
        if args.dataset == "cifar":
            train_dataset = datasets.CIFAR10(
                cifar_data_path, 
                train=True, 
                transform=SimCLRTransformWrapper(train_transforms, args), 
                download=True
            )
            test_dataset = datasets.CIFAR10(
                cifar_data_path, 
                train=False, 
                transform=test_transforms, 
                download=True
            )
            warmup_dataset = datasets.CIFAR10(
                cifar_data_path, 
                train=False, 
                transform=SimCLRTransformWrapper(warmup_transforms, args), 
                download=True
            )
            
            if args.iid:
                user_train_idxs = cifar_iid(
                    train_dataset, 
                    args.num_users, 
                    args.num_items
                )
            
            else:
                if args.unequal:
                    user_train_idxs = cifar_noniid_unequal(
                        train_dataset, 
                        args.num_users,
                        args.num_items, 
                        args.alpha, 
                        args.num_class_per_client
                    )

                else:
                    # Chose equal size splits for every user
                    user_train_idxs = cifar_noniid(
                        train_dataset, 
                        args.num_users, 
                        args.num_items, 
                        args.alpha,
                        args.num_class_per_client
                    )
        
        elif args.dataset == 'mnist':
            train_dataset = datasets.MNIST(
                mnist_data_path, 
                train=True, 
                download=True,
                transform=SimCLRTransformWrapper(train_transforms, args)
            )

            test_dataset = datasets.MNIST(
                mnist_data_path, 
                train=False, 
                download=True,
                transform=test_transforms
            )
            
            warmup_dataset = datasets.MNIST(
                mnist_data_path, 
                train=False, 
                download=True,
                transform=SimCLRTransformWrapper(warmup_transforms, args),
            )

            # sample training data amongst users

            if args.iid:
                # Sample IID user data from Mnist
                user_train_idxs = mnist_iid(
                    train_dataset, 
                    args.num_users, 
                    args.num_items
                )
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose unequal splits for every user
                    user_train_idxs = mnist_noniid_unequal(
                        train_dataset, 
                        args.num_users, 
                        args.num_items, 
                        args.alpha, 
                        args.num_class_per_client
                    )
                else:
                    # Chose equal splits for every user
                    user_train_idxs = mnist_noniid(
                        train_dataset, 
                        args.num_users, 
                        args.num_items, 
                        args.alpha, 
                        args.num_class_per_client
                    )

    
    # Normal FL
    elif args.exp == "FL":
        s = args.strength
        target_size = args.target_size
        
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            #transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * target_size)),
            transforms.ToTensor()
        ])
        
        test_transforms = transforms.Compose([
            transforms.Resize(size=target_size), 
            transforms.ToTensor()
        ])
        
        if args.dataset == 'cifar':
            # transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # ])

            train_dataset = datasets.CIFAR10(
                cifar_data_path, 
                train=True, 
                download=True,
                transform=train_transforms
            )

            test_dataset = datasets.CIFAR10(
                cifar_data_path, 
                train=False, 
                download=True,
                transform=test_transforms
            )
            
            warmup_dataset = datasets.CIFAR10(
                cifar_data_path, 
                train=False, 
                transform=test_transforms, 
                download=True
            )

            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_train_idxs = cifar_iid(
                    train_dataset, 
                    args.num_users, 
                    args.num_items
                )

            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose unequal splits for every user
                    user_train_idxs = cifar_noniid_unequal(
                        train_dataset, 
                        args.num_users,
                        args.num_items, 
                        args.alpha, 
                        args.num_class_per_client
                    )

                else:
                    # Chose equal splits for every user
                    user_train_idxs = cifar_noniid(
                        train_dataset, 
                        args.num_users, 
                        args.num_items, 
                        args.alpha,
                        args.num_class_per_client
                    )

        elif args.dataset == 'mnist':
            train_dataset = datasets.MNIST(
                mnist_data_path, 
                train=True, 
                download=True,      
                transform=train_transforms
            )

            test_dataset = datasets.MNIST(
                mnist_data_path, 
                train=False, 
                download=True,
                transform=test_transforms
            )
             
            warmup_dataset = datasets.MNIST(
                cifar_data_path, 
                train=False, 
                transform=test_transforms, 
                download=True
            )

            # sample training data amongst users

            if args.iid:
                # Sample IID user data from Mnist
                user_train_idxs = mnist_iid(
                    train_dataset, 
                    args.num_users, 
                    args.num_items
                )
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose unequal splits for every user
                    user_train_idxs = mnist_noniid_unequal(
                        train_dataset, 
                        args.num_users, 
                        args.num_items, 
                        args.alpha, 
                        args.num_class_per_client
                    )
                else:
                    # Chose equal splits for every user
                    user_train_idxs = mnist_noniid(
                        train_dataset, 
                        args.num_users, 
                        args.num_items, 
                        args.alpha, 
                        args.num_class_per_client
                    )
    #TODO orchestra
    elif args.exp == "orchestra":
        s = args.strength
        target_size = args.target_size
        
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        if args.dataset == 'cifar':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(target_size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            
            train_dataset = datasets.CIFAR10(
                cifar_data_path, 
                train=True, 
                download=True,
                transform=OrchestraTransformWrapper(transform)
            )

            test_dataset = datasets.CIFAR10(
                cifar_data_path, 
                train=False, 
                download=True,
                transform=transform
            )
            
            warmup_dataset = datasets.CIFAR10(
                cifar_data_path, 
                train=False, 
                transform=OrchestraTransformWrapper(transform), 
                download=True
            )

            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_train_idxs = cifar_iid(
                    train_dataset, 
                    args.num_users, 
                    args.num_items
                )

            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose unequal splits for every user
                    user_train_idxs = cifar_noniid_unequal(
                        train_dataset, 
                        args.num_users,
                        args.num_items, 
                        args.alpha, 
                        args.num_class_per_client
                    )

                else:
                    # Chose equal splits for every user
                    user_train_idxs = cifar_noniid(
                        train_dataset, 
                        args.num_users, 
                        args.num_items, 
                        args.alpha,
                        args.num_class_per_client
                    )
        
        elif args.dataset == 'mnist':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(target_size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            
            train_dataset = datasets.MNIST(
                mnist_data_path, 
                train=True, 
                download=True,      
                transform=OrchestraTransformWrapper(transform)
            )

            test_dataset = datasets.MNIST(
                mnist_data_path, 
                train=False, 
                download=True,
                transform=transform
            )
             
            warmup_dataset = datasets.MNIST(
                cifar_data_path, 
                train=False, 
                transform=OrchestraTransformWrapper(transform), 
                download=True
            )

            # sample training data amongst users

            if args.iid:
                # Sample IID user data from Mnist
                user_train_idxs = mnist_iid(
                    train_dataset, 
                    args.num_users, 
                    args.num_items
                )
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose unequal splits for every user
                    user_train_idxs = mnist_noniid_unequal(
                        train_dataset, 
                        args.num_users, 
                        args.num_items, 
                        args.alpha, 
                        args.num_class_per_client
                    )
                else:
                    # Chose equal splits for every user
                    user_train_idxs = mnist_noniid(
                        train_dataset, 
                        args.num_users, 
                        args.num_items, 
                        args.alpha, 
                        args.num_class_per_client
                    )
    
    return train_dataset, test_dataset, warmup_dataset, user_train_idxs


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0]) # this would be on cuda:0 
    for key in w_avg.keys():
        for i in range(1, len(w)):
            if w_avg[key].get_device() != w[i][key].get_device():
                w[i][key] = w[i][key].to(torch.device("cuda:0"))
            w_avg[key] += w[i][key]
            
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args, writer):
    print('Experimental details:')
    print(f'    Seed            : {args.seed}')
    print(f'    Dataset         : {args.dataset}')
    print(f'    Model           : {args.model}')
    print(f'    Pretrained      : {args.pretrained}')
    print(f'    Optimizer       : {args.optimizer}')
    print(f'    Learning rate   : {args.lr}')
    print(f'    Total Rounds    : {args.epochs}')
    print(f'    Alpha           : {args.alpha}')
    print(f'    Momentum        : {args.momentum}')
    print(f'    Weight decay    : {args.weight_decay}')
    print(f'    Sup Warmup      : {args.sup_warmup}')
    print(f'    Server data frac: {args.server_data_frac}')
    
    sb = ""
    sb = sb + "\nSeed " + str(args.seed)
    sb = sb + "\nDataset " + args.dataset
    sb = sb + "\nModel " + args.model
    sb = sb + "\nPretrained " + str(args.pretrained)
    sb = sb + "\nOptimizer " + args.optimizer
    sb = sb + "\nLR " + str(args.lr)
    sb = sb + "\nTotal rounds " + str(args.epochs)
    sb = sb + "\nAlpha " + str(args.alpha)
    sb = sb + "\nMomentum " + str(args.momentum)
    sb = sb + "\nWeight decay " + str(args.weight_decay)
    sb = sb + "\nExp " + args.exp
    sb = sb + "\nSup Warmup " + str(args.sup_warmup)
    sb = sb + "\nServer data frac " + str(args.server_data_frac)
    
    if args.exp == "simclr":
        print("SimCLR")
        print(f'    Warmup          : {args.warmup}')
        print(f'    Freeze          : {args.freeze}')
        print(f'    Adapt Epochs    : {args.adapt_epoch}')
        print(f'    Warmup Epochs   : {args.warmup_epochs}')
        print(f'    Warmup Batchsize: {args.warmup_bs}')
        print(f'    Temperature     : {args.temperature}')
        print(f'    Output dim      : {args.out_dim}')
        print(f'    N views         : {args.n_views}')

        sb = sb + "\nWarmup " +  str(args.warmup)
        sb = sb + "\nFreeze " +  str(args.freeze)
        sb = sb + "\nAdapt Epochs " +  str(args.adapt_epoch)
        sb = sb + "\nWarmup Epochs " +  str(args.warmup_epochs)
        sb = sb + "\nWarmup Batchsize " +  str(args.warmup_bs)
        sb = sb + "\nTemperature " +  str(args.temperature)
        sb = sb + "\nOutput dim " +  str(args.out_dim)
        sb = sb + "\nN views " +  str(args.n_views)
        
    elif args.exp == "simsiam":
        print("SimSiam")
        print(f'    Warmup          : {args.warmup}')
        print(f'    Freeze          : {args.freeze}')
        print(f'    Adapt Epochs    : {args.adapt_epoch}')
        print(f'    Warmup Epochs   : {args.warmup_epochs}')
        print(f'    Warmup Batchsize: {args.warmup_bs}')
        print(f'    Output dim      : {args.out_dim}')
        print(f'    Pred   dim      : {args.pred_dim}')
        
        sb = sb + "\nWarmup " +  str(args.warmup)
        sb = sb + "\nFreeze " +  str(args.freeze)
        sb = sb + "\nAdapt Epochs " +  str(args.adapt_epoch)
        sb = sb + "\nWarmup Epochs " +  str(args.warmup_epochs)
        sb = sb + "\nWarmup Batchsize " +  str(args.warmup_bs)
        sb = sb + "\nOutput dim " +  str(args.out_dim)
        sb = sb + "\nPred dim " +  str(args.pred_dim)
        
    
    elif args.exp == "FL":
        print("FL")
        print(f'    Warmup          : {args.warmup}')
        print(f'    Freeze          : {args.freeze}')
        print(f'    Adapt Epochs    : {args.adapt_epoch}')
        print(f'    Warmup Epochs   : {args.warmup_epochs}')
        print(f'    Warmup Batchsize: {args.warmup_bs}')
        
        sb = sb + "\nWarmup " + str(args.warmup)
        sb = sb + "\nFreeze " + str(args.freeze)
        sb = sb + "\nAdapt Epochs " + str(args.adapt_epoch)
        sb = sb + "\nWarmup Epochs " + str(args.warmup_epochs)
        sb = sb + "\nWarmup Batchsize " + str(args.warmup_bs)
        
        
    print('Federated parameters:')
    print(f'    Number of users                : {args.num_users}')
    print(f'    Fraction of users              : {args.frac}')
    print(f'    Number of train items per user : {args.num_items}')
    print(f'    Number of classes per user     : {args.num_class_per_client}')
    print(f'    Local Batch size               : {args.local_bs}')
    print(f'    Local Epochs                   : {args.local_ep}')
    print(f'    Checkpoint path                : {args.ckpt_path}')
    print(f'    Tensorboard log path           : {args.log_path}')
    sb = sb + "\nNum users " + str(args.num_users)
    sb = sb + "\nFrac client " + str(args.frac)
    sb = sb + "\nNum items per user " + str(args.num_items)
    sb = sb + "\nNum classes per user " + str(args.num_class_per_client)
    sb = sb + "\nLocal Batchsize " + str(args.local_bs)
    sb = sb + "\nLocal epochs " + str(args.local_ep)
    writer.add_text("Params", sb)
    writer.flush()
    
    
    