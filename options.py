#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import os
import time, datetime

def str2bool(v):
    #https://eehoeskrap.tistory.com/521
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getTimestamp():
    #timezone=60*60*9
    utc_timestamp = int(time.time())#+timezone)
    date = datetime.datetime.fromtimestamp(utc_timestamp).strftime("%Y_%m_%d_%H_%M_%S")
    return date

def args_parser():
    parser = argparse.ArgumentParser()
    
    # model arguments
    parser.add_argument('--model',       type=str,  default='resnet18', help='resnet18|resnet50|vgg16')
    parser.add_argument('--pretrained',  type=str2bool, default=False,       help='pretrained backbone')
    parser.add_argument('--num_classes', type=int,  default=10,         help="number of classes")
    
    # Experimental setup
    parser.add_argument("--exp",           type=str,   default="simclr",  help="FL|simclr|simsiam")
    parser.add_argument("--temperature",   type=float, default=0.1,       help="softmax temperature")
    parser.add_argument("--alpha",         type=float, default=0.9,       help="dirichlet param 0<alpha<1 controls iidness")
    parser.add_argument('--adapt_epoch',   type=int,   default=10,        help="adaptation epochs")
    parser.add_argument('--strength',      type=float, default=0.8,       help="augmentation strength 0<s<1")
    parser.add_argument('--target_size',   type=int,   default=32,        help="augmentation target width (=height)")
    parser.add_argument('--out_dim',       type=int,   default=512,       help="output dimension of the feature for simclr and simsiam")
    parser.add_argument('--freeze',        type=str2bool,  default=False,      help='freeze feature extractor during linear eval')
    parser.add_argument('--n_views',       type=int,   default=2,         help="default simclr n_views=2")
    parser.add_argument('--warmup_bs',     type=int,   default=512,       help="warmup batch size")
    parser.add_argument('--warmup_epochs', type=int,   default=30,        help="warmup epochs")
    parser.add_argument("--pred_dim",      type=int,   default=256,       help="pred dim for simsiam")
    parser.add_argument("--warmup",        type=str2bool,  default=False,      help="warmup at init")
    parser.add_argument("--sup_warmup",    type=str2bool,  default=True,      help="supervised warmup")
    
    parser.add_argument("--ema",           type=float, default=0.996,     help="ema value for target net in orchestra")
    parser.add_argument("--n_glob_clusters",       type=int,  default=128,      help="number of global clusters in orchestra")
    parser.add_argument("--n_loc_clusters",        type=int,  default=16,       help="number of local clusters in orchestra")
    parser.add_argument("--m_size",        type=int,  default=128,      help="Memory size per client in orchestra")
    
    # FL
    parser.add_argument("--num_users",            type=int,    default=100,        help="num users")
    parser.add_argument("--num_items",            type=int,    default=500,        help="num data each client holds")
    parser.add_argument("--num_class_per_client", type=int,    default=5,          help="num classes each client holds")
    parser.add_argument('--unequal',              type=str2bool,   default=False,      help='unequal num of data')
    parser.add_argument('--epochs',               type=int,    default=100,        help="number of rounds of training")
    parser.add_argument('--frac',                 type=float,  default=0.05,       help='the fraction of clients: C')
    parser.add_argument('--local_ep',             type=int,    default=10,         help="the number of local epochs: E")
    parser.add_argument('--local_bs',             type=int,    default=32,         help="local batch size")
    parser.add_argument('--lr',                   type=float,  default=0.001,      help='learning rate')
    parser.add_argument('--momentum',             type=float,  default=0.9,        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay',         type=float,  default=1e-4,       help='weight decay (default: 1e-4)')
    
    # Train setting
    parser.add_argument("--parallel",    type=str2bool,  default=False,                 help="parallel training with threads")
    parser.add_argument("--num_workers", type=int,   default=8,                    help="num workers for dataloader")
    parser.add_argument('--log_path',    type=str,   default='./logs',             help="tensorboard log dir")
    parser.add_argument('--dataset',     type=str,   default='cifar',              help="mnist|cifar")
    parser.add_argument('--optimizer',   type=str,   default='sgd',                help="type of optimizer")
    parser.add_argument('--seed',        type=int,   default=2022,                 help='random seed')
    parser.add_argument('--ckpt_path',   type=str,   default="checkpoint.pth.tar", help="model ckpt save path")
    parser.add_argument('--data_path',   type=str,   default="./data",             help="path to dataset")
    parser.add_argument('--train_device', type=str,  default="0",                  help="gpu device number for train")
    parser.add_argument('--test_device', type=str,   default="0",                  help="gpu device number for test")
    

    args = parser.parse_args()
    
    # if no experiment is specified on path 
    if not args.log_path.split("logs")[-1]: 
        args.log_path = os.path.join(args.log_path, getTimestamp())
    
    args.iid = args.alpha >= 0.9
    assert args.num_class_per_client <= args.num_classes
    return args