# sampling.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users, num_items):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    assert (len(dataset) // num_users) > num_items, f"num_items must be smaller than {len(dataset) // num_users}"
    #num_items = int(len(dataset)/num_users) # Equal amounts 
    dict_users, all_idxs = {}, set([i for i in range(len(dataset))])
    for i in range(num_users):
        idxs_set = set(np.random.choice(list(all_idxs), num_items, replace=False))
        dict_users[i] = list(idxs_set)
        all_idxs = all_idxs - idxs_set
    return dict_users


def mnist_noniid(dataset, num_users, num_items, alpha, num_class_per_client):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    assert (len(dataset) // num_users) > num_items, f"num_items must be smaller than {len(dataset) // num_users}"
    assert num_class_per_client <= 10, f"num_class_per_client must be smaller than/eq to 10" # Mnist
    
    labels = dataset.targets.tolist()
    
    idxs_labels = {i:set() for i in range(10)} # 10 labels
    for idx, label in enumerate(labels):
        idxs_labels[label].add(idx)
    
    # each client's class distribution
    # MNIST has 10 labels
    class_dist = np.random.dirichlet(alpha=[alpha for _ in range(10)], size=num_users)
    
    # TODO: change this line to make it unbalanced
    num_items_per_class = num_items // num_class_per_client 
    dict_users = {i: set() for i in range(num_users)}
    
    for client_id, client_dist in enumerate(class_dist):
        classes = np.random.choice(list(range(10)), p=client_dist, size=num_class_per_client, replace=False)
        
        for _class in classes:
            class_idxs = idxs_labels[_class]
            
            # randomly samples num_items_per_class for each selected class
            data_idxs = set(np.random.choice(list(class_idxs), size=num_items_per_class, replace=False)) 
            
            # accumulate
            # temp = dict_users[client_id]
            # temp = temp.add(data_idxs)
            # dict_users[client_id] = temp
            dict_users[client_id].update(data_idxs)
            
            # exclude assigned idxs
            idxs_labels[_class] = class_idxs - data_idxs
    
    for i, data_idxs in dict_users.items():
        dict_users[i] = list(data_idxs)
    
    return dict_users


def mnist_noniid_unequal(dataset, num_users, num_items, alpha, num_class_per_client):
    labels = dataset.targets.tolist()
    
    idxs_labels = {i:set() for i in range(10)} # 10 labels
    for idx, label in enumerate(labels):
        idxs_labels[label].add(idx)
    
    # each client's class distribution
    # MNIST has 10 labels
    class_dist = np.random.dirichlet(alpha=[alpha for _ in range(10)], size=num_users)
    
    # TODO: change this line to make it unbalanced
    dict_users = {i: set() for i in range(num_users)}
    
    num_items_per_class_dist = num_items * np.random.dirichlet(alpha=[alpha for _ in range(10)], size=num_users)
    num_items_per_class_dist = num_items_per_class_dist.astype(int)
    
    for client_id, client_dist in enumerate(class_dist):
        classes = np.random.choice(list(range(10)), p=client_dist, size=num_class_per_client, replace=False)
        
        for _class in classes:
            class_idxs = idxs_labels[_class]
            
            # randomly samples num_items_per_class for each selected class
            num_items_per_class = num_items_per_class_dist[client_id][_class]
            data_idxs = set(np.random.choice(list(class_idxs), size=num_items_per_class, replace=False)) 
            
            # accumulate
            dict_users[client_id].update(data_idxs)
            
            # exclude assigned idxs
            idxs_labels[_class] = class_idxs - data_idxs
    
    for i, data_idxs in dict_users.items():
        dict_users[i] = list(data_idxs)
    
    return dict_users


def cifar_iid(dataset, num_users, num_items):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    assert (len(dataset) // num_users) >= num_items, f"num_items must be smaller than {len(dataset) // num_users}"
    #num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, set([i for i in range(len(dataset))])
    for i in range(num_users):
        idxs_set = set(np.random.choice(list(all_idxs), num_items, replace=False))
        dict_users[i] = list(idxs_set)
        all_idxs = all_idxs - idxs_set
    return dict_users


def cifar_noniid(dataset, num_users, num_items, alpha, num_class_per_client):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    labels = dataset.targets
    
    idxs_labels = {i:set() for i in range(10)} # 10 labels
    for idx, label in enumerate(labels):
        idxs_labels[label].add(idx)
    
    # each client's class distribution
    # MNIST has 10 labels
    class_dist = np.random.dirichlet(alpha=[alpha for _ in range(10)], size=num_users)
    
    # TODO: change this line to make it unbalanced
    num_items_per_class = num_items // num_class_per_client 
    dict_users = {i: set() for i in range(num_users)}
    
    for client_id, client_dist in enumerate(class_dist):
        classes = np.random.choice(list(range(10)), p=client_dist, size=num_class_per_client, replace=False)
        
        for _class in classes:
            class_idxs = idxs_labels[_class]
            
            # randomly samples num_items_per_class for each selected class
            data_idxs = set(np.random.choice(list(class_idxs), size=num_items_per_class, replace=False)) 
            
            # accumulate
            dict_users[client_id].update(data_idxs)
            
            # exclude assigned idxs
            idxs_labels[_class] = class_idxs - data_idxs
    
    for i, data_idxs in dict_users.items():
        dict_users[i] = list(data_idxs)
    return dict_users

def cifar_noniid_unequal(dataset, num_users, num_items, alpha, num_class_per_client):
    labels = dataset.targets.tolist()
    
    idxs_labels = {i:set() for i in range(10)} # 10 labels
    for idx, label in enumerate(labels):
        idxs_labels[label].add(idx)
    
    # each client's class distribution
    # MNIST has 10 labels
    class_dist = np.random.dirichlet(alpha=[alpha for _ in range(10)], size=num_users)
    
    # change this line to make it unbalanced
    dict_users = {i: set() for i in range(num_users)}
    
    num_items_per_class_dist = num_items * np.random.dirichlet(alpha=[alpha for _ in range(10)], size=num_users)
    num_items_per_class_dist = num_items_per_class_dist.astype(int)
    
    for client_id, client_dist in enumerate(class_dist):
        classes = np.random.choice(list(range(10)), p=client_dist, size=num_class_per_client, replace=False)
        
        for _class in classes:
            class_idxs = idxs_labels[_class]
            
            # randomly samples num_items_per_class for each selected class
            num_items_per_class = num_items_per_class_dist[client_id][_class]
            data_idxs = set(np.random.choice(list(class_idxs), size=num_items_per_class, replace=False)) 
            
            # accumulate
            dict_users[client_id].update(data_idxs)
            
            # exclude assigned idxs
            idxs_labels[_class] = class_idxs - data_idxs
    
    for i, data_idxs in dict_users.items():
        dict_users[i] = list(data_idxs)
    
    return dict_users