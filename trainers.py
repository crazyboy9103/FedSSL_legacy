import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import LinearEvalModel, SimSiamLinear
import copy 
import time
from tqdm import tqdm

class AverageMeter():
    def __init__(self, name):
        self.name = name
        self.values = []
    
    def update(self, value):
        self.values.append(value)
        
    def get_result(self):
        return sum(self.values)/len(self.values)
    
    def reset(self):
        self.values = []

class NormalFL():
    def __init__(self, args, model, train_loader, test_loader, warmup_loader, device, client_id):
        self.args = args
        self.model = model.to(device)
        
        self.local_epochs = args.local_ep
        optim_dict = {
            "sgd": torch.optim.SGD, 
            "adam": torch.optim.Adam
        }
        self.optimizer = optim_dict[args.optimizer](
            self.model.parameters(), 
            args.lr
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max = self.local_epochs, 
            eta_min = 0,
            last_epoch = -1
        )
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.warmup_loader = warmup_loader
        self.device = device
        self.client_id = client_id
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
    def train(self):
        best_loss = 999999
        best_top1 = -999999
        best_top5 = -999999
        best_model_state = None
        
        self.model.train()
        for epoch in range(self.local_epochs):
            # Metric
            running_loss = AverageMeter("loss")
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                preds = self.model(images)
                loss = self.criterion(preds, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss_value = loss.item()
                running_loss.update(loss_value)

                
            self.scheduler.step()
            
            # Train metrics
            avg_loss = running_loss.get_result()
            running_loss.reset()
            
            lr = self.optimizer.param_groups[0]['lr']
            if (epoch+1)%5 == 0:
                # Test metrics
                test_loss, test_top1, test_top5 = self.test()

                print(f"""Client {self.client_id} Epoch [{epoch}/{self.local_epochs}]:
                          learning rate : {lr:.6f}
                          test acc/top1 : {test_top1:.2f}%
                          test acc/top5 : {test_top5:.2f}%
                          test loss : {test_loss:.2f}""")

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_top1 = test_top1
                    best_top5 = test_top5

                    state_dict = {
                        "loss": test_loss, 
                        "top1": test_top1, 
                        "top5": test_top5,
                        "model": self.model.state_dict(), 
                        "optim": self.optimizer.state_dict()
                    }

                    best_model_state = state_dict

        print(f"Training complete best top1/top5: {best_top1:.2f}%/{best_top5:.2f}%")
        return best_model_state
    
    def test(self):
        eval_model = copy.deepcopy(self.model)
        eval_model.eval()
        
        N = len(self.test_loader)
        
        running_loss = AverageMeter("loss")
        running_top1 = AverageMeter("acc/top1")
        running_top5 = AverageMeter("acc/top5")
    
        for batch_idx, (images, labels) in enumerate(self.test_loader):
            with torch.no_grad():
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                preds = eval_model(images)

                loss = self.criterion(preds, labels)

                running_loss.update(loss)

                _, top1_preds = torch.max(preds.data, 1)

                _, top5_preds = torch.topk(preds.data, k=5, dim=-1)

                top1 = ((top1_preds == labels).sum().item() / labels.size(0)) * 100
                top5 = 0
                
                for label, pred in zip(labels, top5_preds):
                    if label in pred:
                        top5 += 1

                top5 /= labels.size(0)
                top5 *= 100
                    
                running_top1.update(top1)
                running_top5.update(top5)
                
        avg_loss = running_loss.get_result()
        avg_top1 = running_top1.get_result()
        avg_top5 = running_top5.get_result()
            
        return avg_loss, avg_top1, avg_top5
    
    def warmup(self, epochs):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=epochs,
            eta_min=0, 
            last_epoch=-1
        )
        best_loss = 999999
        best_top1 = -999999
        best_top5 = -999999
        best_model_state = None
        print("Warming up FL model")
        start = time.time()
        for warm_epoch in range(epochs):
            running_loss = AverageMeter("loss")
            for batch_idx, (images, _) in enumerate(self.warmup_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                preds = self.model(images)
                loss = self.criterion(preds, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss_value = loss.item()
                running_loss.update(loss_value)
            
            
            self.scheduler.step()
            # Train metrics
            avg_loss = running_loss.get_result()
            running_loss.reset()
            
            lr = self.optimizer.param_groups[0]['lr']
            
            if (warm_epoch+1) % 5 == 0:
                # Test metrics
                test_loss, test_top1, test_top5 = self.test()
                end = time.time()
                time_taken = end-start
                start = end
                print(f"""Client {self.client_id} Epoch [{warm_epoch}/{self.args.warmup_epochs}]:
                          learning rate : {lr:.6f}
                          test acc/top1 : {test_top1:.2f}%
                          test acc/top5 : {test_top5:.2f}%
                          test loss : {test_loss:.2f}
                          time taken : {time_taken:.2f} """)
            
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_top1 = test_top1
                    best_top5 = test_top5

                    state_dict = {
                        "loss": test_loss, 
                        "top1": test_top1, 
                        "top5": test_top5,
                        "model": self.model.state_dict(), 
                        "optim": self.optimizer.state_dict()
                    }

                    best_model_state = state_dict
            
        return best_model_state
    
class SimCLR():
    def __init__(self, args, model, train_loader, test_loader, warmup_loader, device, client_id):
        self.args = args
        self.temperature = args.temperature
        self.local_epochs = args.local_ep
        
        self.model = model.to(device)
        optim_dict = {
            "sgd": torch.optim.SGD, 
            "adam": torch.optim.Adam
        }
        self.optimizer = optim_dict[args.optimizer](
            self.model.parameters(), 
            args.lr
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.local_epochs,
            eta_min=0, 
            last_epoch=-1
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.warmup_loader = warmup_loader
        
        self.device = device
        self.client_id = client_id
        
        self.criterion = nn.CrossEntropyLoss(reduction="sum").to(self.device)
        
    def nce_loss(self, features):
        
        #cos = nn.CosineSimilarity(dim=2)
        features = F.normalize(features, dim=1)
        feature1, feature2 = torch.tensor_split(features, 2, 0)
        batch_size = feature1.shape[0]
        LARGE_NUM = 1e9
        
        labels = torch.arange(0, batch_size, device=self.device, dtype=torch.int64)
        masks = torch.eye(batch_size, device=self.device)
        
        
        logits_aa = torch.matmul(feature1, feature1.T) / self.temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        
        logits_bb = torch.matmul(feature2, feature2.T) / self.temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(feature1, feature2.T) / self.temperature
        logits_ba = torch.matmul(feature2, feature1.T) / self.temperature
        
        #print(logits_ab.shape)
        #print(logits_aa.shape)
        loss_a = self.criterion(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = self.criterion(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss= loss_a + loss_b
        return loss
        
    def train(self):
        print("Training simclr model")
        best_loss = 999999
        best_top1 = -999999
        best_top5 = -999999
        best_model_state = None
        
        self.model.train()
        start = time.time()
        for epoch in range(self.local_epochs):
            # Metric
            running_loss = AverageMeter("loss")
           
            
            for batch_idx, (images, _) in enumerate(self.train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.device)
                
                features = self.model(images)
                loss = self.nce_loss(features)
                
                loss_value = loss.item()
                
                running_loss.update(loss_value)

                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            self.scheduler.step()
            # Train metrics
            avg_loss = running_loss.get_result()
            running_loss.reset()
            
            lr = self.optimizer.param_groups[0]['lr']

            if (epoch+1) % 5 == 0:
                # Test metrics
                test_loss, test_top1, test_top5 = self.test()
                end = time.time()
                time_taken = end-start
                start = end
                print(f"""Client {self.client_id} Epoch [{epoch}/{self.local_epochs}]:
                          learning rate : {lr:.6f}
                          test acc/top1 : {test_top1:.2f}%
                          test acc/top5 : {test_top5:.2f}%
                          test loss : {test_loss:.2f}
                          time taken : {int(time_taken)} """)
            
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_top1 = test_top1
                    best_top5 = test_top5

                    state_dict = {
                        "loss": test_loss, 
                        "top1": test_top1, 
                        "top5": test_top5,
                        "model": self.model.state_dict(), 
                        "optim": self.optimizer.state_dict()
                    }

                    best_model_state = state_dict
                
        print(f"Training complete best top1/top5: {best_top1:.2f}%/{best_top5:.2f}%")
        return best_model_state
    
    def test(self):
        print("Linear evaluating simclr model")
        eval_model = LinearEvalModel(
            self.model.linear_eval_model(
                freeze=self.args.freeze, 
                num_classes=self.args.num_classes
            )
        )
        
        eval_model = eval_model.to(self.device)
        
        N = len(self.test_loader)
        optimizer = optim.Adam(
            eval_model.parameters(), 
            lr=0.001
        )
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        
        running_loss = AverageMeter("loss")
        running_top1 = AverageMeter("acc/top1")
        running_top5 = AverageMeter("acc/top5")
        
        for adapt_epoch in range(self.args.adapt_epoch):
            running_loss = AverageMeter("loss")
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                # Finetuning
                if batch_idx < int(0.8 * N):
                    eval_model.train()
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    preds = eval_model(images)
                    optimizer.zero_grad()
                    loss = criterion(preds, labels)
                    loss.backward()
                    optimizer.step()

                # Testing
                else:
                    eval_model.eval()
                    with torch.no_grad():

                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        preds = eval_model(images)

                        loss = criterion(preds, labels)

                        running_loss.update(loss)

                        _, top1_preds = torch.max(preds.data, 1)

                        _, top5_preds = torch.topk(preds.data, k=5, dim=-1)

                        top1 = ((top1_preds == labels).sum().item() / labels.size(0)) * 100
                        top5 = 0
                        for label, pred in zip(labels, top5_preds):
                            if label in pred:
                                top5 += 1

                        top5 /= labels.size(0)
                        top5 *= 100
                    
                        running_top1.update(top1)
                        running_top5.update(top5)
                
            avg_loss = running_loss.get_result()
            avg_top1 = running_top1.get_result()
            avg_top5 = running_top5.get_result()
            
        return avg_loss, avg_top1, avg_top5
    
    def warmup(self, epochs):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=epochs,
            eta_min=0, 
            last_epoch=-1
        )
        best_loss = 999999
        best_top1 = -999999
        best_top5 = -999999
        best_model_state = None
        print("Warming up simclr model")
        start = time.time()
        for warm_epoch in range(epochs):
            running_loss = AverageMeter("loss")
            for batch_idx, (images, _) in enumerate(self.warmup_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.device)
                
                features = self.model(images)
                loss = self.nce_loss(features)
                
                loss_value = loss.item()
                
                running_loss.update(loss_value)

                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            
            self.scheduler.step()
            # Train metrics
            avg_loss = running_loss.get_result()
            running_loss.reset()
            
            lr = self.optimizer.param_groups[0]['lr']
            
            if (warm_epoch+1) % 5 == 0:
                # Test metrics
                test_loss, test_top1, test_top5 = self.test()
                end = time.time()
                time_taken = end-start
                start = end
                print(f"""Client {self.client_id} Epoch [{warm_epoch}/{self.args.warmup_epochs}]:
                          learning rate : {lr:.6f}
                          test acc/top1 : {test_top1:.2f}%
                          test acc/top5 : {test_top5:.2f}%
                          test loss : {test_loss:.2f}
                          time taken : {time_taken:.2f} """)
            
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_top1 = test_top1
                    best_top5 = test_top5

                    state_dict = {
                        "loss": test_loss, 
                        "top1": test_top1, 
                        "top5": test_top5,
                        "model": self.model.state_dict(), 
                        "optim": self.optimizer.state_dict()
                    }

                    best_model_state = state_dict
            
        return best_model_state

class SimSiam():
    def __init__(self, args, model, train_loader, test_loader, warmup_loader, device, client_id):
        self.args = args
        self.local_epochs = args.local_ep
        
        self.model = model.to(device)
        optim_dict = {
            "sgd": torch.optim.SGD, 
            "adam": torch.optim.Adam
        }
        self.optimizer = optim_dict[args.optimizer](
            self.model.parameters(), 
            args.lr
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.local_epochs,
            eta_min=0, 
            last_epoch=-1
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.warmup_loader = warmup_loader
        
        self.device = device
        self.client_id = client_id
        
        self.criterion = nn.CosineSimilarity(dim=1).to(self.device)
    
    def simsiam_loss(self, p1, p2, z1, z2):
        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        return loss
    
    def train(self):
        print("Training simsiam model")
        best_loss = 999999
        best_top1 = -999999
        best_top5 = -999999
        best_model_state = None
        
        self.model.train()
        start = time.time()
        for epoch in range(self.local_epochs):
            # Metric
            running_loss = AverageMeter("loss")
           
            
            for batch_idx, (images, _) in enumerate(self.train_loader):
                images[0] = images[0].to(self.device)
                images[1] = images[1].to(self.device)
                p1, p2, z1, z2 = self.model(images[0], images[1]) 
                loss = self.simsiam_loss(p1, p2, z1, z2)
                loss_value = loss.item()
                
                running_loss.update(loss_value)

                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            self.scheduler.step()
            # Train metrics
            avg_loss = running_loss.get_result()
            running_loss.reset()
            
            lr = self.optimizer.param_groups[0]['lr']

            if (epoch+1) % 5 == 0:
                # Test metrics
                test_loss, test_top1, test_top5 = self.test()
                end = time.time()
                time_taken = end-start
                start = end
                print(f"""Client {self.client_id} Epoch [{epoch}/{self.local_epochs}]:
                          learning rate : {lr:.6f}
                          test acc/top1 : {test_top1:.2f}%
                          test acc/top5 : {test_top5:.2f}%
                          test loss : {test_loss:.2f}
                          time taken : {int(time_taken)} """)
            
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_top1 = test_top1
                    best_top5 = test_top5

                    state_dict = {
                        "loss": test_loss, 
                        "top1": test_top1, 
                        "top5": test_top5,
                        "model": self.model.state_dict(), 
                        "optim": self.optimizer.state_dict()
                    }

                    best_model_state = state_dict
                
        print(f"Training complete best top1/top5: {best_top1:.2f}%/{best_top5:.2f}%")
        return best_model_state
    
    def test(self):
        print("Linear evaluating simsiam model")
        eval_model = SimSiamLinear(
            trained_encoder = self.model.encoder,
            freeze = self.args.freeze, 
            num_classes = self.args.num_classes
        )
        
        eval_model = eval_model.to(self.device)
        
        N = len(self.test_loader)
        optimizer = optim.Adam(
            eval_model.parameters(), 
            lr=0.001
        )
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        
        running_loss = AverageMeter("loss")
        running_top1 = AverageMeter("acc/top1")
        running_top5 = AverageMeter("acc/top5")
        
        for adapt_epoch in range(self.args.adapt_epoch):
            running_loss = AverageMeter("loss")
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                # Finetuning
                if batch_idx < int(0.8 * N):
                    eval_model.train()
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    preds = eval_model(images)
                    optimizer.zero_grad()
                    loss = criterion(preds, labels)
                    loss.backward()
                    optimizer.step()

                # Testing
                else:
                    eval_model.eval()
                    with torch.no_grad():

                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        preds = eval_model(images)

                        loss = criterion(preds, labels)

                        running_loss.update(loss)

                        _, top1_preds = torch.max(preds.data, 1)

                        _, top5_preds = torch.topk(preds.data, k=5, dim=-1)

                        top1 = ((top1_preds == labels).sum().item() / labels.size(0)) * 100
                        top5 = 0
                        for label, pred in zip(labels, top5_preds):
                            if label in pred:
                                top5 += 1

                        top5 /= labels.size(0)
                        top5 *= 100
                    
                        running_top1.update(top1)
                        running_top5.update(top5)
                
            avg_loss = running_loss.get_result()
            avg_top1 = running_top1.get_result()
            avg_top5 = running_top5.get_result()
            
        return avg_loss, avg_top1, avg_top5
    
    def warmup(self, epochs):
        self.model.train()
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=epochs,
            eta_min=0, 
            last_epoch=-1
        )
        best_loss = 999999
        best_top1 = -999999
        best_top5 = -999999
        best_model_state = None
        print("Warming up simsiam model")
        start = time.time()
        
        for warm_epoch in range(epochs):
            running_loss = AverageMeter("loss")
            for batch_idx, (images, _) in enumerate(self.warmup_loader):
                images[0] = images[0].to(self.device)
                images[1] = images[1].to(self.device)
                p1, p2, z1, z2 = self.model(images[0], images[1]) 
                loss = self.simsiam_loss(p1, p2, z1, z2)
            
                loss_value = loss.item()
                
                running_loss.update(loss_value)

                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            
            self.scheduler.step()
            # Train metrics
            avg_loss = running_loss.get_result()
            running_loss.reset()
            
            lr = self.optimizer.param_groups[0]['lr']
            
            if (warm_epoch+1) % 5 == 0:
                # Test metrics
                test_loss, test_top1, test_top5 = self.test()
                end = time.time()
                time_taken = end-start
                start = end
                print(f"""Client {self.client_id} Epoch [{warm_epoch}/{self.args.warmup_epochs}]:
                          learning rate : {lr:.6f}
                          test acc/top1 : {test_top1:.2f}%
                          test acc/top5 : {test_top5:.2f}%
                          test loss : {test_loss:.2f}
                          time taken : {time_taken:.2f} """)
            
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_top1 = test_top1
                    best_top5 = test_top5

                    state_dict = {
                        "loss": test_loss, 
                        "top1": test_top1, 
                        "top5": test_top5,
                        "model": self.model.state_dict(), 
                        "optim": self.optimizer.state_dict()
                    }

                    best_model_state = state_dict
            
        return best_model_state