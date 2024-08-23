import numpy as np
import torch
import copy
from client.client_base import Client_Base
from torch.optim import Optimizer
import utils

class PerFedAvgOptimizer(Optimizer):
    def __init__(self, params, finetune_lr):
        defaults = dict(lr=finetune_lr)
        super().__init__(params, defaults)

    def step(self, eta=0):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if eta != 0:
                    param.data.add_(other=grad, alpha=-eta)
                else:
                    param.data.add_(other=grad, alpha=-group['lr'])


class Client_PerFedAvg(Client_Base):
    def __init__(self, args, clientINFO, **kwargs):
        super().__init__(args, clientINFO, **kwargs)
        self.finetune_lr = args.finetune_lr

    def evaluate_importance(self, model):
        model = copy.deepcopy(model)
        loader = self.get_dataloader(train=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.finetune_lr)
        model.train()
        images, labels = next(iter(loader))
        images = images.to(self.device)
        labels = labels.to(self.device)
        output = model(images)
        loss = self.criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        total = 0
        loss = 0
        with torch.no_grad():
            for images, labels in self.get_dataloader(train=True):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                total += labels.shape[0]
                loss += loss.item() * labels.shape[0]
        loss = loss / total
        return loss
    
    def update_model(self, model, iteration):
        optimizer = PerFedAvgOptimizer(model.parameters(), finetune_lr=self.finetune_lr)
        dataLoader = self.get_dataloader(train=True)
        model.train()

        for epoch in range(iteration):
            dataLoader_inner_iter = iter(self.get_dataloader(train=True))
            for images1, labels1 in dataLoader:
                current_model = copy.deepcopy(list(model.parameters()))
                images1 = images1.to(self.device)
                labels1 = labels1.to(self.device)
                outputs = model(images1)
                loss = self.criterion(outputs, labels1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                images2, labels2 = next(dataLoader_inner_iter)
                images2 = images2.to(self.device)
                labels2 = labels2.to(self.device)
                output = model(images2)
                loss = self.criterion(output, labels2)
                optimizer.zero_grad()
                loss.backward()
                utils.clone_model(current_model, model)
                optimizer.step(eta=self.lr)

    def train(self):
        self.update_model(self.model, self.epochs)
        self.update_model(copy.deepcopy(self.model), self.epochs * self.heterogeneity)
            
            


