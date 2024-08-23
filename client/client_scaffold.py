import torch
from client.client_base import Client_Base
from torch.optim import Optimizer
import utils
import copy


class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, global_c, client_c):
        for group in self.param_groups:
            for param, gc, cc in zip(group['params'], global_c, client_c):
                param.data.add_(other=(param.grad.data + gc - cc), alpha=-group['lr'])


class Client_Scaffold(Client_Base):
    def __init__(self, args, clientINFO, **kwargs):
        super().__init__(args, clientINFO, **kwargs)
        self.optimizer = ScaffoldOptimizer(self.model.parameters(), lr=self.lr)
        self.client_c = []
        for param in self.model.parameters():
            self.client_c.append(torch.zeros_like(param))
        self.coff = self.batch_size/(self.dataVolume*self.epochs*self.lr)
        self.global_c = None
        self.global_model = None
        self.delta_x = None
        self.delta_c = None
            
    def set_parameters(self, model, global_c):
        utils.clone_model(model, self.model)
        self.global_c = global_c
        self.global_model = model

    def update_model(self, model, iteration):
        optimizer = ScaffoldOptimizer(model.parameters(), lr=self.lr)
        dataLoader = self.get_dataloader(train=True)
        model.train()
        for epoch in range(iteration):
            for images, labels in dataLoader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(self.global_c, self.client_c)

    def train(self):
        self.update_model(self.model, self.epochs)

        for cc, gc, gm, cm in zip(self.client_c, self.global_c, self.global_model.parameters(), self.model.parameters()):
            cc.data = cc - gc + self.coff * (gm - cm)
        
        self.delta_x = []
        self.delta_c = []
        for gc, gm, cm in zip(self.global_c, self.global_model.parameters(), self.model.parameters()):
            self.delta_x.append(cm - gm)
            self.delta_c.append(- gc + self.coff * (gm - cm))

        self.update_model(copy.deepcopy(self.model), self.epochs * self.heterogeneity)

    

