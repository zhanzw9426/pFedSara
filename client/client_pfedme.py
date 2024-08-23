import numpy as np
import copy
import torch
import utils
from client.client_base import Client_Base
from torch.optim import Optimizer


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.1, lamda=0.01, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_model):
        for group in self.param_groups:
            for param, localweight in zip(group['params'], local_model):
                grad = param.grad.data + group['lamda'] * (param.data - localweight.data) + group['mu'] * param.data
                param.data = param.data - group['lr'] * grad
        return group['params']


class Client_pFedMe(Client_Base):
    def __init__(self, args, clientINFO, **kwargs):
        super().__init__(args, clientINFO, **kwargs)
        self.K = 5
        self.lamda = 1
        self.lr = args.lr
        self.model = self.model.to(self.device)
        self.theta_params = copy.deepcopy(list(self.model.parameters()))
        self.inner_lr = 0.1
    
    def update_model(self, model, iteration):
        optimizer = pFedMeOptimizer(model.parameters(), self.inner_lr, self.lamda)
        self.w_params = copy.deepcopy(list(model.parameters()))
        dataLoader = self.get_dataloader(train=True)
        model.train()
        for epoch in range(iteration):
            for images, labels in dataLoader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                for i in range(self.K):
                    output = model(images)
                    loss = self.criterion(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    self.theta_params = optimizer.step(self.w_params)

                for thetaweight, localweight in zip(self.theta_params, self.w_params):
                    localweight.data = localweight.data - self.lamda * self.lr * (localweight.data - thetaweight.data)

    def train(self):
        self.update_model(self.model, self.epochs)
        utils.clone_model(self.w_params, self.model)
        self.update_model(copy.deepcopy(self.model), self.epochs * self.heterogeneity)
        self.inner_lr *= 0.95

