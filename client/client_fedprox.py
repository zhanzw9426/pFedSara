import copy
import torch
from client.client_base import Client_Base
from torch.optim import Optimizer


class ProxOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        defaults = dict(lr=lr, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for param, gparam in zip(group['params'], global_params):
                if param.grad is None:
                    continue
                gparam = gparam.to(device)
                grad = param.grad.data + group['mu'] * (param.data - gparam.data)
                param.data.add_(other=grad, alpha=-group['lr'])


class Client_FedProx(Client_Base):
    def __init__(self, args, clientINFO, **kwargs):
        super().__init__(args, clientINFO, **kwargs)
        self.mu = args.mu
        self.optimizer = ProxOptimizer(self.model.parameters(), lr=self.lr, mu=self.mu)

    def update_model(self, model, iteration):
        global_params = copy.deepcopy(list(model.parameters()))
        optimizer = ProxOptimizer(model.parameters(), lr=self.lr, mu=self.mu)
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
                optimizer.step(global_params, self.device)

    def train(self):
        self.update_model(self.model, self.epochs)
        self.update_model(copy.deepcopy(self.model), self.epochs * self.heterogeneity)
