import numpy as np
import torch
import copy
from client.client_base import Client_Base
from torch.optim import Optimizer
import utils
import time


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


class Client_pFedSara(Client_Base):
    def __init__(self, args, clientINFO, **kwargs):
        super().__init__(args, clientINFO, **kwargs)
        self.threshold = 0.0
        self.finetune_lr = args.finetune_lr
        self.tau = -1
        self.timelimit = args.timelimit
        self.optimizer = PerFedAvgOptimizer(self.model.parameters(), finetune_lr=self.finetune_lr)

    def train_sgd(self):
        dataLoader = self.get_dataloader(train=True)
        model_nouse = copy.deepcopy(self.model)
        optim_nouse = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        optim = torch.optim.SGD(model_nouse.parameters(), lr=self.lr)
        self.model.train()
        model_nouse.train()
        for epoch in range(self.epochs):
            for images, labels in dataLoader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
            
            for _ in range(self.heterogeneity):
                for images, labels in dataLoader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model_nouse(images)
                    loss = self.criterion(outputs, labels)
                    optim_nouse.zero_grad()
                    loss.backward()
                    optim_nouse.step()
            
    def train_maml(self, estimate=False):
        model_nouse = copy.deepcopy(self.model)
        optimizer = PerFedAvgOptimizer(self.model.parameters(), finetune_lr=self.finetune_lr)
        optimizer_nouse = PerFedAvgOptimizer(model_nouse.parameters(), finetune_lr=self.finetune_lr)
        self.model.train()
        model_nouse.train()
        dataLoader = self.get_dataloader(train=True)
        dataLoader_inner = self.get_dataloader(train=True)

        if estimate:
            self.tau = self.epochs
            starttime = time.time()

        for epoch in range(self.tau):
            for images1, labels1 in dataLoader:
                current_model = copy.deepcopy(list(self.model.parameters()))
                images1 = images1.to(self.device)
                labels1 = labels1.to(self.device)
                outputs = self.model(images1)
                loss = self.criterion(outputs, labels1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                images2, labels2 = next(iter(dataLoader_inner))
                images2 = images2.to(self.device)
                labels2 = labels2.to(self.device)
                output = self.model(images2)
                loss = self.criterion(output, labels2)
                optimizer.zero_grad()
                loss.backward()
                utils.clone_model(current_model, self.model)
                optimizer.step(eta=self.lr)
            
            for _ in range(self.heterogeneity):
                for images1, labels1 in dataLoader:
                    current_model = copy.deepcopy(list(model_nouse.parameters()))
                    images1 = images1.to(self.device)
                    labels1 = labels1.to(self.device)
                    outputs = model_nouse(images1)
                    loss = self.criterion(outputs, labels1)
                    optimizer_nouse.zero_grad()
                    loss.backward()
                    optimizer_nouse.step()

                    images2, labels2 = next(iter(dataLoader_inner))
                    images2 = images2.to(self.device)
                    labels2 = labels2.to(self.device)
                    output = model_nouse(images2)
                    loss = self.criterion(output, labels2)
                    optimizer_nouse.zero_grad()
                    loss.backward()
                    utils.clone_model(current_model, model_nouse)
                    optimizer_nouse.step(eta=self.lr)
        if estimate:
            interal = time.time() - starttime
            self.tau = min(int(self.timelimit / interal * self.epochs), 5)

    def train(self):
        dataLoader = self.get_dataloader(train=True)
        dataLoader_inner = self.get_dataloader(train=True)
        model_ini = utils.get_flat_model_params(self.model)
        self.model.train()
        images1, labels1 = next(iter(dataLoader))
        current_model = copy.deepcopy(list(self.model.parameters()))
        images1 = images1.to(self.device)
        labels1 = labels1.to(self.device)
        outputs = self.model(images1)
        loss = self.criterion(outputs, labels1)
        self.optimizer.zero_grad()
        loss.backward()
        grad_sgd = utils.get_flat_grad(self.model)
        self.optimizer.step()

        images2, labels2 = next(iter(dataLoader_inner))
        images2 = images2.to(self.device)
        labels2 = labels2.to(self.device)
        output = self.model(images2)
        loss = self.criterion(output, labels2)
        self.optimizer.zero_grad()
        loss.backward()
        grad_maml = utils.get_flat_grad(self.model)
        self.update_parameters(current_model)
        self.optimizer.step(eta=self.lr)
        
        self.similarity = torch.nn.functional.cosine_similarity(grad_maml.unsqueeze(0), grad_sgd.unsqueeze(0))
        if self.similarity >= self.threshold and self.tau < self.epochs:
            self.is_app = True
            self.train_sgd()
        else:
            self.is_app = False
            self.train_maml()
            reweight_model = self.epochs / self.tau * (utils.get_flat_model_params(self.model) - model_ini) + model_ini
            utils.set_flat_model_params(self.model, reweight_model)

        #updates_avg = utils.get_flat_model_params(model_avg) - model_ini
        #updates_per = utils.get_flat_model_params(self.model) - model_ini
        #self.similarity = torch.nn.functional.cosine_similarity(updates_avg.unsqueeze(0), updates_per.unsqueeze(0))
        #self.distance = torch.norm(updates_avg - updates_per)
        #self.norm_gap = torch.norm(updates_per) / torch.norm(updates_avg)

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


