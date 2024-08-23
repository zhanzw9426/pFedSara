import torch
import numpy as np
import copy
from client.client_base import Client_Base


class Client_FedLC(Client_Base):
    def __init__(self, args, clientINFO, **kwargs):
        super().__init__(args, clientINFO, **kwargs)
        self.sample_per_class = torch.zeros(self.num_classes).to(self.device)
        dataLoader = self.get_dataloader(train=True)
        for images, labels in dataLoader:
            labels = labels.to(self.device)
            self.sample_per_class += torch.bincount(labels, minlength=self.num_classes)
        self.calibration = None

    def update_model(self, model, iteration):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        dataLoader = self.get_dataloader(train=True)
        model.train()
        for epoch in range(iteration):
            for images, labels in dataLoader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs - self.calibration, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def train(self):
        self.update_model(self.model, self.epochs)
        self.update_model(copy.deepcopy(self.model), self.epochs * self.heterogeneity)
