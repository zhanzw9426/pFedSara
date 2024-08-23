import numpy as np
import copy
import torch
from client.client_base import Client_Base

class Client_FedAvg(Client_Base):
    def __init__(self, args, clientINFO, **kwargs):
        super().__init__(args, clientINFO, **kwargs)
    
    def update_model(self, model, iteration):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
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
                optimizer.step()

    def train(self):
        self.update_model(self.model, self.epochs)
        self.update_model(copy.deepcopy(self.model), self.epochs * self.heterogeneity)
