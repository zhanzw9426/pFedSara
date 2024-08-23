import torch
import numpy as np
import utils
import copy
from client.client_base import Client_Base


class Client_FedRep(Client_Base):
    def __init__(self, args, clientINFO, **kwargs):
        super().__init__(args, clientINFO, **kwargs)
            
    def set_parameters(self, model):
        utils.clone_model(model, self.model.extractor)

    def update_model(self, model, iteration):
        optimizer_extractor = torch.optim.SGD(model.extractor.parameters(), lr=self.lr)
        optimizer_classifier = torch.optim.SGD(model.classifier.parameters(), lr=self.lr)
        dataLoader = self.get_dataloader(train=True)
        for epoch in range(iteration):
            for images, labels in dataLoader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                optimizer_classifier.zero_grad()
                optimizer_extractor.zero_grad()
                loss.backward()
                optimizer_classifier.step()
        
        for _ in range(self.heterogeneity):
            for images, labels in dataLoader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                optimizer_classifier.zero_grad()
                optimizer_extractor.zero_grad()
                loss.backward()
                optimizer_extractor.step()
    
    def train(self):
        self.update_model(self.model, self.epochs)
        self.update_model(copy.deepcopy(self.model), self.epochs * self.heterogeneity)