import copy
import torch
from client.client_base import Client_Base
import utils

class Client_FedBABU(Client_Base):
    def __init__(self, args, clientINFO, **kwargs):
        super().__init__(args, clientINFO, **kwargs)

    def set_parameters(self, model):
        utils.clone_model(model, self.model.extractor)

    def update_model(self, model, iteration):
        optimizer_extractor = torch.optim.SGD(model.extractor.parameters(), lr=self.lr)
        optimizer_classifier = torch.optim.SGD(model.classifier.parameters(), lr=0)
        dataLoader = self.get_dataloader(train=True)
        model.train()
        for epoch in range(iteration):
            for images, labels in dataLoader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                optimizer_extractor.zero_grad()
                optimizer_classifier.zero_grad()
                loss.backward()
                optimizer_extractor.step()
    
    def train(self):
        self.update_model(self.model, self.epochs)
        self.update_model(copy.deepcopy(self.model), self.epochs * self.heterogeneity)
        