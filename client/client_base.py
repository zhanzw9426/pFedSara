import copy
import utils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MyDataset
import numpy as np

class Client_Base(object):
    def __init__(self, args, clientINFO, **kwargs):
        self.lr = args.lr
        self.id = clientINFO['id']
        self.testset = MyDataset(clientINFO['test_data'], clientINFO['test_label'])
        self.trainset = MyDataset(clientINFO['train_data'], clientINFO['train_label'])
        self.dataVolume = len(self.trainset)
        self.epochs = args.epochs
        self.device = args.device
        self.algorithm = args.algorithm
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes
        self.num_clients = args.num_clients
        self.model = copy.deepcopy(args.model).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
    def set_parameters(self, model):
        utils.clone_model(model, self.model)

    def update_parameters(self, new_params):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()
        
    def get_dataloader(self, train:bool):
        if train:
            return DataLoader(dataset=self.trainset, batch_size=self.batch_size, shuffle=True)
        else:
            return DataLoader(dataset=self.testset, batch_size=self.batch_size, shuffle=True)
        
    def evaluate_importance(self, model):
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
    
    