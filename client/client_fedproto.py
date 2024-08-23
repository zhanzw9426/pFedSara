import copy
import torch
import torch.nn as nn
from client.client_base import Client_Base
from collections import defaultdict

class client_FedProto(Client_Base):
    def __init__(self, args, clientINFO, **kwargs):
        super().__init__(args, clientINFO, **kwargs)
        self.local_protos = {label:None for label in range(args.num_classes)}
        self.global_protos = None
        self.criterion_proto = nn.MSELoss()
        self.lamda = 1

    def set_protos(self, global_protos):
        self.global_protos = global_protos

    def evaluate(self, global_evaluate_loader=None, global_fintuning_loader=None, finetune=False, finetune_lr=0.01):
        if global_evaluate_loader is None:
            evaluate_loader = self.get_dataloader(True)
            finetune_loader = self.get_dataloader(False)
        else:
            evaluate_loader = global_evaluate_loader
            finetune_loader = global_fintuning_loader
        model = copy.deepcopy(self.model) if finetune else self.model

        if finetune:
            assert finetune_loader is not None
            optimizer = torch.optim.SGD(model.parameters(), lr=finetune_lr)

            model.train()
            iter_loader = iter(finetune_loader)
            images, labels = next(iter_loader)
            images = images.to(self.device)
            labels = labels.to(self.device)
            representations = model.extractor(images)
            #outputs = model.classifier(representations) 
            proto = copy.deepcopy(representations.detach())
            for i, label in enumerate(labels):
                proto[i, :].data = self.global_protos[label.item()].data
            loss = self.criterion_proto(proto, representations)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        metric = {'total': 0, 'correct': 0, 'accuracy':0.0}
        with torch.no_grad():
            for images, labels in evaluate_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                representations = model.extractor(images)
                output = float('inf') * torch.ones(labels.shape[0], self.num_classes).to(self.device)
                for i, representation in enumerate(representations):
                    for j, proto in self.global_protos.items():
                        if type(proto) != type([]):
                            output[i, j] = nn.MSELoss()(representation, proto)
                metric['total'] += labels.shape[0]
                metric['correct'] += (torch.sum(torch.argmin(output, dim=1) == labels)).item()
        metric['accuracy'] = metric['correct'] / metric['total']
        return metric
    
    def update_model(self, model, iteration):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        dataLoader = self.get_dataloader(train=True)
        model.train()
        for epoch in range(iteration):
            for images, labels in dataLoader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                representations = model.extractor(images)
                outputs = model.classifier(representations)
                loss = self.criterion(outputs, labels)
                proto = copy.deepcopy(representations.detach())
                for i, label in enumerate(labels):
                    proto[i, :].data = self.global_protos[label.item()].data
                loss += self.criterion_proto(proto, representations) * self.lamda
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def train(self):
        self.update_model(self.model, self.epochs)

        dataloader = self.get_dataloader(train=True)
        protos = defaultdict(list)
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                representations = self.model.extractor(images)
                for i, label in enumerate(labels):
                    protos[label.item()].append(representations[i, :].detach().data)

        for [label, proto_list] in protos.items():
            proto = proto_list[0].data
            for i in range(1, len(proto_list)):
                proto += proto_list[i].data
            self.local_protos[label] = proto / len(proto_list)
        self.update_model(copy.deepcopy(self.model), self.epochs * self.heterogeneity)
