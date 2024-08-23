import os
import copy
import numpy as np
from dataset import partition_data
import torch
import torch.nn as nn
import utils
import random
from torch.utils.data import DataLoader
from dataset import MyDataset

class Server_Base(object):
    def __init__(self, args):
        self.args = args
        self.lr = args.lr
        self.finetune_lr = args.finetune_lr
        self.epochs = args.epochs
        self.logger = args.logger
        self.device = args.device
        self.dataset = args.dataset
        self.partition_mode = args.partition_mode
        self.partition_alpha = args.partition_alpha
        self.avg_data_size = args.avg_data_size
        self.algorithm = args.algorithm
        self.batch_size = args.batch_size
        self.sample_size = int(args.sample_rate * args.num_clients)
        self.comm_round = args.comm_round
        self.num_classes = args.num_classes
        self.num_clients = args.num_clients
        self.evaluate_gap = args.evaluate_gap
        self.global_model = copy.deepcopy(args.model).to(self.device)
        
        self.clients = []
        self.select_clients = []
        self.client_weights = []
        self.client_models = []

    def set_clients(self, clientObj):
        dataset = partition_data(self.dataset, self.partition_mode, self.num_clients, self.num_classes, self.partition_alpha, avgSize=self.avg_data_size)
        transform = [None] * self.num_clients
        trainX = []
        trainY = []
        testX = []
        testY = []
        for id in range(self.num_clients):
            trainX.append(dataset['train']['data'][id])
            trainY.append(dataset['train']['label'][id])
            testX.append(dataset['test']['data'][id])
            testY.append(dataset['test']['label'][id])
            clientINFO = {  'id':id,
                            'train_data':dataset['train']['data'][id], 
                            'train_label':dataset['train']['label'][id],
                            'test_data':dataset['test']['data'][id], 
                            'test_label':dataset['test']['label'][id], 
                            'transform':transform[id], }
            client = clientObj(self.args, clientINFO)
            self.clients.append(client)
        self.global_trainset = MyDataset(np.vstack(trainX), np.hstack(trainY))
        self.global_testset = MyDataset(np.vstack(testX), np.hstack(testY))
        self.global_trainloader = DataLoader(dataset=self.global_trainset, batch_size=self.batch_size, shuffle=True)
        self.global_testloader = DataLoader(dataset=self.global_testset, batch_size=1024, shuffle=True)
        
        heterogeneities = np.array([0,1,2,3,4] * (self.num_clients // 5))
        assert heterogeneities.shape[0] == self.num_clients
        heterogeneities = np.random.permutation(heterogeneities)
        for id in range(self.num_clients):
            self.clients[id].heterogeneity = heterogeneities[id]

    def sample_clients(self, method="random"):
        if method == "random":
            self.select_clients = random.sample(self.clients, self.sample_size)
        if method == "loss":
            self.evaluate_importance()
            sorted_ids = np.argsort(self.importance)
            self.select_clients = [self.clients[id] for id in sorted_ids[-self.sample_size:]]

    def distribute_global(self):
        for client in self.clients:
            client.set_parameters(self.global_model)

    def collect_locals(self):
        self.client_weights = []
        self.client_models = []
        totalVolume = 0
        for client in self.select_clients:
            totalVolume += client.dataVolume
            self.client_weights.append(client.dataVolume)
            self.client_models.append(client.model)
        for i, w in enumerate(self.client_weights):
            self.client_weights[i] = w / totalVolume

    def aggregate(self):
        for param in self.global_model.parameters():
            param.data.zero_()
        for w, client_model in zip(self.client_weights, self.client_models):
            for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w

    def evaluate(self, is_global=False, is_finetune_global=False, is_local=False, is_finetune_local=False):
        G_acc = 0.0
        FG_acc = 0.0
        L_acc = 0.0
        FL_acc = 0.0
        if is_global:
            metric = utils.evaluate_model(self.global_model, self.global_testloader, self.device)
            G_acc = metric['accuracy']
        if is_finetune_global:
            metric = utils.evaluate_model(self.global_model, self.global_testloader, self.device, True, self.global_trainloader, self.finetune_lr)
            FG_acc = metric['accuracy']
        if is_local:
            for client in self.clients:
                metric = utils.evaluate_model(self.global_model, client.get_dataloader(False), self.device)
                L_acc += metric['accuracy']
            L_acc /= self.num_clients
        if is_finetune_local:
            for client in self.clients:
                metric = utils.evaluate_model(self.global_model, client.get_dataloader(False), self.device, True, client.get_dataloader(True), self.finetune_lr)
                FL_acc += metric['accuracy']
            FL_acc /= self.num_clients
        return G_acc, FG_acc, L_acc, FL_acc
    
    def evaluate_privatemodel(self, is_global=False, is_finetune_global=False, is_local=False, is_finetune_local=False):
        G_acc = 0.0
        FG_acc = 0.0
        L_acc = 0.0
        FL_acc = 0.0
        if is_global:
            for client in self.clients:
                metric = utils.evaluate_model(client.model, self.global_testloader, self.device)
                G_acc += metric['accuracy']
            G_acc /= self.num_clients
        if is_finetune_global:
            for client in self.clients:
                metric = utils.evaluate_model(client.model, self.global_testloader, self.device, True, self.global_trainloader, self.finetune_lr)
                FG_acc += metric['accuracy']
            FG_acc /= self.num_clients
        if is_local:
            for client in self.clients:
                metric = utils.evaluate_model(client.model, client.get_dataloader(False), self.device)
                L_acc += metric['accuracy']
            L_acc /= self.num_clients
        if is_finetune_local:
            for client in self.clients:
                metric = utils.evaluate_model(client.model, client.get_dataloader(False), self.device, True, client.get_dataloader(True), self.finetune_lr)
                FL_acc += metric['accuracy']
            FL_acc /= self.num_clients
        return G_acc, FG_acc, L_acc, FL_acc
    
    def evaluate_importance(self):
        self.importance = np.zeros(self.num_clients)
        for id, client in enumerate(self.clients):
            self.importance[id] = client.evaluate_importance(self.global_model)
