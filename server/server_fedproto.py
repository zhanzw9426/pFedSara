from client.client_fedproto import client_FedProto
from server.server_base import Server_Base
from collections import defaultdict
import time
import torch
import utils

class Server_FedProto(Server_Base):
    def __init__(self, args):
        super().__init__(args)
        self.set_clients(client_FedProto)
        _, size = utils.get_out_dim(self.global_model.extractor, args.input_size)
        self.global_protos = {label:torch.rand(size).to(self.device) for label in range(args.num_classes)}

    def train(self):
        for round in range(self.comm_round+1):
            self.distribute_global()
            if round % self.evaluate_gap == 0:
                G, FG, L, FL = self.evaluate_proto(True, True, True, True)
                self.logger.info("## Round %3d ## Global: %.4f, Finetune Global: %.4f, Local: %.4f, Finetune Local: %.4f"%(round, G, FG, L, FL))    
            self.sample_clients()
            roundtime = 0.0    
            for client in self.select_clients:
                starttime = time.time()
                client.train()
                roundtime = max(roundtime, time.time()-starttime)
            self.logger.info("## Round %3d ## Round time: %.4f"%(round, roundtime))
            self.collect_locals()
            self.aggregate()
            
    def distribute_global(self):
        for client in self.clients:
            client.set_protos(self.global_protos)
            
    def collect_locals(self):
        self.client_protos = []
        for client in self.select_clients:
            self.client_protos.append(client.local_protos)
            
    def aggregate(self):
        protos = defaultdict(list)
        for client_protos in self.client_protos:
            for label in range(self.num_classes):
                if client_protos[label] is not None:
                    protos[label].append(client_protos[label])

        for label in range(self.num_classes):
            if len(protos[label]) > 0: 
                proto = protos[label][0].data
                for i in range(1, len(protos[label])):
                    proto += protos[label][i].data
                self.global_protos[label] = proto / len(protos[label])
            
    def evaluate_proto(self, is_global=False, is_finetune_global=False, is_local=False, is_finetune_local=False):
        G_acc = 0.0
        FG_acc = 0.0
        L_acc = 0.0
        FL_acc = 0.0
        if is_global:
            for client in self.clients:
                metric = client.evaluate(self.global_testloader)
                G_acc += metric['accuracy']
            G_acc /= self.num_clients
        if is_finetune_global:
            for client in self.clients:
                metric = client.evaluate(self.global_testloader, self.global_trainloader, True, self.finetune_lr)
                FG_acc += metric['accuracy']
            FG_acc /= self.num_clients
        if is_local:
            for client in self.clients:
                metric = client.evaluate()
                L_acc += metric['accuracy']
            L_acc /= self.num_clients
        if is_finetune_local:
            for client in self.clients:
                metric = client.evaluate(finetune=True, finetune_lr=self.finetune_lr)
                FL_acc += metric['accuracy']
            FL_acc /= self.num_clients
        return G_acc, FG_acc, L_acc, FL_acc
