import copy
import torch
from client.client_scaffold import Client_Scaffold
from server.server_base import Server_Base
import time

class Server_Scaffold(Server_Base):
    def __init__(self, args):
        super().__init__(args)
        self.set_clients(Client_Scaffold)
        self.server_lr = args.server_lr
        self.global_c = []
        for param in self.global_model.parameters():
            self.global_c.append(torch.zeros_like(param))
        print(">> Initialization completed.")
        self.coff = self.server_lr / self.num_clients

    def train(self):
        for round in range(self.comm_round+1):
            self.sample_clients()
            self.distribute_global()

            if round % self.evaluate_gap == 0:
                G, FG, L, FL = self.evaluate(True, True, True, True)
                self.logger.info("## Round %3d ## Global: %.4f, Finetune Global: %.4f, Local: %.4f, Finetune Local: %.4f"%(round, G, FG, L, FL))    
            roundtime = 0.0
            for client in self.select_clients:
                starttime = time.time()
                client.train()
                roundtime = max(roundtime, time.time()-starttime)
                print(time.time()-starttime)
            self.logger.info("## Round %3d ## Round time: %.4f"%(round, roundtime))

            self.collect_locals()
            self.aggregate()

        self.save_results()

    def distribute_global(self):
        for client in self.select_clients:    
            client.set_parameters(self.global_model, self.global_c)

    def collect_locals(self):
        self.client_delta_x = []
        self.client_delta_c = []
        self.client_weights = []
        totalVolume = 0
        for client in self.select_clients:
            totalVolume += client.dataVolume
            self.client_weights.append(client.dataVolume)
            self.client_delta_x.append(client.delta_x)
            self.client_delta_c.append(client.delta_c)
        for i, w in enumerate(self.client_weights):
            self.client_weights[i] = w / totalVolume


    def aggregate(self):        
        global_model = copy.deepcopy(self.global_model)
        global_c = copy.deepcopy(self.global_c)
        for delta_x, delta_c in zip(self.client_delta_x, self.client_delta_c):
            for parm, dx in zip(global_model.parameters(), delta_x):
                parm.data += self.coff * dx.data.clone()
            for parm, dc in zip(global_c, delta_c):
                parm.data += dc.data.clone() / self.num_clients
        self.global_model = global_model
        self.global_c = global_c
