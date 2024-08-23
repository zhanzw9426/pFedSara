import time
import torch
import torch.nn as nn
from client.client_fedlc import Client_FedLC
from server.server_base import Server_Base


class Server_FedLC(Server_Base):
    def __init__(self, args):
        super().__init__(args)
        self.set_clients(Client_FedLC)
        print(">> Initialization completed.")
        sample_per_class = torch.zeros(self.num_classes).to(self.device)
        for client in self.clients:
            sample_per_class += client.sample_per_class
        val = 1.0 * sample_per_class ** (-1/4)
        for client in self.clients:
            client.calibration = val

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
            self.logger.info("## Round %3d ## Round time: %.4f"%(round, roundtime))
            self.collect_locals()
            self.aggregate()
