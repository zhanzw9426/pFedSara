import time
from client.client_pfedsara import Client_pFedSara
from server.server_base import Server_Base
import numpy as np

class Server_pFedSara(Server_Base):
    def __init__(self, args):
        super().__init__(args)
        self.set_clients(Client_pFedSara)
        for client in self.clients:
            client.train_maml(True)
        dicts = {-1:0}
        for i in range(self.epochs+1):
            dicts[i] = 0
        for client in self.clients:
            dicts[client.tau] += 1
        print(">> Initialization completed.")

    def train(self):
        threshold = 1.0
        for round in range(self.comm_round+1):
            self.sample_clients('loss')
            self.distribute_global()
            for client in self.select_clients:
                if self.args.static:
                    client.threshold = self.args.threshold
                else:
                    client.threshold = threshold
            if round % self.evaluate_gap == 0:
                G, FG, L, FL = self.evaluate(True, True, True, True)
            self.logger.info("## Round %3d ## Global: %.4f, Finetune Global: %.4f, Local: %.4f, Finetune Local: %.4f"%(round, G, FG, L, FL))    
            similarity = 0.0
            roundtime = 0.0
            for client in self.select_clients:
                starttime = time.time()
                client.train()
                roundtime = max(roundtime, time.time()-starttime)
                similarity += client.similarity
            self.logger.info("## Round %3d ## Round time: %.4f"%(round, roundtime))
            self.collect_locals()
            self.aggregate()
            similarity /= len(self.select_clients)
            threshold = max(similarity, 0.2)
            
