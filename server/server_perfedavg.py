import copy
from client.client_perfedavg import Client_PerFedAvg
from server.server_base import Server_Base
import time

class Server_PerFedAvg(Server_Base):
    def __init__(self, args):
        super().__init__(args)
        self.set_clients(Client_PerFedAvg)
        print(">> Initialization completed.")

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
                print(time.time() - starttime)
            self.logger.info("## Round %3d ## Round time: %.4f"%(round, roundtime))

            self.collect_locals()
            self.aggregate()

