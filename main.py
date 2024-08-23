import argparse
import random
import copy
import numpy as np
import torch
import logging

from dataset import *
from model import *
from server.server_fedavg import Server_FedAvg
from server.server_fedprox import Server_FedProx
from server.server_perfedavg import Server_PerFedAvg
from server.server_scaffold import Server_Scaffold
from server.server_pfedme import Server_pFedMe
from server.server_fedbabu import Server_FedBABU
from server.server_fedrep import Server_FedRep
from server.server_pfedsara import Server_pFedSara
from server.server_fedlc import Server_FedLC
from server.server_fedproto import Server_FedProto

Servers_dict = {
    "FedAvg": Server_FedAvg,
    "FedProx": Server_FedProx,
    "PerFedAvg": Server_PerFedAvg,
    "pFedMe": Server_pFedMe,
    "Scaffold": Server_Scaffold,
    "FedBABU": Server_FedBABU,
    "FedRep": Server_FedRep,
    "pFedSara": Server_pFedSara,
    'FedLC':Server_FedLC,
    'FedProto':Server_FedProto
}

def add_args(parser):
    parser.add_argument('--algorithm', type=str, default='pFedSara', help="alorithm, supporting {'FedMoE'}")
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset used for training')
    parser.add_argument('--partition_mode', type=str, default='dirichlet', help="types of data partition")
    parser.add_argument('--partition_alpha', type=float, default=0.1, help='available parameters for data partition method')
    parser.add_argument('--sample_mode', type=str, default='random', help="types of client sampling")
    parser.add_argument('--sample_rate', type=float, default=0.2, help='samping rate')
    parser.add_argument('--mu', type=float, default=1, help='parameter for FedProx')
    parser.add_argument('--finetune_lr', type=float, default=0.001, help='parameter for finetuning')
    parser.add_argument('--batch_size', type=int, default=100, help='local batch size for training')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--server_lr', type=float, default=1, help='learning rate')
    parser.add_argument('--timelimit', type=float, default=1.2, help='Time limit for local update')
    parser.add_argument('--threshold', type=float, default=0.9, help='Threshold for pFedSara')
    parser.add_argument('--static', action='store_true', help='Static threshold for pFedSara')
    parser.add_argument('--epochs', type=int, default=5, help='local training epochs for each client')
    parser.add_argument('--num_clients', type=int, default=50, help='number of workers in a distributed cluster')
    parser.add_argument('--avg_data_size', type=int, default=500, help='number of samples in a client')
    parser.add_argument('--comm_round', type=int, default=1000, help='total communication rounds')
    parser.add_argument('--evaluate_gap', type=int, default=2, help='the frequency of the test algorithms')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--logger_path", type=str, default='log')
    return parser


if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser(description='Adaptive Federated Mixture of Experts'))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, 
        format='%(message)s', 
        handlers=[logging.FileHandler("./logs/{}.log".format(args.logger_path), mode='a'), ]
    )
    args.logger = logging.getLogger()

    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    info = get_info(args.dataset)
    args.input_size = (info['input_channel'], info['input_width'],  info['input_height'])
    args.num_classes = info['num_classes']
    if args.algorithm in ['FedPer', 'FedBABU', 'FedRep', 'FedProto']:
        model = CNN(args.input_size, args.num_classes)
        classifier = copy.deepcopy(model.fc)
        model.fc = nn.Identity()
        args.model = Model_decoupling(model, classifier)
    else:
        args.model = CNN(args.input_size, args.num_classes)
    
    try:
        server = Servers_dict[args.algorithm](args)
        server.train()
    except KeyError:
        value = ">> The algorithm %s is not implemented."%args.algorithm
