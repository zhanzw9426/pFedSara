import torch
import numpy as np
import copy
import torch.nn as nn


def get_info(dataset):
    info = dict()
    if dataset == 'MNIST':
        info['input_channel'] = 1
        info['input_width'] = 28
        info['input_height'] = 28
        info['num_classes'] = 10
    elif dataset == 'FashionMNIST':
        info['input_channel'] = 1
        info['input_width'] = 28
        info['input_height'] = 28
        info['num_classes'] = 10
    elif dataset == 'EMNIST':
        info['input_channel'] = 1
        info['input_width'] = 28
        info['input_height'] = 28
        info['num_classes'] = 26
    elif dataset == 'SVHN':
        info['input_channel'] = 3
        info['input_width'] = 32
        info['input_height'] = 32
        info['num_classes'] = 10
    elif dataset == 'CIFAR10':
        info['input_channel'] = 3
        info['input_width'] = 32
        info['input_height'] = 32
        info['num_classes'] = 10
    elif dataset == 'CIFAR100':
        info['input_channel'] = 3
        info['input_width'] = 32
        info['input_height'] = 32
        info['num_classes'] = 100
    return info


class KMeansTorch:
    def __init__(self, n_clusters, filter=None, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.filter = filter
    
    def fit(self, X):
        num_X = X.size(0)
        centroids = X[torch.randperm(num_X)[:self.n_clusters]]
        for _ in range(self.max_iter):
            cnt = 0
            distances = torch.cdist(X, centroids)
            labels = torch.full((num_X,), -1, dtype=torch.int)
            flattened_matrix = distances.view(-1)
            _, sorted_indices = torch.sort(flattened_matrix)
            sorted_i = sorted_indices // self.n_clusters
            sorted_j = sorted_indices % self.n_clusters
            ready_set = [set() for j in range(self.n_clusters)]
            for i,j in zip(sorted_i.tolist(), sorted_j.tolist()):
                if labels[i] == -1:
                    flag = True
                    for element in ready_set[j]:
                        if self.filter[i][element]:
                            flag = False
                            break
                    if flag:
                        ready_set[j].add(i)
                        labels[i] = j
                        cnt += 1
                    if cnt == num_X:
                        break    
            new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(self.n_clusters)])
            if torch.norm(new_centroids - centroids) < self.tol:
                break
            centroids = new_centroids
        self.centroids = centroids
        self.labels = labels
        

def spectral_clustering(similarity_matrix:torch.tensor, n_clusters:int, filter=None):
    similarity_matrix = (similarity_matrix + 1) / 2
    D = torch.diag(similarity_matrix.sum(dim=1))
    L = D - similarity_matrix
    eigvals, eigvecs = torch.linalg.eigh(L)
    feature_matrix = eigvecs[:, 1:n_clusters]
    kmeans = KMeansTorch(n_clusters=n_clusters, filter=filter)
    kmeans.fit(feature_matrix)
    return kmeans.labels

def get_out_dim(module, indim):
    if isinstance(module, list):
        module = torch.nn.Sequential(*module)
    fake_input = torch.zeros(indim).unsqueeze(0).to(next(module.parameters()).device)
    with torch.no_grad():
        fake_output = module(fake_input).squeeze(0)
    output_size = fake_output.view(-1).size()[0]
    return fake_output.size(), output_size


def clone_model(source, target):
    if isinstance(source, list):
        for s_param, t_param in zip(source, target.parameters()):
            t_param.data = s_param.data.clone()
    else:
        for s_param, t_param in zip(source.parameters(), target.parameters()):
            t_param.data = s_param.data.clone()


def get_flat_model_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)
    return flat_params.detach()

def get_flat_grad(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
        else:
            grads.append(torch.zeros_like(param).view(-1))
    flat_grads = torch.cat(grads)
    return flat_grads.detach()

def set_flat_model_params(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def evaluate_model(model, evaluate_loader, device, finetune=False, finetune_loader=None, finetune_lr=0.01):
    criterion = nn.CrossEntropyLoss()
    model = copy.deepcopy(model) if finetune else model
    if finetune:
        assert finetune_loader is not None
        optimizer = torch.optim.SGD(model.parameters(), lr=finetune_lr)
        model.train()
        iter_loader = iter(finetune_loader)
        images, labels = next(iter_loader)
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    metric = {'total': 0, 'correct': 0, 'accuracy':0.0, 'loss': 0.0}
    model.eval()
    with torch.no_grad():
        for images, labels in evaluate_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            metric['total'] += labels.shape[0]
            metric['correct'] += (torch.sum(torch.argmax(outputs, dim=1) == labels)).item()
            metric['loss'] += loss.item() * labels.shape[0]
    metric['accuracy'] = metric['correct'] / metric['total']
    metric['loss'] = metric['loss'] / metric['total']
    return metric