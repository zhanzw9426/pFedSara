import os
import torch
import pickle
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10, SVHN, FashionMNIST, CIFAR100, EMNIST
import argparse
from utils import *


class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        super(MyDataset, self).__init__()
        self.data = torch.from_numpy(data.astype(np.float32))
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform is not None:
            image = self.transform(image)
        labels = self.labels[idx]
        return image, labels
    
    def __len__(self):
        return self.data.shape[0]


'''
This function is used to preprocessing the dataset
The currently supported datasets are [MNIST, FashionMNIST, SVHN, CIFAR10, CIFAR100]
The dimensionality of the processed dataset:
# data(float): num_data, channel, width, height,
# label(long): num_data,
The output file is formatted as npy, and the flie name is set to the following format:
./data/processed/{dataset}_{data/labels}_{train/test}.npy
'''
def datasets_preprocessing(dataset):
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    if not os.path.isdir('./data/processed'):
        os.mkdir('./data/processed')
    trainData = None
    trainLabel = None
    testData = None
    testLabel = None
    if dataset == 'MNIST':
        transform = transforms.Normalize((33.31842041015625,), (78.56748962402344,))
        mnist_train_obj = MNIST(root='./data', train=True, download=True)
        trainData = transform(mnist_train_obj.data.float()).float()
        trainData = torch.unsqueeze(trainData,dim=1)
        trainLabel = mnist_train_obj.targets
        mnist_test_obj = MNIST(root='./data', train=False, download=True)
        testData = transform(mnist_test_obj.data.float()).float()
        testData = torch.unsqueeze(testData,dim=1)
        testLabel = mnist_test_obj.targets
    elif dataset == 'FashionMNIST':
        transform = transforms.Normalize((72.94034576416016,), (90.02118682861328,))
        fmnist_train_obj = FashionMNIST(root='./data', train=True, download=True)
        trainData = transform(fmnist_train_obj.data.float()).float()
        trainData = torch.unsqueeze(trainData,dim=1)
        trainLabel = fmnist_train_obj.targets
        fmnist_test_obj = FashionMNIST(root='./data', train=False, download=True)
        testData = transform(fmnist_test_obj.data.float()).float()
        testData = torch.unsqueeze(testData,dim=1)
        testLabel = fmnist_test_obj.targets
    elif dataset == 'EMNIST':
        transform = transforms.Normalize((43.91796244399202,), (84.39138996531052,))
        emnist_train_obj = EMNIST(root='./data', split='letters', train=True, download=True)
        trainData = transform(emnist_train_obj.data.float()).float()
        trainData = torch.unsqueeze(trainData,dim=1)
        trainLabel = emnist_train_obj.targets-1
        emnist_test_obj = EMNIST(root='./data', split='letters', train=False, download=True)
        testData = transform(emnist_test_obj.data.float()).float()
        testData = torch.unsqueeze(testData,dim=1)
        testLabel = emnist_test_obj.targets-1
    elif dataset == 'SVHN':
        transform = transforms.Normalize((111.60893667531344, 113.161274663812, 120.5651276685803,), (50.49768207683532, 51.258984644817524, 50.2442164739218))
        svhn_train_obj = SVHN(root='./data', split='train', download=True)
        trainData = transform(torch.FloatTensor(svhn_train_obj.data)).float()
        trainLabel = torch.LongTensor(svhn_train_obj.labels)
        svhn_test_obj = SVHN(root='./data', split='test', download=True)
        testData = transform(torch.FloatTensor(svhn_test_obj.data)).float()
        testLabel = torch.LongTensor(svhn_test_obj.labels)
    elif dataset == 'CIFAR10':
        transform = transforms.Normalize((125.30690002441406, 122.95014953613281, 113.86599731445312,), (62.993221282958984, 62.088706970214844, 66.70490264892578))
        cifar_train_obj = CIFAR10(root='./data', train=True, download=True)
        trainData = transform(torch.FloatTensor(np.transpose(cifar_train_obj.data, (0, 3, 1, 2)))).float()
        trainLabel = torch.LongTensor(cifar_train_obj.targets)
        cifar_test_obj = CIFAR10(root='./data', train=False, download=True)
        testData = transform(torch.FloatTensor(np.transpose(cifar_test_obj.data, (0, 3, 1, 2)))).float()
        testLabel = torch.LongTensor(cifar_test_obj.targets)
    elif dataset == 'CIFAR100':
        transform = transforms.Normalize((129.30386352539062, 124.06986999511719, 112.43356323242188,), (68.17024230957031, 65.39180755615234, 70.41837310791016))
        cifar_train_obj = CIFAR100(root='./data', train=True, download=True)
        trainData = transform(torch.FloatTensor(np.transpose(cifar_train_obj.data, (0, 3, 1, 2)))).float()
        trainLabel = torch.LongTensor(cifar_train_obj.targets)
        cifar_test_obj = CIFAR100(root='./data', train=False, download=True)
        testData = transform(torch.FloatTensor(np.transpose(cifar_test_obj.data, (0, 3, 1, 2)))).float()
        testLabel = torch.LongTensor(cifar_test_obj.targets)
    trainData = trainData.numpy()
    trainLabel = trainLabel.numpy()
    testData = testData.numpy()
    testLabel = testLabel.numpy()
    np.save('./data/processed/{}_data_train.npy'.format(dataset), trainData)
    np.save('./data/processed/{}_labels_train.npy'.format(dataset), trainLabel)
    np.save('./data/processed/{}_data_test.npy'.format(dataset), testData)
    np.save('./data/processed/{}_labels_test.npy'.format(dataset), testLabel)
    print("The dataset {} is processed.".format(dataset))
    print("train data: {}".format(trainData.shape))
    print("train label: {}".format(trainLabel.shape))
    print("test data: {}".format(testData.shape))
    print("test label: {}".format(testLabel.shape))
    
def scale(totalSize:int, dataVolume:int):
    idxs = list(range(dataVolume)) * (int)(np.ceil(totalSize/dataVolume))
    np.random.shuffle(idxs)
    idxs = idxs[:totalSize]
    return idxs


# totalSize is a array with shape (num_classes,)
def scaleByGroup(labels, totalSize):
    idxs = dict()
    for y in range(totalSize.shape[0]):
        idxs[y] = np.where(labels == y)[0]
        idxs[y] = np.repeat(idxs[y], (int)(np.ceil(totalSize[y]/idxs[y].shape[0])))
        np.random.shuffle(idxs[y])
    return idxs
    
    
# split function
def split_homo(trainData, trainLabel, testData, testLabel, num_clients, avgSize):
    totalSize = avgSize*num_clients
    clientData = {'num_clients':num_clients, 'train':{'data':{}, 'label':{}}, 'test':{'data':{}, 'label':{}}}
    
    idxs = scale(totalSize, trainData.shape[0])
    batch_idxs = np.array_split(idxs, num_clients)
    for i in range(num_clients):
        clientData['train']['data'][i] = trainData[batch_idxs[i]]
        clientData['train']['label'][i] = trainLabel[batch_idxs[i]]

    idxs = scale(totalSize, testData.shape[0])
    batch_idxs = np.array_split(idxs, num_clients)
    for i in range(num_clients):
        clientData['test']['data'][i] = testData[batch_idxs[i]]
        clientData['test']['label'][i] = testLabel[batch_idxs[i]]
    
    return clientData


def split_quantitySkew(trainData, trainLabel, testData, testLabel, num_clients, avgSize, alpha):
    totalSize = avgSize*num_clients
    clientData = {'num_clients':num_clients, 'train':{'data':{}, 'label':{}}, 'test':{'data':{}, 'label':{}}}

    idxs_train = scale(totalSize, trainData.shape[0])
    idxs_test = scale(totalSize, testData.shape[0])

    min_size = 0
    while min_size < 10:
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        min_size = np.min(proportions*totalSize)
        
    proportions = (np.cumsum(proportions)*totalSize).astype(int)[:-1]
    batch_idxs = np.split(idxs_train, proportions)
    for i in range(num_clients):
        clientData['train']['data'][i] = trainData[batch_idxs[i]]
        clientData['train']['label'][i] = trainLabel[batch_idxs[i]]

    batch_idxs = np.split(idxs_test, proportions)
    for i in range(num_clients):
        clientData['test']['data'][i] = testData[batch_idxs[i]]
        clientData['test']['label'][i] = testLabel[batch_idxs[i]]
    
    return clientData


def split_dirichlet(trainData, trainLabel, testData, testLabel, num_clients, avgSize, alpha, num_classes):
    clientData = {'num_clients':num_clients, 'train':{'data':{}, 'label':{}}, 'test':{'data':{}, 'label':{}}}
    
    label_distribution = np.random.dirichlet(np.repeat(alpha, num_classes), num_clients)
    label_nums = np.round(label_distribution*avgSize).astype(int)
    totalSize = label_nums.sum(axis=0)
    
    idxs_train = scaleByGroup(trainLabel, totalSize)
    idxs_test = scaleByGroup(testLabel, totalSize)

    for i in range(num_clients):
        clientData['train']['data'][i] = np.empty((0,) + trainData[0].shape)
        clientData['train']['label'][i] = np.empty((0,), dtype=np.int_)
        clientData['test']['data'][i] = np.empty((0,) + trainData[0].shape)
        clientData['test']['label'][i] = np.empty((0,), dtype=np.int_)
        for y in range(num_classes):
            idx, idxs_train[y] = np.split(idxs_train[y], [label_nums[i][y]])
            clientData['train']['data'][i] = np.append(clientData['train']['data'][i], trainData[idx], axis=0)
            clientData['train']['label'][i] =  np.append(clientData['train']['label'][i], trainLabel[idx], axis=0)
            
            idx, idxs_test[y] = np.split(idxs_test[y], [label_nums[i][y]])
            clientData['test']['data'][i] = np.append(clientData['test']['data'][i], testData[idx], axis=0)
            clientData['test']['label'][i] =  np.append(clientData['test']['label'][i], testLabel[idx], axis=0)

    return clientData
        

# Each clients only owns fixed number of label
def split_pathology(trainData, trainLabel, testData, testLabel, num_clients, avgSize, num_classes, alpha):
    clientData = {'num_clients':num_clients, 'train':{'data':{}, 'label':{}}, 'test':{'data':{}, 'label':{}}}
    label_nums = np.zeros((num_clients,num_classes), dtype=int)
    
    num_class1 = (int)(alpha * avgSize)
    num_class2 = avgSize - num_class1
    
    for i in range(num_clients):
        classes = np.random.permutation(num_classes)
        idx = np.where(classes == i%num_classes)[0]
        classes[0], classes[idx] = classes[idx], classes[0]
        label_nums[i][classes[0]] = num_class1
        label_nums[i][classes[1]] = num_class2
    totalSize = label_nums.sum(axis=0)
    
    idxs_train = scaleByGroup(trainLabel, totalSize)
    idxs_test = scaleByGroup(testLabel, totalSize)

    for i in range(num_clients):
        clientData['train']['data'][i] = np.empty((0,) + trainData[0].shape)
        clientData['train']['label'][i] = np.empty((0,), dtype=np.int_)
        clientData['test']['data'][i] = np.empty((0,) + trainData[0].shape)
        clientData['test']['label'][i] = np.empty((0,), dtype=np.int_)
        for y in range(num_classes):
            idx, idxs_train[y] = np.split(idxs_train[y], [label_nums[i][y]])
            clientData['train']['data'][i] = np.append(clientData['train']['data'][i], trainData[idx], axis=0)
            clientData['train']['label'][i] =  np.append(clientData['train']['label'][i], trainLabel[idx], axis=0)
            
            idx, idxs_test[y] = np.split(idxs_test[y], [label_nums[i][y]])
            clientData['test']['data'][i] = np.append(clientData['test']['data'][i], testData[idx], axis=0)
            clientData['test']['label'][i] =  np.append(clientData['test']['label'][i], testLabel[idx], axis=0)

    return clientData


def show_partition_info(clientData, num_classes):
    print("num_clients: " + str(clientData['num_clients']))
    for split in ['train', 'test']:
        print('split: ' + split)
        for id in range(clientData['num_clients']):
            print('client ' + str(id), end=':')
            cnt = np.zeros(num_classes)
            for y in range(num_classes):
                cnt[y] = (clientData[split]['label'][id] == y).sum()
            print(cnt)
        print()
            
'''
{
    num_clients: ...
    train:
    {
        data:
        {
            0: [x00, x01, ...]
            ...
        }
        label:
        {
            0: [y00, y01, ...]
            ...
        }
        ...
    }
    test:
    {
        data:...
        label:...
    }
}
'''
# DATA = pickle.load(open('filename.pkl', 'rb'))
def partition_data(dataset, partition, num_clients, num_classes, alpha, avgSize=500):
    if not os.path.isdir('./data/clientData'):
        os.makedirs('./data/clientData')
    assert dataset in ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'EMNIST']
    assert partition in ['homo', 'dirichlet', 'quantitySkew', 'pathology']
    assert alpha > 0
    
    if not (os.path.exists('./data/processed/{}_data_train.npy'.format(dataset)) \
      and os.path.exists('./data/processed/{}_labels_train.npy'.format(dataset)) \
      and os.path.exists('./data/processed/{}_data_test.npy'.format(dataset)) \
      and os.path.exists('./data/processed/{}_labels_test.npy'.format(dataset))):
        datasets_preprocessing(dataset)
    
    trainData = np.load('./data/processed/{}_data_train.npy'.format(dataset))
    trainLabel = np.load('./data/processed/{}_labels_train.npy'.format(dataset))
    testData = np.load('./data/processed/{}_data_test.npy'.format(dataset))
    testLabel = np.load('./data/processed/{}_labels_test.npy'.format(dataset))

    if partition == "homo":
        clientData = split_homo(trainData, trainLabel, testData, testLabel, num_clients, avgSize)
    
    elif partition == "quantitySkew":
        clientData = split_quantitySkew(trainData, trainLabel, testData, testLabel, num_clients, avgSize, alpha)

    elif partition == "dirichlet":
        clientData = split_dirichlet(trainData, trainLabel, testData, testLabel, num_clients, avgSize, alpha, num_classes)

    elif partition == "pathology":
        assert alpha <= 1
        clientData = split_pathology(trainData, trainLabel, testData, testLabel, num_clients, avgSize, num_classes, alpha)

    with open('./data/clientData/{}_clientData_{}.pkl'.format(dataset, partition), 'wb') as outfile:
        pickle.dump(clientData, outfile)
    #print('>>> Data allocation completed successfully.')
    show_partition_info(clientData, num_classes)
    return clientData


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--partition', type=str, default='pathology')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.7)

    args = parser.parse_args()
    partition_data(args.dataset, args.partition, args.num_clients, get_info(args.dataset)['num_classes'], args.alpha, avgSize=500)
