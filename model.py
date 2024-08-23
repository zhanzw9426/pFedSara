import utils
import torch
import copy
import torch.nn as nn
import numpy as np
from utils import *

class Model_decoupling(nn.Module):
    def __init__(self, extractor, classifier):
        super(Model_decoupling, self).__init__()
        self.extractor = extractor
        self.classifier = classifier
        
    def forward(self, x):
        out = self.extractor(x)
        out = self.classifier(out)
        return out

class CNN(nn.Module):
    def __init__(self, input_size=(3,32,32), num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size[0], 8, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8,16, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        _, dim = utils.get_out_dim(nn.Sequential(self.conv1,self.conv2), input_size)
        self.fc = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
