from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
def define_model(name ='resnet32'):
    if name=='resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    if name=='resnet32':
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features  
        model.fc = nn.Linear(num_ftrs, 2)    
    if name=='densenet':
        model = models.densenet161(pretrained=True)
        num_ftrs = model.classifier.in_features  
        model.classifier = nn.Linear(num_ftrs, 2)    
    if name=='resneXt':
        from RESNEXT.model import ResNeXt101
        model = ResNeXt101().net
    return model


if __name__=='__main__':
#    a = define_model(name ='densenet')
    a = define_model(name ='resneXt')
    print(a)
