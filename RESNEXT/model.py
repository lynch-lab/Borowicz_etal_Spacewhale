import torch
from torch import nn as nn
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
#from . import resnext_101_32x4d_
from . import resneXt
#from . import resnext_101_32x4d_
class ResNeXt101(nn.Module):
    def __init__(self,out_nc=2):
        print(os.getcwd())
        super(ResNeXt101, self).__init__()
        #net = resnext_101_32x4d_.resnext_101_32x4d
        net = resneXt.resnext_101_32x4d
        #net = resnext_101_32x4d_.resnext_101_32x4d
        #resnext_101_32_path = './RESNEXT/resnext_101_32x4d.pth'
        resnext_101_32_path = './RESNEXT/resnext_101_32x4d.pth'
        net.load_state_dict(torch.load(resnext_101_32_path))
        
        for param in net.parameters():
            param.require_grad=False
        num_features = 8192
        features=list()
        features = list(net.children())[:-2] # Remove the last layer
        #features.extend([nn.Conv2d(2048, out_nc, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0), bias=True)])
        features.extend([nn.Conv2d(2048, out_nc, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0), bias=True)]) 
        self.net = nn.Sequential(*features) # Replace the model classifier
        
    def forward(self, x):
        return torch.squeeze(self.net(x))
if __name__=="__main__":
    a = ResNeXt101()
    b=torch.rand(3,3,256,256)
    a.net = nn.DataParallel(a.net)
    a.net.to(0)
    b=torch.rand(3,3,256,256)
    b.to(0)
    k= a(b)
    print(k.shape)
