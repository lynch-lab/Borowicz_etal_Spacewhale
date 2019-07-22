import numpy as np
from PIL import Image
import time
import torch
import os.path
import argparse
from scipy import misc
from m_util import *


s = spacewhale()

parse = argparse.ArgumentParser()
parse.add_argument('--root',type=str,default='./Water_Training')
parse.add_argument('--step',type=int,default=500)
parse.add_argument('--size',type=int,default=30)
parse.add_argument('--output',type=str,default='./water')
opt = parse.parse_args()
opt.im_fold = opt.root
opt.results = opt.output

s.sdmkdir(opt.results)
opt.input_nc =3
imlist=[]
imnamelist=[]

for root,_,fnames in sorted(os.walk(opt.root)):
    for fname in fnames:
        if fname.lower().endswith('.png'):
            path = os.path.join(root,fname)
            imlist.append((path,fname))
            imnamelist.append(fname)
            
for im_path,imname in  imlist:
    png = misc.imread(im_path,mode='RGB')
    w,h,z = png.shape

    s.savepatch_train(png,w,h,opt.step,opt.size,opt.results+'/'+imname[:-4]+'#')
