### SPACEWHALE PROJECT
### aborowicz@coa.edu
### This script resizes images to meet the standard for the various pytorch models
### Specifically that they should be 224x224. We had previously tiled them to 32x32
### Now we'll basically just stretch them to 224x224 from 32
### Adapted from http://effbot.org/imagingbook/introduction.htm
### and https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio



#######

# Usage example: python resize_images.py --root ./full_air --size 224 --output ./full_air/resize_224

####### 

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
parse.add_argument('--root',type=str)
parse.add_argument('--size',type=int,default=224)
parse.add_argument('--output',type=str)
opt = parse.parse_args()
opt.im_fold = opt.root
opt.results = opt.output
size = (opt.size, opt.size)

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
    png = Image.open(im_path)
    #w,h,z = png.shape

    s.resize_ims(png,size,opt.results+'/'+imname)

