import os

import cv2
import torch
from torch.utils import data

import numpy as np
import random
from edge_detector import hed_edge
prototxt='deploy.prototxt'
caffemodel='hed_pretrained_bsds.caffemodel'
random.seed(10)
test_folder='./edge'
def load_sal_label(name):
    name=name[:-4]+'_GT.png'
    print(name)


    combined  = hed_edge(name,prototxt,caffemodel)
    filename=os.path.join(test_folder, name[:-4] + '_edge.png')
    out=cv2.imwrite(filename,combined)
  
    return combined
sal_root = '../../../'
sal_source = 'test.lst'


with open(sal_source, 'r') as f:
    sal_list = [x.strip() for x in f.readlines()]

sal_num = len(sal_list)
print('total images',sal_num)

for item in range(sal_num):
    
    gt_name = sal_list[item % sal_num].split()[1]
    print(gt_name)
    name=sal_list[item % sal_num].split()[0].split('/')[1]
    name1=name[:-4]+'_GT.png'
    print(name,name1)
    im = cv2.imread(name1, cv2.IMREAD_GRAYSCALE)
    #gradient
    gX = cv2.Sobel(im, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gY = cv2.Sobel(im, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    combined = np.array(combined , dtype=np.float32)
    combined  = cv2.resize(combined , (320,320))
    #combined  = combined  / 255.0
    #combined  = combined [..., np.newaxis]
    sal_edge=combined
    #sal_edge = load_sal_label(name)
    #name=sal_list[item % sal_num].split()[0].split('/')[1]
    filename=os.path.join(test_folder, name[:-4] + '_edge.png')
    out=cv2.imwrite(filename,sal_edge)
    print(item,out)
       








