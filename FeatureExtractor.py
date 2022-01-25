 #Import libraries 
import torch
import torch.nn as nn
from torch.autograd import Variable
import streamlit as st
import numpy as np
from numpy import linalg as LA
import torchvision
import torch.nn.functional as F
import time
from PIL import Image
from skimage.transform import resize
import warnings
import os
warnings.filterwarnings('ignore')

# Create the network to extract the features

class MyResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet, transform_input=False):
        super(MyResNetFeatureExtractor, self).__init__()
        self.transform_input = transform_input
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # self.fc = resnet.fc
        # stop where you want, copy paste from the model def

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[0] = x[0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[1] = x[1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[2] = x[2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.conv1(x)
        # 149 x 149 x 32
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        # 147 x 147 x 32
        x = self.layer1(x)
        # 147 x 147 x 64
        x = self.layer2(x)
        # 73 x 73 x 64
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, kernel_size=7, stride=7)

        return x

model = torchvision.models.resnet50(pretrained = True) 

model.train(False)

my_resnet = MyResNetFeatureExtractor(model)

def extractor(data):
    since = time.time()
    list_imgs_names = os.listdir(data)
    N = len(list_imgs_names)
    fea_all = np.zeros((N, 2048))
    image_all = [] 
    for ind, img_name in enumerate(list_imgs_names):
        img_path = os.path.join(data, img_name)
        image_np = Image.open(img_path)
        image_np = np.array(image_np)
        image_np = resize(image_np, (224, 227))
        image_np = torch.from_numpy(image_np).permute(2, 0, 1).float()
        image_np = Variable(image_np.unsqueeze(0))
        fea = my_resnet(image_np)
        fea = fea.squeeze()
        fea = fea.cpu().data.numpy()
        fea = fea.reshape((1, 2048))
        fea = fea / LA.norm(fea)
        fea_all[ind] = fea
        image_all.append(img_name)
        
    time_elapsed = time.time() - since 

    st.write('Feature extraction complete in {:.02f}s'.format(time_elapsed % 60))

    return fea_all, image_all

