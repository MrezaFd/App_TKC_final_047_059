 #Import libraries 
from FeatureExtractor import extractor
import streamlit as st
import numpy as np
import torch.nn.functional as F
import os
from os import path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

path = './aaaa'        
feats, image_list = extractor(path)

def load_image(image_file):
	img = Image.open(image_file)
	return img

image_file = st.file_uploader("Choose a image file", type=['png','jpg','jpeg'])

if image_file is not None:
    img = load_image(image_file)
    with open(os.path.join("test",image_file.name),"wb") as f: 
      f.write(image_file.getbuffer())         
    image = Image.open(image_file)
    st.image(image, caption='QUERY Image. Source from Google')
    st.success("Saved File to Directory test")
    
    Genrate_pred = st.button("Generate Retrieval")   
    if Genrate_pred:
        # test image path
        test = './test'
        feat_single, image = extractor(test)
        scores  = np.dot(feat_single, feats.T)
        sort_ind = np.argsort(scores)[0][::-1]
        scores = scores[0, sort_ind]

        maxres = 10
        imlist = [image_list[index] for i, index in enumerate(sort_ind[0:maxres])]

        fig=plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('xkcd:white')

        for i in range(len(imlist)):
            sample = imlist[i]
            img = mpimg.imread('./aaaa' + '/' + sample)
            #ax = plt.subplot(figsize)
            ax = fig.add_subplot(2, 5, i+1)
            ax.autoscale()
            plt.tight_layout()
            plt.imshow(img, interpolation='nearest')
            ax.set_title('{:.3f}%'.format(scores[i]))
            ax.axis('off')
        st.pyplot(fig)
        
        dir = './test'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))