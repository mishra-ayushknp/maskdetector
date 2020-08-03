#importing libraries
import numpy as np 
import os
import cv2
import random
import pickle
from keras.utils import np_utils

t_d = []
ddir = "C:\\Users\\AYUSH MISHRA\\Desktop" # Path directory where the dataset is stored. 
categories = ["with_mask","without_mask"]  # covid19 and Normal datset .
for c in categories :
    path = os.path.join(ddir, c)  # joining the path of the covid19 and normal image to the Path directory .
    c_n = categories.index(c)    
    for i in os.listdir(path):
        img = cv2.imread(os.path.join(path,i))  
        imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_resize = cv2.resize(imgg ,(100,100))   #  resizing the image into (150,150)
        t_d.append([img_resize,c_n])
random.shuffle(t_d)                # shuffling the data in random manner
print(len(t_d))      # printing the length of the training data
x = []
y = []
for f , l in t_d :
    x.append(f)
    y.append(l)

xen = np.array(x).reshape(-1,100,100,1)    # reshaping the image into four dimension
yen = np.array(y)
new_target = np_utils.to_categorical(yen)
np.save('xen',xen)
np.save('target',new_target)
print(xen.shape,yen.shape)

