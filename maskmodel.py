from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Dense , Activation , Dropout , Flatten
from tensorflow.keras.models import Sequential
import pickle 
import numpy as np 
import os
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
# loading training dataset using pickle 
data = np.load('xen.npy')
target = np.load('target.npy')
data = data/255.0
model = Sequential()
model.add(Conv2D(200,(3,3),input_shape = data.shape[1:]) )     # regularizing datasets
model.add(Activation("relu"))                 # relu removes the negative part
model.add(MaxPooling2D(pool_size =(2,2)))         # It reduces the amount of parameter and computation in the network
model.add(Conv2D(100,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size =(2,2)))
         # It reduces overfitting and increases the accuracy of the training datasets

model.add(Flatten())        #  It convert the datasets into 1D array
model.add(Dropout(0.5))  
model.add(Dense(50))          
model.add(Activation("relu"))
model.add(Dense(2))          
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy',optimizer = "adam",metrics = ['accuracy'])

train_data , test_data ,train_target , test_target = train_test_split(data,target,test_size = 0.1)
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor= 'val_loss',verbose = 0,save_best_only = True ,mode='auto')
model.fit(train_data,train_target,callbacks=[checkpoint],epochs = 20,validation_split = 0.2)   
model.save('mask.model') # saving the model

