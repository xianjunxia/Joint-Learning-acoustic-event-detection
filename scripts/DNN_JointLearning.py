from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import theano
import keras.backend as K

import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import RMSprop, Adadelta
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten
from keras.layers import Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.models import model_from_json
from keras import regularizers
from keras.constraints import maxnorm

from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

import numpy as np

K.set_image_dim_ordering('th')

maxToAdd = 1
DatDim = 40
AggreNum = 10

learning_rate = 0.001
Train_Epoch = 50

MLP = 1

Class = 0
Reg = 1
Joint = 0


def joint_error(y_true, y_pred):
    Error = K.binary_crossentropy(y_pred[:,0:6], y_true[:,0:6])
    R_Error = K.square(y_pred[:,6:12] - y_true[:,6:12])
    return 0.5*Error + 0.5*R_Error
   
def AggreData(feature):
    fvecs = np.ones((feature.shape[0],DatDim*AggreNum))    
    for i in range(feature.shape[0]-AggreNum+1):
        for j in range(AggreNum):
            fvecs[i,j*DatDim:(j+1)*DatDim] = feature[i+j,0:DatDim]    
    for a in range(AggreNum-1):
        i = i + 1
        z = np.zeros((1,DatDim))       
        k =0        
        fvecs_temp = np.zeros((1,AggreNum*DatDim))     
        fvecs_temp[0,0:DatDim] = feature[i+k,0:DatDim]
        k = k + 1
        while k<AggreNum-a-1:
            fvecs_temp[0,k*DatDim:(k+1)*DatDim] = feature[i+k,0:DatDim]            
            k = k+ 1
        fvecs[i,:] = fvecs_temp
    fvecs_np = np.array(fvecs).astype(np.float32)
    return fvecs_np    
def extract_data_feature(filename_feature):    
    labels = []
    fvecs = []
    fvecl = []
    file = open(filename_feature)
    i = 0
    for line in file:
        row = line.split(",")
        i = i + 1
        fvecs.append([float(x) for x in row[0:DatDim]])      
    fvecs_np = np.array(fvecs).astype(np.float32)      
    return fvecs_np,i
 
def CopyRegressionFun(row):
    fvecs = np.ones((6))
    for i in range(6):
        fvecs[i] = row[46+2*i]
    fvecs_np = np.array(fvecs).astype(np.float32)      
    return fvecs_np

def CopyClassificationFun(row):
    fvecs = np.ones((6))
    for i in range(6):
        fvecs[i] = row[40+1*i]
    fvecs_np = np.array(fvecs).astype(np.float32)      
    return fvecs_np    
    
def CopyJointFun(row):
    fvecs = np.ones((12))
    for i in range(6):
        fvecs[i] = row[40+i]
    for i in range(6):
        fvecs[i+6] = row[46+2*i]        
        if(fvecs[i+6]<0.5):
            fvecs[i] = 0 
    fvecs_np = np.array(fvecs).astype(np.float32)      
    return fvecs_np  
     
    
def CopyFun(row):
    fvecs = np.ones((6))
    for i in range(6):
        fvecs[i] = row[46+2*i]
    fvecs_np = np.array(fvecs).astype(np.float32)      
    return fvecs_np            

def extract_data_label(filename_feature):

    # Arrays to hold the labels and feature vectors.    
    labels = []
    fvecs = []
    fvecl = []
    file = open(filename_feature)
    i = 0
    for line in file:
        row = line.split(",")
        i = i + 1
        if(Class == 1):
            fvecl.append(CopyClassificationFun(row))
        if(Reg == 1):
            fvecl.append(CopyJointFun(row))        
        if(Joint == 1):
            fvecl.append(CopyJointFun(row))
    fvecs_label = np.array(fvecl).astype(np.float)
    return fvecs_label,i    
  
class_weight = np.array([1, 0.1, 1, 0.3, 0.3, 0.15])
TrainLabel, TrainNum = extract_data_label('Training Data/TrainDat_mel_scale.txt')
TestLabel, TestNum = extract_data_label('Training Data/TestDat_mel_scale.txt')

TrainData, TrainNum = extract_data_feature('Training Data/TrainDat_mel_scale.txt')
TestData, TestNum = extract_data_feature('Training Data/TestDat_mel_scale.txt')
  

TrainData = AggreData(TrainData)
TestData = AggreData(TestData)
################################################################################# MLP based
if(MLP == 1):
    
    first_model = Sequential()
    first_model.add(Dense(30, activation='relu', input_dim = AggreNum*DatDim))
    first_model.add(BatchNormalization())
    first_model.add(Dropout(0.2)) 

    first_model.summary()
    second_model = Sequential() 
    second_model.add(Dense(30, activation='relu', input_dim = AggreNum*DatDim)) 
    second_model.add(BatchNormalization())
    second_model.add(Dropout(0.2))
    second_model.add(Dense(30, activation='relu')) 
    second_model.add(BatchNormalization())
    second_model.add(Dropout(0.2))    

    second_model.summary()
    model = Sequential()
    model.add(Merge([first_model, second_model], mode='concat'))
    model.add(Dense(50, activation='relu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(12,activation = 'sigmoid'))
    model.summary()
    
X_train = TrainData
X_test = TestData
y_train = TrainLabel
y_test = TestLabel
optimizer = Adam(lr=learning_rate)
print("X_train shape: ",X_train.shape)
print("y_train shape: ",y_train.shape)
if(Class == 1):
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    check = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.5f}.hdf5", monitor='accuracy')
    model.fit(X_train, y_train, batch_size=200,shuffle=True, callbacks = [check],validation_split=0.2,class_weight=class_weight,nb_epoch=Train_Epoch)
if(Reg == 1):
    model.compile(loss='mse', optimizer=optimizer,metrics=['mse'])
    check = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.5f}.hdf5", monitor='mse')
    model.fit([X_train,X_train], y_train, batch_size= 400,shuffle=True, callbacks = [check],validation_split=0.2,class_weight=class_weight,nb_epoch=Train_Epoch)
if(Joint == 1):
    model.compile(loss=joint_error, optimizer=optimizer,metrics=['mse'])
    check = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.5f}.hdf5", monitor='mse')
    model.fit([X_train,X_train], y_train, batch_size=400,shuffle=True, callbacks = [check],validation_split=0.2,class_weight=class_weight,nb_epoch=Train_Epoch)
######################################### save the model      
model_json = model.to_json()
with open("DNN.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("DNN.h5")
print("Saved model to disk")
