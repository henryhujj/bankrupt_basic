# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 21:19:21 2021

@author: a0952
"""
import pandas as pd
from scipy.io import arff
import pandas as pd, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#from file arff file create csv training data

def transform(fpath):
    f = open(fpath)
    line = f.readlines()
    content =[]
    for i in line:
        content.append(i)
    data = []
    for j in content:
        cs = j.split(",")
        data.append(cs)
    df = pd.DataFrame(data=data,index=None,columns=None)
    filename = fpath[:fpath.find('.arff')]+'.csv'
    df.to_csv(filename, index = None)
    print("sucess")
    
def spec(train):
    for  i in range(len(train)):
        for j in range(65):
            if train[i][j] =="?" or train[i][j] =='':
                train[i][j]=0
    train = np.array(train[1:])[:, 1:].astype(float)
    print("done")
    return(train)

with open('Data/1year.csv', 'r') as fp:
    train1= list(csv.reader(fp))
with open('Data/2year.csv', 'r') as fp:
    train2= list(csv.reader(fp))
with open('Data/3year.csv', 'r') as fp:
    train3= list(csv.reader(fp))
with open('Data/4year.csv', 'r') as fp:
    train4= list(csv.reader(fp))
    
train1=spec(train1)
train2=spec(train2)
train3=spec(train3)
train4=spec(train4)

final_train = np.concatenate([train1[6485:],train2[9373:],train3[9513:],train4[8712:]])
df = pd.DataFrame(data = final_train,index=None,columns=None)
df.to_csv('Data/final_train.csv',index=None)

#create testing data
with open('Data/5year.csv', 'r') as fp:
    train5= list(csv.reader(fp))
train5=spec(train5)
df = pd.DataFrame(data = train5,index=None,columns=None)
df.to_csv('Data/final_test.csv',index=None)










