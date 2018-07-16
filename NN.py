import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation
import csv
import string
from collections import Counter
from tqdm import tqdm
import collections, re
import random
from random import randint
from sklearn.metrics import average_precision_score
import pandas as pd
from scipy import misc as cv2
import glob
import tensorflow as tf
from PIL import Image
from skimage import transform
import copy
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import os
import time
import imageio
import plotly.plotly as py
import plotly.graph_objs as go
import csv


path="./Model/model.tflearn.meta"



if os.path.exists(path):

    tf.reset_default_graph()
    convnet=input_data(shape=[None,784],name='input')

    convnet=fully_connected(convnet,32,activation='relu')
    convnet=fully_connected(convnet,64,activation='relu')
    convnet=fully_connected(convnet,128,activation='relu')
    convnet=fully_connected(convnet,32,activation='relu')

    convnet=dropout(convnet,0.7)
    convnet=fully_connected(convnet,4,activation='softmax')
    convnet=regression(convnet,optimizer='adam',learning_rate=0.005,loss='categorical_crossentropy',name='DigitsClassifier')
    model=tflearn.DNN(convnet,tensorboard_dir='log',tensorboard_verbose=0)
    print("Loading the model")
    model.load('./model.tflearn')


    #testing here
    df2=pd.read_csv('testData.csv')

    print("Getting features of test data..")
    testX=[]
    with tqdm(total=len(list(df2.iterrows()))) as pbar:
        for i in range(len(list(df2.iterrows()))):
            testX.append(df2.loc[i])
            pbar.update(1)
    i=0
    x=0

    print("Predicting...")
    predictions=[]
    for i in range(len(testX)):
        model_out=model.predict([testX[i]])[0]
        if np.argmax(model_out) ==0:
            x=1
        elif np.argmax(model_out) ==1:
            x=3
        elif np.argmax(model_out) ==2:
            x=4
        else:
            x=8
        predictions.append([i+1,x])
        

    csvfile = "mysubmissionNN.csv"

    i=0
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        if(i==0):
            writer.writerow(["ID","label"])
        i+=1
        writer.writerows(predictions)

    print("data dumped to file")


else:
    df=pd.read_csv('trainData.csv')       #getting file

    df.fillna(' ', inplace=True)     #all NAN are replaced by ' '

    #print(df.head())

    X=[]
    print("Getting features..")
    with tqdm(total=len(list(df.iterrows()))) as pbar:
        for i in range(len(list(df.iterrows()))):
            X.append(df.loc[i])
            pbar.update(1)


    #print(len(X))

    df1=pd.read_csv('trainLabels.csv')

    df1.fillna(' ',inplace=True)

    y=[]
    i=0
    print("Getting labels..")
    with tqdm(total=len(list(df1.iterrows()))) as pbar:
        for i in range(len(list(df1.iterrows()))):
            #print(df1.loc[i]['Label'])
            if df1.loc[i]['Label']==1:
                tr=[1,0,0,0]
            elif df1.loc[i]['Label']==3:
                tr=[0,1,0,0]
            elif df1.loc[i]['Label']==4:
                tr=[0,0,1,0]
            elif df1.loc[i]['Label']==8:
                tr=[0,0,0,1]
            else:
                print("blekh ")
            y.append(tr)
            pbar.update(1)
    #labels are 1,3,4,8 in one hot form
    #data is in a list size is 784


    X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.1)



    tf.reset_default_graph()
    convnet=input_data(shape=[None,784],name='input')
    
    convnet=fully_connected(convnet,32,activation='relu')
    convnet=fully_connected(convnet,64,activation='relu')
    convnet=fully_connected(convnet,128,activation='relu')
    convnet=fully_connected(convnet,32,activation='relu')

    convnet=dropout(convnet,0.7)
    convnet=fully_connected(convnet,4,activation='softmax')
    convnet=regression(convnet,optimizer='adam',learning_rate=0.005,loss='categorical_crossentropy',name='DigitsClassifier')
    model=tflearn.DNN(convnet,tensorboard_dir='log',tensorboard_verbose=0)
    model.fit(X_train,y_train, n_epoch=15,validation_set=(X_test,y_test), snapshot_step=20,show_metric=True,run_id='DigitsClassifier')
    print("Saving the model")
    model.save('model.tflearn')

    #testing here
    df2=pd.read_csv('testData.csv')

    print("Getting features of test data..")
    testX=[]
    with tqdm(total=len(list(df2.iterrows()))) as pbar:
        for i in range(len(list(df2.iterrows()))):
            testX.append(df2.loc[i])
            pbar.update(1)
    i=0
    x=0
    predictions=[]
    print("Predicting...")
    for i in range(len(testX)):
        model_out=model.predict([testX[i]])[0]
        if np.argmax(model_out) ==0:
            x=1
        elif np.argmax(model_out) ==1:
            x=3
        elif np.argmax(model_out) ==2:
            x=4
        else:
            x=8
        predictions.append([i+1,x])
        
    csvfile = "NN_results.csv"

    i=0
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        if(i==0):
            writer.writerow(["ID","label"])
        i+=1
        writer.writerows(predictions)

    print("data dumped to file")








