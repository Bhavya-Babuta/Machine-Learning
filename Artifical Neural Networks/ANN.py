#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 18:26:39 2018

@author: bhavyababuta
"""

import numpy as np
import pandas as pd

df=pd.read_csv('Churn_Modelling.csv')
data=df.values

x=data[:,:-1]
y=data[:,-1]

X=np.array(x)
Y=np.array(y)

X=X[:,3:13]

X2=np.zeros((X.shape[0],X.shape[1]+2))
X2[:,0]=X[:,0]
#OneHotEncoder , LabelEncoder
for i in range(X.shape[0]):
    if X[i,1]=='France':
        X[i,1]=0
    if X[i,1]=='Spain':
        X[i,1]=1
    if X[i,1]=='Germany':
        X[i,1]=2
    if X[i,2]=='Male':
        X[i,2]=0
    if X[i,2]=='Female':
        X[i,2]=1
    t=X[i,1]
    X2[i,t+1]=1
X2[:,4:12]=X[:,2:10]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X2 = sc.fit_transform(X2)

from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
classifier.add(Dense(units=8,kernel_initializer='uniform',activation='relu',input_dim=12))
classifier.add(Dense(units=8,kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(units=8,kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X2,Y,epochs=100,batch_size=10)
new=classifier.predict(X2)
new= np.round(new)
prediction=classifier.predict(np.array([[511,0,1,0,1,66,4,0,1,1,0,1643.11]]))
print(prediction>0.5)