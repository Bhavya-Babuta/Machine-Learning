#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 23:12:10 2018

@author: bhavyababuta
"""

import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 

data=pd.read_csv('mushrooms.csv')

data.info()
data.head(20)
data.describe(include='all')

numerical_feats = data.dtypes[data.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = data.dtypes[data.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))

data['class'].unique()
data['stalk-color-above-ring'].unique()
print(data.groupby('class').size())
data['class'].value_counts()

data.corr()
# =============================================================================
# def LabelEncoder(list):
#     labels=[]
#     for i in range(len(list)):
#         if list[i] not in labels:
#             labels.append(list[i])
#     print(labels)
#     for i in range(len(list)):
#         list[i]=labels.index(str(list[i]))
#     return list
# =============================================================================

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

data = data.apply(label_encoder.fit_transform)

numerical_feats = data.dtypes[data.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = data.dtypes[data.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))

Y_data=data['class']
data[['class','cap-shape']].groupby(['class']).mean().sort_values(by='cap-shape',ascending=False).plot.bar()
data[['cap-shape','class']].groupby(['cap-shape']).mean().sort_values(by='class',ascending=False).plot.bar()

data[['class','cap-surface']].groupby(['class']).mean().sort_values(by='cap-surface',ascending=False).plot.bar()
data[['cap-surface','class']].groupby(['cap-surface']).mean().sort_values(by='class',ascending=False).plot.bar()

data[['cap-color','class']].groupby(['cap-color']).mean().sort_values(by='class',ascending=False).plot.bar()
data[['cap-color','class']].groupby(['cap-color']).mean().sort_values(by='class',ascending=False).plot.bar()

data.drop(["class"],axis=1,inplace=True)  


X_data = (data - np.min(data))/(np.max(data)-np.min(data)).values

#This column contains only NaN values
X_data.drop(["veil-type"],axis=1,inplace=True)
#This column contains only the same values
X_data.drop(["veil-color"],axis=1,inplace=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)
predicted=lr_model.predict(x_test)
print("test accuracy: ", lr_model.score(x_test,y_test))#0.9458461538461539*100

def classification_rate(y,Py):
    return np.mean(y==Py)

print('Classification Rate:' ,classification_rate(predicted,y_test)) #0.9458461538461539*100

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

model_tree = DecisionTreeClassifier()
model_tree.fit(x_train, y_train)
pred=model_tree.predict(x_test)
print('Classification Rate:' ,classification_rate(pred,y_test))#1.0*100

#ANN
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
mlp.fit(x_train,y_train)
predNN=mlp.predict(x_test)
print('Classification Rate:' ,classification_rate(predNN,y_test))#1.0*100

x_train.insert(loc=0,column='Ones',value=np.ones((x_train.shape[0])))
w=np.linalg.solve(np.dot(x_train.T,x_train),np.dot(x_train.T,y_train))
Yhat=np.dot(x_train,w)
d1=y_train-np.round(Yhat)
d2=y_train-y_train.mean()
r2=1-d1.dot(d1)/d2.dot(d2)
print('Classification Rate:' ,classification_rate(np.round(Yhat),y_train)) #0.9478381289429143*100

print("The R Squared is" ,r2) #The R Squared is 0.7910885831568812

