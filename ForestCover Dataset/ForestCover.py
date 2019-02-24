#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:08:54 2018

@author: bhavyababuta
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 

data=pd.read_csv('covtype.csv')

data.info()
data.describe(include='all')
data.head(20)

print('No of Rows')
print(data.shape[0])
print('No of Columns')
print(data.shape[1])

print('No of Numerical Features:')
print(data.dtypes[data.dtypes!="object"].size)

print('No of Numerical Features:')
print(data.dtypes[data.dtypes=="object"].size)

print('Columns Names')
print(data.columns)

print('Null Values')
print(data.isnull().sum())

data.groupby('Cover_Type').size()

plti.figure(figsize=(8,6))
class_dist=data.groupby('Cover_Type').size()
class_label=pd.DataFrame(class_dist,columns=['Size'])
sns.barplot(x=data['Cover_Type'].unique(),y=data.groupby('Cover_Type').size(),data=class_label)


for i,num in enumerate(class_dist):
    per=(num/class_dist.sum())*100
    print('Percentage of Cover Type %s in the Entire Dataset : ' %class_dist.index[i] )
    print('%.2f' % per ,'%')
    
continousdata=data.loc[:,'Elevation':'Horizontal_Distance_To_Fire_Points']
binarydata=data.loc[:,'Wilderness_Area1':'Soil_Type40']
Wilderness_data=data.loc[:,'Wilderness_Area1': 'Wilderness_Area4']
Soil_data=data.loc[:,'Soil_Type1':'Soil_Type40']

for col in binarydata:
    count=binarydata[col].value_counts()
    print(col,count)
    
for i, col in enumerate(continousdata):
    plt.figure(i)
    sns.distplot(continousdata[col])
    
for i, col in enumerate(continousdata.columns):
    plt.figure(i,figsize=(8,4))
    sns.boxplot(x=data['Cover_Type'], y=col, data=data, palette="coolwarm")
    
for i, col in enumerate(binarydata.columns):
    plt.figure(i,figsize=(6,4))
    sns.countplot(x=col, hue=data['Cover_Type'] ,data=data, palette="rainbow")
    
plt.figure(figsize=(15,8))
sns.heatmap(continousdata.corr(),cmap='magma',linecolor='white',linewidths=1,annot=True)

g = sns.PairGrid(continousdata)
g.map(plt.scatter)

X=data.loc[:,'Elevation':'Soil_Type40']
y=data['Cover_Type']
rem=['Hillshade_3pm','Soil_Type7','Soil_Type8','Soil_Type14','Soil_Type15',
     'Soil_Type21','Soil_Type25','Soil_Type28','Soil_Type36','Soil_Type37']

X.drop(rem, axis=1, inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)
from sklearn.neighbors import KNeighborsClassifier

neighbors = np.arange(1,7)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 
    
plt.figure(figsize=(10,6))
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
#plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
Accuracy=knn.score(X_test,y_test)
print('KNN Accuracy:',Accuracy)