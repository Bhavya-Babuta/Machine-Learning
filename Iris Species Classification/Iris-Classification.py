#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 10:57:07 2018

@author: bhavyababuta
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
data=pd.read_csv('iris.csv')

data.info() # we understand that there are no null values
data.head(20)
data.describe(include='all')
print('No of Features', data.shape[1])
print('No of Observations' , data.shape[0])

print('No of null values in the dataset:\n',data.isnull().sum()) 

print('No of Numerical Feature', data.dtypes[data.dtypes!='object'].size)
print('No of Categorial Feature', data.dtypes[data.dtypes=='object'].size)

print("Name of the feature\n")
print(data.columns)

X=data.loc[:,'Id':'PetalWidthCm']
Y=data.loc[:,'Species']

plt.figure()
class_distribution=Y.value_counts()
class_label=pd.DataFrame(class_distribution,columns=['Species'])
sns.barplot(x=class_distribution.index,y=data.groupby('Species').size(),data=class_label)

for i,num in enumerate(class_distribution):
    per=(num/class_distribution.sum())*100
    print('Percentage of Iris Species %s in the Entire Dataset : ' %class_distribution.index[i] )
    print('%.2f' % per ,'%')

X.drop('Id',axis=1,inplace=True)
data.drop('Id',axis=1,inplace=True)

for i,col in enumerate(X):
    plt.figure(i)
    sns.distplot(X[col]) 
    
plt.figure(figsize=(8,8))
sns.heatmap(data.corr(),cmap='magma',linecolor='white',linewidths=1,annot=True)


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=101)

classifiers = [LinearSVC() , DecisionTreeClassifier() , LogisticRegression() , GaussianNB() ,RandomForestClassifier() , 
     GradientBoostingClassifier()]
accuracyscore=[]
for i in classifiers:
    model=i
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    accuracyscore.append(accuracy_score(prediction,y_test))
    
dic = { "Algorithm" : classifiers, "Accuracy" : accuracyscore }
accuracy_dataFrame=pd.DataFrame(dic)

sns.barplot(x=classifiers,y=accuracy_dataFrame['Accuracy'],data=accuracy_dataFrame)
sns.distplot(accuracy_dataFrame['Accuracy'])

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