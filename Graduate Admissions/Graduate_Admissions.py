#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:12:38 2019

@author: bhavyababuta
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
%matplotlib inline
from scipy import stats
from scipy.stats import norm
import math

train=pd.read_csv('Admission_Predict_Ver1.1.csv')
test=pd.read_csv('Admission_Predict.csv')

print("Number of Columns in Training Dataset: ",train.shape[1])
print("Percentage of Null Values")
print(train.isnull().sum()/train.shape[0]*100)
print("Number of Numerical Features: ",train.dtypes[train.dtypes!=object].size)

Y=train['Chance of Admit ']
X=train.drop('Chance of Admit ',axis=1)
for i in ['GRE Score','TOEFL Score','SOP','LOR ']:
    plt.figure()
    sns.distplot(X[i]).set_title(i)
    
X=X.drop('Serial No.',axis=1)
test=test.drop('Serial No.',axis=1)

print(X.dtypes)
sns.countplot(y="University Rating",data=X)
sns.countplot(y="Research",data=X)
sns.countplot(y="SOP",data=X)
sns.countplot(y="LOR ",data=X)

for i in ['GRE Score','TOEFL Score','SOP','LOR ','University Rating']:
    plt.figure()
    sns.lineplot(X[i],Y).set_title(i)

X['GRE Score'].describe()
X['TOEFL Score'].describe()
X['SOP'].describe()
X['LOR '].describe()
plt.figure()
sns.scatterplot(data=X,x='GRE Score',y='TOEFL Score',hue='Research')
plt.figure()
sns.scatterplot(data=X,x='GRE Score',y=Y)
plt.figure()
sns.scatterplot(data=X,x='TOEFL Score',y=Y)

fig = plt.figure()
res = stats.probplot(Y, plot=plt)
plt.show()
sns.distplot(Y)
sns.distplot(np.log1p(Y), fit=norm)


plt.figure(figsize=(8,5))
sns.heatmap(pd.concat([X,Y],axis=1).corr())

X=pd.concat([pd.DataFrame(np.ones((X.shape[0],1))),X],axis=1)
X.drop('Research',axis=1,inplace=True)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y,test_size = 0.20)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_val=sc_X.transform(x_val)

def r2_score(Y,Yhat):
    d1=Y-Yhat
    d2=Y-Y.mean()
    r2=1-d1.dot(d1)/d2.dot(d2)
    return r2

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01,max_iter=10e5)
lasso.fit(x_train,y_train)
train_score=lasso.score(x_train,y_train)
test_score=lasso.score(x_val,y_val)
coeff_used = np.sum(lasso.coef_!=0)  


test_y=test['Chance of Admit ']
test_x=test[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA']]
test_x=pd.concat([pd.DataFrame(np.ones((test_x.shape[0],1))),test_x],axis=1)
test_x=sc_X.transform(test_x)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

Accuracy=[]
Model=[]
Prediction=[]
MSE=[]
Classifiers = [DecisionTreeRegressor(random_state = 42),
               RandomForestRegressor(n_estimators = 100, random_state = 42),
               LinearRegression(),
               XGBRegressor(max_depth = 6)]
for classifier in Classifiers:
    fit=classifier.fit(x_train,y_train)
    pred=classifier.predict(x_val) 
    Prediction.append(r2_score(test_y,classifier.predict(test_x)))
    Accuracy.append(r2_score(y_val,pred))
    Model.append(classifier.__class__.__name__)
    print('MSE :',classifier.__class__.__name__,'is ',mean_squared_error(pred,y_val))
    MSE.append(mean_squared_error(pred,y_val))
    print('Accuracy of '+classifier.__class__.__name__+'is '+str(r2_score(y_val,pred)))

plt.figure()
plt.bar(Model,Accuracy)
plt.xticks(rotation=90)
plt.ylabel('Accuracy On Validation Dataset')
plt.xlabel('Model')
plt.title('Accuracy Bar') 

plt.figure()
plt.bar(Model,Prediction)
plt.xticks(rotation=90)
plt.ylabel('Accuracy on Test Dataset')
plt.xlabel('Model')
plt.title('Accuracy Bar') 
     
plt.figure()
plt.bar(Model,MSE)
plt.xticks(rotation=90)
plt.ylabel('MSE')
plt.xlabel('Model')
plt.title('MSE Bar')

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators = 1000,random_state = 123)
rf_model = RandomForestRegressor(n_estimators = 1000,random_state = 123)
rf_model.fit(x_train,y_train)
feature_importance = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X.columns)), columns=['Value','Feature'])
plt.figure(figsize=(10, 6))
sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))
plt.xlabel('Value',fontsize=20)
plt.ylabel('Feature',fontsize=20)
plt.title('Random Forest Feature Importance',fontsize=25)
plt.grid()
plt.ioff()
plt.tight_layout()