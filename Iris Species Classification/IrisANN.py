#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:43:03 2018

@author: bhavyababuta
"""
#Classification using Artifical Neural Networks
"""

Input Layer : 5 neurons
Hidden Layer : 5 neurons
Output Layer : 3 neurons (One for each class)
Scaling training data using sklearn.preprocessing StandardScaler
Sigmoid Function
Sigmoid Gradient 
Gradient Descend
Cost Funtion
Optimation
"""


import numpy as np 
import pandas as pd 
from scipy import optimize as opt
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('Iris.csv')
Y=data.iloc[:,-1]

print(data.shape)
print('Number of Feature :',data.shape[1])
print('Numberof Observations :',data.shape[0])

print('Class Distribution',data.iloc[:,-1].value_counts())
class_dist=data.groupby('Species').size()
sns.barplot(x=class_dist.index,y=data.groupby('Species').size())
labels=list(data.iloc[:,-1].unique())
print('Labels :',labels)
#One Hot Encoding
Y = pd.get_dummies(data.iloc[:,-1])
data.head(20)

y_d=np.zeros((Y.shape[0],1))
for i in range(y_d.shape[0]):
    y_d[i]=labels.index(data.iloc[i,-1])
y_d=y_d+1
X=np.array(data.iloc[:,0:-1])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

    
input_layer=X.shape[1]
hidden_layer=5
lmbda=1
num_class=(data.iloc[:,-1].unique().shape[0])

#Initialize Random Weights
def randInitializeWeights(L_in, L_out):
    epsilon = 0.15
    return np.random.rand(L_out, L_in+1) * 2 * epsilon - epsilon
w1=randInitializeWeights(input_layer,hidden_layer)
w2=randInitializeWeights(hidden_layer,num_class)
nn_initial_params = np.hstack((w1.ravel(order='F'), w2.ravel(order='F')))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    return np.multiply(sigmoid(z),1-sigmoid(z))

def Cost_Function(nn_params,input_layer, hidden_layer, num_classs,X,Y,lmbda):
    w1 = np.reshape(nn_params[:hidden_layer*(input_layer+1)], (hidden_layer, input_layer+1), 'F')
    w2 = np.reshape(nn_params[hidden_layer*(input_layer+1):], (num_class, hidden_layer+1), 'F')
    
    m=len(Y)
    ones=np.zeros((m,1))
    a1=np.hstack((ones,X))
    
    a2=sigmoid(a1.dot(w1.T))
    a2=np.hstack((ones,a2))
    h=sigmoid(a2.dot(w2.T))

    temp1 = np.multiply(Y, np.log(h))
    temp2 = np.multiply(1-Y, np.log(1-h))
    temp3 = np.sum(temp1 + temp2)
    
    sum1 = np.sum(np.sum(np.power(w1[:,1:],2), axis = 1))
    sum2 = np.sum(np.sum(np.power(w2[:,1:],2), axis = 1))
    
    return np.sum(temp3 / (-m)) + (sum1 + sum2) * lmbda / (2*m)
    

a=Cost_Function(nn_initial_params,input_layer,hidden_layer,num_class,X,Y,lmbda)
    
def Gradient(nn_params,input_layer, hidden_layer,num_labels, X,Y,lmbda):
    w1 = np.reshape(nn_params[:hidden_layer*(input_layer+1)], (hidden_layer, input_layer+1), 'F')
    w2 = np.reshape(nn_params[hidden_layer*(input_layer+1):], (num_class, hidden_layer+1), 'F')

    delta1=np.zeros(w1.shape)
    delta2=np.zeros(w2.shape)
    m=len(Y)
    for i in range(X.shape[0]):
        one=np.ones(1)
        a1=np.hstack((one,X[i]))
        z2=a1.dot(w1.T)
        a2=np.hstack((one,sigmoid(z2)))
        z2 = np.hstack((one, z2))
        z3=a2.dot(w2.T)
        h=sigmoid(z3)
        
        #Error at the output layer
        d=h-Y.iloc[i,:][np.newaxis,:]
        

        #Error at the second layer
        d2 = np.multiply(w2.T @ d.T, sigmoidGradient(z2).T[:,np.newaxis])
        delta1 = delta1 + d2[1:,:].dot(a1[np.newaxis,:])
        delta2 = delta2 + d.T.dot(a2[np.newaxis,:])
        
    delta1 /= m
    delta2 /= m
    
    delta1[:,1:] = delta1[:,1:] + w1[:,1:] * lmbda / m
    delta2[:,1:] = delta2[:,1:] + w2[:,1:] * lmbda / m
        
    return np.hstack((delta1.ravel(order='F'), delta2.ravel(order='F')))

a=Gradient(nn_initial_params,input_layer,hidden_layer,num_class,X,Y,lmbda)

    

theta_opt = opt.fmin_cg(maxiter = 50, f = Cost_Function, x0 = nn_initial_params, fprime = Gradient, \
                        args = (input_layer, hidden_layer, num_class, X, Y, lmbda))

theta1_opt = np.reshape(theta_opt[:hidden_layer*(input_layer+1)], (hidden_layer, input_layer+1), 'F')
theta2_opt = np.reshape(theta_opt[hidden_layer*(input_layer+1):], (num_class, hidden_layer+1), 'F')

def predict(theta1,theta2,X,Y):
    m = len(Y)
    ones = np.ones((m,1))
    a1 = np.hstack((ones, X))
    a2 = sigmoid(a1.dot(theta1.T))
    a2 = np.hstack((ones, a2))
    h = sigmoid(a2.dot(theta2.T))
    return np.argmax(h, axis = 1) + 1

pY=predict(theta1_opt,theta2_opt,X,Y)

def classification_rate(pY,y_d):
    count=0
    for i in range(len(pY)):
        if pY[i] == y_d[i]:
            count+=1
        else:
            continue
    return (count/len(pY))*100
    
acc=classification_rate(pY,y_d)