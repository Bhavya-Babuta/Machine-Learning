#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 00:15:32 2018

@author: bhavyababuta
"""

import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from scipy.io import loadmat
from scipy import optimize as opt
data=loadmat('ex4data1.mat')
X=data['X']
Y=data['y']
#_, axarr = plt.subplots(10,10,figsize=(10,10))
#for i in range(10):
#    for j in range(10):
#        axarr[i,j].imshow(X[np.random.randint(X.shape[0])].\reshape((20,20), order = 'F'))          
#        axarr[i,j].axis('off')
      
weights=loadmat('ex4weights.mat')
w1=weights['Theta1']
w2=weights['Theta2']

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lmbda = 1

nn_params = np.hstack((w1.ravel(order='F'), w2.ravel(order='F')))  

#Sigmoid Function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#Derivative or Gradient of Sigmoid Function
def sigmoidGrad(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z))

#Initialize random weights :
def randInitializeWeights(L_in, L_out):
    epsilon = 0.12
    return np.random.rand(L_out, L_in+1) * 2 * epsilon - epsilon

initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# unrolling parameters into a single column vector
nn_initial_params = np.hstack((initial_theta1.ravel(order='F'), initial_theta2.ravel(order='F')))

def nnCostFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    
    theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1), 'F')
    theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1), 'F')

    m = len(y)
    ones = np.ones((m,1))
    a1 = np.hstack((ones, X))
    a2 = sigmoid(a1.dot(theta1.T))
    a2 = np.hstack((ones, a2))
    h = sigmoid(a2.dot(theta2.T))
    
    y_d = pd.get_dummies(y.flatten())
   
    #Cross Entropy Error
    temp1 = np.multiply(y_d, np.log(h))
    temp2 = np.multiply(1-y_d, np.log(1-h))
    temp3 = np.sum(temp1 + temp2)
    
    sum1 = np.sum(np.sum(np.power(theta1[:,1:],2), axis = 1))
    sum2 = np.sum(np.sum(np.power(theta2[:,1:],2), axis = 1))
    
    return np.sum(temp3 / (-m)) + (sum1 + sum2) * lmbda / (2*m)
aa=nnCostFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lmbda)

def nnGrad(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    
    initial_theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1), 'F')
    initial_theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1), 'F')
    y_d = pd.get_dummies(y.flatten())
    delta1 = np.zeros(initial_theta1.shape)
    delta2 = np.zeros(initial_theta2.shape)
    m = len(y)
    
    for i in range(X.shape[0]):
        ones = np.ones(1)
        #input layer a1: (1 X 401) 
        a1 = np.hstack((ones, X[i]))
        #intial_theta: (25 X 401) z2: (1 X 25)
        z2 = a1.dot(initial_theta1.T)
        #hidden layer a2: (1 X 26)
        a2 = np.hstack((ones, sigmoid(z2)))
        #inital_theta2 : (10 , 26), z3 : (1 X 10)
        z3 = a2.dot(initial_theta2.T)
        a3 = sigmoid(z3)

        d3 = a3 - y_d.iloc[i,:][np.newaxis,:]
        print(d3.shape)
        z2 = np.hstack((ones, z2))
        print(z2.shape)
        print(d3.T.shape)
        print(initial_theta2.shape)
        print(sigmoidGrad(z2).T[:,np.newaxis].shape)
        d2 = np.multiply(initial_theta2.T @ d3.T, sigmoidGrad(z2).T[:,np.newaxis])        
        delta1 = delta1 + d2[1:,:].dot(a1[np.newaxis,:])
        delta2 = delta2 + d3.T.dot(a2[np.newaxis,:])
        
    delta1 /= m
    delta2 /= m
    
    delta1[:,1:] = delta1[:,1:] + initial_theta1[:,1:] * lmbda / m
    delta2[:,1:] = delta2[:,1:] + initial_theta2[:,1:] * lmbda / m
        
    return np.hstack((delta1.ravel(order='F'), delta2.ravel(order='F')))



nn_backprop_Params = nnGrad(nn_initial_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lmbda)

theta_opt = opt.fmin_cg(maxiter = 50, f = nnCostFunc, x0 = nn_initial_params, fprime = nnGrad, \
                        args = (input_layer_size, hidden_layer_size, num_labels, X, Y.flatten(), lmbda))

theta1_opt = np.reshape(theta_opt[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1), 'F')
theta2_opt = np.reshape(theta_opt[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1), 'F')

def predict(theta1, theta2, X, y):
    m = len(y)
    ones = np.ones((m,1))
    a1 = np.hstack((ones, X))
    a2 = sigmoid(a1.dot(theta1.T))
    a2 = np.hstack((ones, a2))
    h = sigmoid(a2.dot(theta2.T))
    return np.argmax(h, axis = 1) + 1

pY=predict(theta1_opt,theta2_opt,X,Y)
np.mean(pY == Y.flatten()) * 100



    