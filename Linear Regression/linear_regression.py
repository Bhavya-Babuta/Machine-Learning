#Created by Bhavya Babuta
#ONE DIMENSIONAL LINEAR REGRESSION

import numpy as numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 

X=[]
Y=[]
for line in open('data_1d.csv'):
	x,y=line.split(',')
	X.append(float(x))
	Y.append(float(y))
X=numpy.array(X)
Y=numpy.array(Y)

denominator=X.dot(X)-X.mean()*X.sum()

a=(X.dot(Y)-Y.mean()*X.sum())/denominator
b=(Y.mean()*X.dot(X)-X.mean()*X.dot(Y))/denominator

#Calculate the value of the predicated Y
Yhat=a*X+b

plt.scatter(X,Y)
plt.plot(X,Yhat)
plt.show()	

#Calculate the R _ Squared
d1=Y-Yhat
d2=Y-Y.mean()
r=1-d1.dot(d1)/d2.dot(d2)

print ("The R Sqaured is  ",r)