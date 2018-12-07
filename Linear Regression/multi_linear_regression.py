##Create by Bhavya Babuta
#Mutli-Dimensional Linear Regression

import numpy as numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

X=[]
Y=[]
for line in open('data_2d.csv'):
	x,x2,y=line.split(',')
	X.append([1,float(x),float(x2)])
	Y.append(float(y))

X=numpy.array(X)
Y=numpy.array(Y)

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],Y)
plt.show()

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,2],Y)
plt.show()

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X[:,1],X[:,2],Y)
plt.show()

#calculate weights
w=numpy.linalg.solve(numpy.dot(X.T,X),numpy.dot(X.T,Y))
print(w)

Yhat=numpy.dot(X,w)
test=[1,21.882803904,46.8415051959]

test=numpy.array(test)
print(numpy.dot(test,w))

##Calculate R Squared 
d1=Y-Yhat
d2=Y-Y.mean()
r2=1-d1.dot(d1)/d2.dot(d2)

print("The R Squared is" ,r2)