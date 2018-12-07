#Create by Bhavya Babuta
#Poly-Linear Regression
import numpy as numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 

X=[]
Y=[]

for line in open('data_poly.csv'):
	x,y=line.split(',')
	x=float(x)
	X.append([1,x,x*x])
	Y.append(float(y))

X=numpy.array(X)
Y=numpy.array(Y)

#calculate weights and finding the line of best fit
w=numpy.linalg.solve(numpy.dot(X.T,X),numpy.dot(X.T,Y))
# print(w)
Yhat=numpy.dot(X,w)
# Yhat1=numpy.dot(33.3137480056,w)

#print(Yhat1)
plt.scatter(X[:,1],Y)
plt.plot(sorted(X[:,1]),sorted(Yhat))
plt.show()

#Calculate R Squared
d1=Y-Yhat
d2=Y-Y.mean()
r2=1-d1.dot(d1)/d2.dot(d2)

print("The R Squared is " ,r2)