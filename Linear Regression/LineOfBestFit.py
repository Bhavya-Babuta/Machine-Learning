#Create by Bhavya Babuta
#Code for finding the line of best fit from our dataset
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

X1=numpy.copy(X)
Y1=numpy.copy(Y)

# Mean of X
Xm=X.mean()
#Mean of Y
Ym=Y.mean()
#Subtracting the mean of X from vector X
for i in range(X.size):
	X[i]-=Xm
#Subtracting the mean of Y from vector Y	
for i in range(Y.size):
	Y[i]-=Ym
# num = new values of X and Y from the above operation
num=X.dot(Y)
den=X.dot(X)

slope=num/den
yin=Ym-slope*Xm

yhat=slope*X1+yin

#Calculate the R Squared
d1=Y1-yhat
d2=Y1-Y1.mean()
r=1-d1.dot(d1)/d2.dot(d2)

print ("The R Squared is  ",r)
plt.scatter(X1,Y1)
plt.plot(X1,yhat)
plt.show()