import numpy as np 
import pandas as pd

def get_data():
	df=pd.read_csv('ecommerce_data.csv')
	data=df.values
	np.random.shuffle(data)
	
	X=data[:,:-1]
	
	Y=data[:,-1]
	N,D=X.shape
	X2=np.zeros((N,D+3))
	X2[:,0:D-1]=X[:,0:D-1]
	for i in range(N):
		t=int(X[i,D-1])
		X2[i,t+D-1]=1
	X2[:,1]=X2[:,1]-X2[:,1].mean()/X2[:,1].std()
	X2[:,2]=X2[:,2]-X2[:,2].mean()/X2[:,2].std()
	return X2,Y

def get_binary_data():
	X,Y=get_data()
	X2=X[Y <= 1]
	Y2=Y[Y <= 1]
	return X2,Y2

def sigmoid(Z):
	# print("Z   :",Z)
	# print("Z Data Type:    ",Z.dtype)
	return 1/(1+np.exp(-Z))

def forward_propogation(X,W,b):
	return sigmoid(X.dot(W)+b)

def classification_rate(Y,pY):
	return (np.mean(Y==pY))

def cross_entropy(T, pY):
	print(T.shape)
	print(pY.shape)
	print(T.dtype)
	print(pY.dtype)
	E=0
	for i in range(T.shape[0]):
		if T[i]==1:
			E-=np.log(pY[i])
		else:
			E-=(1-T[i])*np.log(1-pY[i])
	return E

X,Y=get_binary_data()
print("Independent Variable Shape  :",X.shape)
print("Independent Variable     :",X)
print("Dependent Variable :   ",Y)
print("Dependent Variable     :",Y.shape)
print()
N,D=X.shape
b=0
W=np.random.randn(D)
learning_rate=0.01
for i in range(10000):
	# print("Weights   :",W)
	# print("Weights Shape:   ",W.shape)
	# print("Bias      :",b)
	P_Y_given_X=forward_propogation(X,W,b)
	predictions=np.round(P_Y_given_X)
	training_cost=cross_entropy(Y,predictions)
	if i%100==0:
		print("Cost Of Training  :     ",training_cost)
	print("Updating Weights")
	W-=(learning_rate*X.T.dot(P_Y_given_X-Y))
	print("Updating Bias")
	b-=learning_rate*(P_Y_given_X-Y).sum()

print("Score: ",classification_rate(Y,predictions))
#print("Final Predicitions",np.round(P_Y_given_X))
print("Predicting the entire Independent Variable we get   :",np.round(forward_propogation(X,W,0)))
print(Y)
print("Predictions",np.round(forward_propogation(np.array([0,0,0.0058375210477,0,0,0,0,1]),W,0)))