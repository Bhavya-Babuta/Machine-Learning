import numpy as np
import pandas as pd 



def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))


def forward_prog(X,Y,W,b):
	yhat= sigmoid(X.dot(W)+b)
	loss=-Y*np.log(yhat)-(1-Y)*np.log(1-yhat)
	cost = (np.sum(loss)) / X.shape[1]
	return yhat,loss,cost

def classification_rate(Y,P):
	return np.mean(Y == P)

# def cross_entropy(Y,predictions):
# 		E=0
# 		for i in range(Y.shape[0]):
# 			if Y[i]==1:
# 				E-=np.log(predictions[i])
# 			else:
# 				E-=np.log((1-predictions[i]))
# 		return E

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY) + (1 - T)*np.log(1 - pY))

def predict(W,b,X):
    z = sigmoid(np.dot(W.T,X) + b)           
    y_prediction = np.round(z)
    return y_prediction;    

X,Y=get_binary_data()
D=X.shape[1]
W=np.random.randn(D)
b=0
Y2=Y[:,1]
learning_rate=0.001
for i in range(10000):
	
	
	predictions=forward_prog(X,W,b)
	W-=(learning_rate*X.T.dot(predictions-Y2))/X.shape[1]
	b-=learning_rate*(predictions-Y2).sum()/X.shape[1]
	if i%1000==0:
		print("Cross Entropy Error", cross_entropy(Y2,predictions))



print("Cross Entropy Error", cross_entropy(Y2,predictions))

print("Score" ,classification_rate(Y,np.round(predictions)))

