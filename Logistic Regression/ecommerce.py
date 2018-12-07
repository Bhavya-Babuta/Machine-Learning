import numpy as np
import pandas as pd 

b=0
def get_data():
	df=pd.read_csv('ecommerce_data.csv') 
	data = df.values
	np.random.shuffle(data)
	X = data[:,:-1]
	Y = data[:,-1].astype(np.int32)
	N, D = X.shape
	X2 = np.zeros((N, D+3))
	X2[:,0:(D-1)] = X[:,0:(D-1)] # non-categorical
	for n in range(N):
		t = int(X[n,D-1])
		X2[n,t+D-1] = 1

	X2[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()
	X2[:,2]=(X[:,2]-X[:,2].mean())/X[:,1].std()
	return X2,Y


def get_binary_data():
  # return only the data from the first 2 classes
   X,Y = get_data()
   X2 = X[Y <= 1]
   Y2 = X[Y <= 1]
   return X2,Y2

def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))


def forward_prog(X,W,b):

	return sigmoid(X.dot(W)+b)

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


X,Y=get_binary_data()
D=X.shape[1]
W=np.random.randn(D)
b=0
Y2=Y[:,1]
learning_rate=0.001
for i in range(10000):
	
	
	predictions=forward_prog(X,W,b)
	W-=learning_rate*X.T.dot(predictions-Y2)
	b-=learning_rate*(predictions-Y2).sum()
	if i%1000==0:
		print("Cross Entropy Error", cross_entropy(Y2,predictions))



print("Cross Entropy Error", cross_entropy(Y2,predictions))

print("Score" ,classification_rate(Y,np.round(predictions)))

