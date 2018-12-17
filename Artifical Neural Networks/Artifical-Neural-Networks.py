#Create by Bhavya Babuta
#Self-Written & Self Taught Artifical Neural Network Code
#Start Date: 11/11/2018 End Date: 12/Nov/2018
#One Hidden Layer (Input Dimensions=12 , Output Dimensions = )

import numpy as np
import pandas as pd
class ANN():
	
	def __init__(self,x,y,learning_rate):
		self.X=x
		self.Y=y
		self.output=np.random.randn(Y.shape[0],1)
		self.weights1=np.random.randn(self.X.shape[1],1)
		self.weights2=np.random.rand(1,1)   
		self.learning_rate=learning_rate
		print(self.learning_rate)

	def forward_propogation(self):
		Y=self.Y.reshape([self.Y.shape[0],1])
		print("Y:  ",Y)
		print("Y Shape:  ",Y.shape)

		print("Weights1:   ",self.weights1)
		print(self.weights1.shape)

		print("Weights2:   ",self.weights2)
		print(self.weights2.shape)
		
		self.layer1=self.sigmoid(np.dot(self.X,self.weights1))
		print("Layer1 Output:   ",self.layer1)
		print("Layer1 Shape:     ",self.layer1.shape)
		
		self.output=self.sigmoid(np.dot(self.layer1,self.weights2.T))
		print("Output Layer output:   ",self.output)
		print("output Shape:     ",self.output.shape)
		print("Training Cost", self.cross_entropy(self.Y,self.output))
		
		#backward propogation using gradient descent
		self.weights1=self.weights1-((self.learning_rate)*(np.dot(self.X.T,(self.layer1-Y))))
		print("Weights1 Shape:   ",self.weights1.shape)

		self.weights2=self.weights2-((self.learning_rate)*(np.dot(self.layer1.T,(self.output-Y))))
		print("Weights2 Shape:   ",self.weights2.shape)
		print("X transpose shape" ,self.X.T.shape)
		print("Updating Weights")
   		

        
	def sigmoid(self,Z):
		Z = Z.astype(float)
		print("Z:   ",Z)
		print("Z Shape:   ",Z.shape)
		print("Z  Type:   ",Z.dtype)
		return 1/(1+np.exp(-Z))

	def classification_rate(self,Y,pY):
		return np.mean(Y==pY)
	
	def reLU(self,X):
		return np.maximum(0, X)

	def cross_entropy(self,Y,pY):
		print(Y.shape)
		print(pY.shape)
		print(Y.dtype)
		print(pY.dtype)
		E=0
		for i in range(Y.shape[0]):
			if Y[i]==1:
				E-=np.log(pY[i])
			else:
				E-=(1-Y[i])*np.log(1-pY[i])
		return E
	# def back_propogation(self):
		

X=pd.read_csv('Churn_Modelling.csv')
data=X.values
X=data[:,:-1]
Y=data[:,-1]
Y=np.array(Y)
X=X[:,3:13]

X2=np.zeros((X.shape[0],X.shape[1]+2))
X2[:,0]=X[:,0]
#OneHotEncoder , LabelEncoder
for i in range(X.shape[0]):
    if X[i,1]=='France':
        X[i,1]=0
    if X[i,1]=='Spain':
        X[i,1]=1
    if X[i,1]=='Germany':
        X[i,1]=2
    if X[i,2]=='Male':
        X[i,2]=0
    if X[i,2]=='Female':
        X[i,2]=1
    t=X[i,1]
    X2[i,t+1]=1
X2[:,4:12]=X[:,2:10]
print(X2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X2 = sc.fit_transform(X2)

print(Y.shape)
model=ANN(X2,Y,0.001)
for i in range(10):
	model.forward_propogation()
	# model.back_propogation()
print("Score",model.classification_rate(Y,np.round(model.output)))
print(np.round(model.output))
