#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:14:48 2018

@author: bhavyababuta
"""

import pandas as pd
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

data=pd.read_csv('/Users/bhavyababuta/Desktop/ML and Data Visualisations/House_Sales_King County_USA/kc_house_data.csv')

Y_data=data['price']

data.head(10)
data.info()
data.describe(include='all')

numerical_feats = data.dtypes[data.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = data.dtypes[data.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))

data[['bedrooms','price']].groupby(['bedrooms']).mean().sort_values(by='price',ascending=False).plot.bar()

data[['bathrooms','price']].groupby(['bathrooms']).mean().sort_values(by='price',ascending=False).plot.bar()

data[['floors','price']].groupby(['floors']).mean().sort_values(by='price',ascending=False).plot.bar()

data[['waterfront','price']].groupby(['waterfront']).mean().sort_values(by='price',ascending=False).plot.bar()

data[['grade','price']].groupby(['grade']).mean().sort_values(by='price',ascending=False).plot.bar()

data[['condition','price']].groupby(['condition']).mean().sort_values(by='price',ascending=False).plot.bar()

data[['view','price']].groupby(['view']).mean().sort_values(by='price',ascending=False).plot.bar()

data[['bedrooms','sqft_lot']].groupby(['bedrooms']).mean().sort_values(by='sqft_lot',ascending=False).plot.bar()

data[data['bedrooms']==0]
data['bedrooms'].value_counts()

sns.distplot(data['sqft_living'])
sns.distplot(data['sqft_lot'])
sns.distplot(data['sqft_above'])
sns.distplot(data['sqft_basement'])

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 

plt.plot(list(data['sqft_living']),list(data['price']))
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.show()

plt.scatter(list(data['sqft_living']),list(data['price']))
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.show()

plt.plot(list(data['sqft_lot']),list(data['price']))
plt.xlabel('sqft_lot')
plt.ylabel('price')
plt.show()

plt.scatter(list(data['sqft_lot']),list(data['price']))
plt.xlabel('sqft_lot')
plt.ylabel('price')
plt.show()

plt.plot(list(data['sqft_above']),list(data['price']))
plt.xlabel('sqft_above')
plt.ylabel('price')
plt.show()

plt.scatter(list(data['sqft_above']),list(data['price']))
plt.xlabel('sqft_above')
plt.ylabel('price')
plt.show()

plt.plot(list(data['sqft_basement']),list(data['price']))
plt.xlabel('sqft_basement')
plt.ylabel('price')
plt.show()

plt.scatter(list(data['sqft_basement']),list(data['price']))
plt.xlabel('sqft_basement')
plt.ylabel('price')
plt.show()

plt.plot(list(data['yr_built']),list(data['price']))
plt.xlabel('yr_built')
plt.ylabel('price')
plt.show()

plt.scatter(list(data['yr_built']),list(data['price']))
plt.xlabel('yr_built')
plt.ylabel('price')
plt.show()

data['yr_built'].value_counts()
for i in range(len(list(data['yr_built']))):
    if int(data['yr_built'][i]) >=2000:
        data['yr_built'][i]=3
    elif int(data['yr_built'][i]) <= 1999 and int(data['yr_built'][i]) >=1950:
        data['yr_built'][i]=2
    else:
        data['yr_built'][i]=1
        
data[['yr_built','price']].groupby('yr_built').mean().sort_values(by='price',ascending=False).plot.bar()
data['yr_renovated'].value_counts()

data[['yr_renovated','price']].groupby('yr_renovated').mean().sort_values(by='price',ascending=False).plot.bar()

for i in range(len(list(data['yr_renovated']))):
    if int(data['yr_renovated'][i]) >=2000:
        data['yr_renovated'][i]=3
    elif int(data['yr_renovated'][i]) <= 1999 and int(data['yr_renovated'][i]) >=1950:
        data['yr_renovated'][i]=2
    else:
        data['yr_renovated'][i]=1 

def correlation_heatmap(df1):
    _, ax = plt.subplots(figsize = (15, 10))
    colormap= sns.diverging_palette(220, 10, as_cmap = True)
    sns.heatmap(df1.corr(), annot=True, cmap = colormap)
correlation_heatmap(data)
data=data[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated']]
data.columns

sns.pairplot(data,x_vars=data.columns,y_vars=Y_data,height=7,aspect=0.7,kind = 'reg')

data.describe(include='all')


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(data, Y_data, test_size = 0.1,random_state=0)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifiers=LogisticRegression(C=0.1)
classifiers.fit(data,Y_data)
predicted_=classifiers.predict(data)
mean_squared_error = mean_squared_error(Y_data, predicted_)
print('Intercept: ', classifiers.intercept_)
print('Coefficient:', classifiers.coef_)
print('Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2))
print('R-squared (training) ', round(classifiers.score(data,Y_data), 3))

#Linear_Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(data, Y_data)
predicted_=regressor.predict(data)
mean_squared_error = mean_squared_error(Y_data, predicted_)
print('Intercept: ', regressor.intercept_)
print('Coefficient:', regressor.coef_)
print('Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2))
print('R-squared (training) ', round(regressor.score(data,Y_data), 3))

_, ax = plt.subplots(figsize= (10, 12))
plt.scatter(data, Y_data, color= 'darkgreen', label = 'data')
plt.plot(data, reg.predict(x_test), color='red', label= ' Predicted Regression line')
plt.xlabel('Living Space (sqft)')
plt.ylabel('price')
plt.legend()
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


data.insert(loc=0,column='Ones',value=np.ones((data.shape[0])))

#Linear_Regression without Any Library Import
w=np.linalg.solve(np.dot(data.T,data),np.dot(data.T,Y_data))
Yhat=np.dot(data,w)
d1=Y_data-Yhat
d2=Y_data-Y_data.mean()
r2=1-d1.dot(d1)/d2.dot(d2)

print("The R Squared is" ,r2)