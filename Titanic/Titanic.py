#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:25:50 2018

@author: bhavyababuta
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split


train=pd.read_csv('/Users/bhavyababuta/Desktop/Titanic-Kaggle/train.csv')
test=pd.read_csv('/Users/bhavyababuta/Desktop/Titanic-Kaggle/test.csv')

print(train.shape)
print(test.shape)


print(train.columns.values)

train.info()

test.info()

train.describe(include='all')

#Histogram to find a relationship between the survival rate in each of the PassengerClasses since this feature has no missing value and is already a numerical value
train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False).plot.bar()

#Histogram to find a relationship between the survival rate with respect to the Sibling or Spouse count since this feature has no missing value and is already a numerical value
train[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False).plot.bar()

#Histogram to find a relationship between the survival rate with respect to the Parent and Child count since this feature has no missing value and is already a numerical value
train[['Parch', 'Survived']].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False).plot.bar()


familySize=[]

for index,row in train.iterrows():
    familySize.append(row['SibSp']+row['Parch']+1)

train['Family Size']=familySize
train['isAlone']=''


for i in range(len(familySize)):
    if familySize[i] ==1:
        train['isAlone'][i]=1
    else:
        train['isAlone'][i]=0

train[['Family Size','Survived']].groupby(['Family Size']).mean().sort_values(by='Survived',ascending=False).plot.bar()

train[['isAlone','Survived']].groupby(['isAlone']).mean().sort_values(by='Survived',ascending=False).plot.bar()


def LabelEncoder(list):
    labels=[]
    for i in range(len(list)):
        if list[i] not in labels:
            labels.append(list[i])
    print(labels)
    for i in range(len(list)):
        list[i]=labels.index(str(list[i]))
    return list

def OneHotEncoder(list):
    labels=[]
    for i in range(len(list)):
        if list[i] not in labels:
            labels.append(list[i])
    print(labels)
    for i in range(len(list)):
        list[i]=labels.index(str(list[i]))
    
    one_hot_encoding=np.zeros((len(list),len(labels)))
    for i in range(len(list)):
        one_hot_encoding[i][list[i]]=1
    return one_hot_encoding

train['Sex'] = pd.DataFrame(LabelEncoder(list(train['Sex'])))
train.groupby(['Survived','Sex'])['Survived'].count()



train.Age.isnull().sum()
train['Age']=train.Age.fillna(train.Age.mean())


sns.distplot(train['Age'])
train.Fare.isnull().sum()


sns.distplot(train['Fare'])
train['Embarked'] = train['Embarked'].fillna(max(train['Embarked'].value_counts().keys())) 

drop_columns=['Cabin','Name','Ticket','PassengerId','SibSp','Parch']
train.drop(drop_columns,axis=1,inplace=True)

train[[ 'EmbarkedS', 'EmbarkedC','EmbarkedQ']]=pd.DataFrame(OneHotEncoder(list(train['Embarked'])))

train[[ 'EmbarkedS', 'EmbarkedC','EmbarkedQ','Survived']].groupby( ['EmbarkedS', 'EmbarkedC','EmbarkedQ']).mean().sort_values(by='Survived',ascending=False).plot.bar()

train.drop('Embarked',axis=1,inplace=True)

y = train["Survived"]

X= train.drop("Survived", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,random_state=0)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

classifiers=LogisticRegression(C=0.1)

classifiers.fit(X_train, y_train)
train_predictions = classifiers.predict(X_test)

def classification_rate(pY,Y_test):
    pY=np.array(pY)
    Y_test=np.array(Y_test)
    return np.mean(pY==Y_test)

print(classification_rate(train_predictions,y_test))#0.8
acc = accuracy_score(y_test, train_predictions)

from sklearn.naive_bayes import GaussianNB
classifiers=GaussianNB()
classifiers.fit(X_train, y_train)
train_predictions = classifiers.predict(X_test)
print(classification_rate(train_predictions,y_test)) #0.8333333333333334

from sklearn.tree import DecisionTreeClassifier
classifiers=DecisionTreeClassifier()
classifiers.fit(X_train, y_train)
train_predictions = classifiers.predict(X_test)
print(classification_rate(train_predictions,y_test))#0.7666666666666667

from sklearn.svm import SVC

classifiers=SVC(probability=True)
classifiers.fit(X_train, y_train)
train_predictions = classifiers.predict(X_test)
print(classification_rate(train_predictions,y_test))#0.6555555555555556


test.info()
test=test.drop("Cabin",axis=1)
test.head()
test.describe(include='all')

test['Sex'] = pd.DataFrame(LabelEncoder(list(test['Sex'])))

familySize=[]

for index,row in test.iterrows():
    familySize.append(row['SibSp']+row['Parch']+1)

test['Family Size']=familySize
test['isAlone']=''


for i in range(len(familySize)):
    if familySize[i] ==1:
        test['isAlone'][i]=1
    else:
        test['isAlone'][i]=0
test=test.drop(['Ticket','SibSp','Parch'],axis=1)
test[[ 'EmbarkedS', 'EmbarkedC','EmbarkedQ']]=pd.DataFrame(OneHotEncoder(list(test['Embarked'])))
test=test.drop(['Embarked','Name'],axis=1)
test=test.drop('PassengerId',axis=1)

test['Age']=train.Age.fillna(test.Age.mean())

test['Fare']=train.Age.fillna(0)
gender_submission=pd.read_csv('/Users/bhavyababuta/Desktop/Titanic-Kaggle/gender_submission.csv')
gender_submission=gender_submission['Survived']

classifiers=LogisticRegression(C=0.1)

train_predictions = classifiers.predict(test)
print(classification_rate(train_predictions,gender_submission))#0.9832535885167464
acc = accuracy_score(gender_submission, train_predictions)

classifiers=GaussianNB()

train_predictions = classifiers.predict(test)
print(classification_rate(train_predictions,gender_submission))#0.9832535885167464
acc = accuracy_score(gender_submission, train_predictions)

