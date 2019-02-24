#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 00:54:30 2019

@author: bhavyababuta
"""

import pandas as pd
import numpy as np
import re
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud,STOPWORDS

import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
stopWordList=stopwords.words('english')


data=pd.read_csv('Tweets.csv')
data.head(20)

data.isnull().sum()
data.describe(include='all')

for columns in data.columns:
    print('%s'%(columns))
    data[columns].value_counts()
    
data['airline'].value_counts()
data['retweet_count'].value_counts()


for airline in data['airline'].value_counts().keys():
    Index = [1,2,3]
    plt.figure()
    plt.bar(Index,data[data['airline']==airline]['airline_sentiment'].value_counts())
    plt.xticks(Index,['negative','neutral','positive'],rotation=45)
    plt.ylabel('Sentiment Count')
    plt.xlabel('Sentiment')
    plt.title('Sentiment Count %s' %airline)
    
data['negativereason'].value_counts()

plt.figure()
plt.bar(data['negativereason'].value_counts().keys(),data['negativereason'].value_counts())
plt.xticks(rotation=90)
plt.ylabel('Negative Count')
plt.xlabel('Negative Reason')
plt.title('Negative Reason Count' )

for airline in data['airline'].value_counts().keys():
    plt.figure()
    plt.bar(data[data['airline']==airline]['negativereason'].value_counts().keys(),data[data['airline']==airline]['negativereason'].value_counts())
    plt.xticks(rotation=90)
    plt.ylabel('Negative Count')
    plt.xlabel('Negative Reason')
    plt.title('Negative Reason Count %s' %airline )
    
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt  

def removeStopWords(list):
	filtered_list=[]
	for line in list:
		filtered_sentence=[]
		word_tokens = word_tokenize(line)
		for w in word_tokens:
			if  w not in stopWordList:
				filtered_sentence.append(w)
		text=' '.join(str(x) for x in filtered_sentence)
		filtered_list.append(text)
	return filtered_list

def hashtag_extract(x):
    hashtags = []
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

def removeCharDigit(list):
	s=[]
	str='`1234567890-=~@#$%^&*()_+[!{;”:\’><.,/?”}]'
	for line in list:
 		for w in line:
 			if w in str:
 				line=line.replace(w,'')
 		s.append(line)
	return s

def removeHttpLinks(list):
    a=[]
    for line in list:
        line = re.sub(r"http\S+", "", line)
        a.append(line)
    return a

def removeShortWords(list):
    b=[]
    for line in list:
        a=[]
        for word in word_tokenize(line):
            if len(word)>=3:
                a.append(word)
        a=' '.join(str(x) for x in a)
        b.append(a)
    return b

def get_words(list):
    a=[]
    for words in list:
        for word in words:
            word=word.lower()
            a.append(word)
    return a

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def get_senti_tweets(data):
    tweets=[]
    for index,row in data.iterrows():
        tweets.append((row.tokenized_tweet,row.airline_sentiment))
    return tweets
    

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=1000,
                      height=1000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
data['TweetText']=np.vectorize(remove_pattern)(data['text'], "@[\w]*")
data['hash_tags']=hashtag_extract(data['TweetText'])
data['TweetText']=np.vectorize(remove_pattern)(data['TweetText'], "#[\w]*")
data['TweetText']=removeHttpLinks(data['TweetText'])
data['TweetText']=np.vectorize(remove_pattern)(data['TweetText'], "RT")
data['TweetText']=removeStopWords(data['TweetText'])
data['TweetText']=removeCharDigit(data['TweetText'])
data['TweetText']=np.vectorize(remove_pattern)(data['TweetText'], "'[\w]*")
data['TweetText']=removeShortWords(data['TweetText'])
data['TweetText']=np.vectorize(remove_pattern)(data['TweetText'], "amp")
tokenized_tweet =data['TweetText'].apply(lambda x: x.split())

total_words=get_words(tokenized_tweet)

wordlist = nltk.FreqDist(total_words)
wordcloud_draw(wordlist.keys())
w_features=wordlist.keys()

data['tokenized_tweet']=tokenized_tweet

tweets=[]
tweets=get_senti_tweets(data)
a=extract_features(data['TweetText'])

data=data[data['airline_sentiment']!='neutral']

pos_sentiment=data[data['airline_sentiment']=='positive']
pos_words=' '.join(pos_sentiment['TweetText'])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(pos_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

neg_sentiment=data[data['airline_sentiment']=='negative']
neg_words=' '.join(neg_sentiment['TweetText'])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(neg_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

from sklearn.model_selection import train_test_split
train, test = train_test_split(data,test_size = 0.1)
train_pos = train[ train['airline_sentiment'] == 'Positive']
train_pos = train_pos['text']
train_neg = train[ train['airline_sentiment'] == 'Negative']
train_neg = train_neg['text']

test_pos = test[ test['airline_sentiment'] == 'Positive']
test_pos = test_pos['text']
test_neg = test[ test['airline_sentiment'] == 'Negative']
test_neg = test_neg['text']

training_set = nltk.classify.apply_features(extract_features,tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

neg_cnt = 0
pos_cnt = 0
res_list=[]
for obj in test_neg: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Negative'): 
        neg_cnt = neg_cnt + 1
        res_list.append((obj,res,'Neg'))
for obj in test_pos: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Positive'): 
        pos_cnt = pos_cnt + 1
        res_list.append((obj,res,'Pos'))

test_res=[]
for obj in test['text']:
        res =  classifier.classify(extract_features(obj.split()))
        test_res.append((obj,res))
count=0
i=0
for index,row in test.iterrows():
   if row['text']==test_res[i][0]:
       if row['airline_sentiment']==test_res[i][1]:
           count+=1
   i+=1
print('Accuracy',(count/len(test))*100)
res =  classifier.classify(extract_features("What a nice flight. Aweomse. Awesome hospitality".split()))


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['TweetText'].values)
X = tokenizer.texts_to_sequences(data['TweetText'].values)
X = pad_sequences(X)
X=pd.DataFrame(X)
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

print((data['airline_sentiment']).values)
Y = pd.get_dummies(data['airline_sentiment']).values
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)


score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.5f" % (score))
print("acc: %.5f" % (acc))

twt = ['What a nice flight. Aweomse. Awesome hospitality']
#vectorizing the tweet by the pre-fitted tokenizer instance
twt = tokenizer.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, maxlen=22, dtype='int32', value=0)
print(twt)
sentiment = model.predict(twt,batch_size=1,verbose = 7)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")
    
train_clean_tweet=[]
for tweet in train['TweetText']:
    train_clean_tweet.append(tweet)
test_clean_tweet=[]
for tweet in test['TweetText']:
    test_clean_tweet.append(tweet)
        
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer(analyzer = "word")
train_features= v.fit_transform(train_clean_tweet)
test_features=v.transform(test_clean_tweet)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
Classifiers = [
    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True,gamma="scale"),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier(),
    GaussianNB()]

dense_features=train_features.toarray()
dense_test= test_features.toarray()
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['airline_sentiment'])
        pred = fit.predict(test_features)
    except Exception:
        fit = classifier.fit(dense_features,train['airline_sentiment'])
        pred = fit.predict(dense_test)
    accuracy = accuracy_score(pred,test['airline_sentiment'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy)) 

plt.figure()
plt.bar(Model,Accuracy)
plt.xticks(rotation=90)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.title('Accuracy Bar')  


