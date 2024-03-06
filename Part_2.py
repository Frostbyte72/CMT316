import os
import operator
import sys

import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , precision_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2 , SelectKBest

#NLTK initalisation
lemmatizer = nltk.stem.WordNetLemmatizer()
#nltk.download('punkt') # Needed for tokenisation
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.add(".")
stopwords.add(",")
stopwords.add("-")
stopwords.add("''")
stopwords.add("[]")
stopwords.add("()")
stopwords.add("``")
stopwords.add(":")
stopwords.add("'s")

dataset = pd.DataFrame(columns = ['Category','Cat_id','Text'])
categories = ['entertainment', 'business', 'sport', 'politics', 'tech']

idx = 0
id = 0
word_freq = {}
for cat in categories:
    for file in os.listdir('datasets_coursework1/bbc/{}'.format(cat)):
        text = open('datasets_coursework1/bbc/{c}/{f}'.format(c = cat, f = file), "r").read()
        
        # Process the text into frequency vector
        # ====================================
        # Tokenisation of the text
        text_token_list = nltk.tokenize.word_tokenize(text)
        
        # Lemmatize the tokens
        for token in text_token_list:
            text = text + token
            lemmatizsed_text = lemmatizer.lemmatize(token).lower()
            #filter out stop words and add the word frequencies to word_freq dict
            if lemmatizsed_text in stopwords:
                continue
            elif lemmatizsed_text in word_freq:
                word_freq[lemmatizsed_text] += 1
            else:
                word_freq[lemmatizsed_text] = 1

        dataset.loc[idx] = [cat,id,text_token_list]
        idx += 1

    wc = WordCloud().generate_from_frequencies(word_freq)
    wc.to_file('{}_wordcloud.png'.format(cat))
    id += 1

most_common = sorted(word_freq.items(),key=operator.itemgetter(1),reverse = True )[:1000]
vocab = []
for word,count in most_common:
    vocab.append(word)

# Generates word frequency vectors
def create_freq_vec(vocab,text):
    freq_vec = np.zeros(len(vocab))
    for i , word in enumerate(vocab):
        if word in text:
            freq_vec[i] = text.count(word)
        else:
            freq_vec[i] = 0

    return freq_vec

#Calculate Features
print('Calculating features')
dataset['BagOfWords'] = dataset['Text'].apply(lambda x: list(create_freq_vec(vocab,x)))

def dummy(token):
    return token

v = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy)
x = v.fit_transform(dataset['Text'])

TFIDF_vectorizer = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy,max_features=2000,min_df=0.02,max_df=0.4)
TFIDF = TFIDF_vectorizer.fit_transform(dataset['Text'].values)
TFIDF  = TFIDF.toarray()

N_gram_vectorizer  = CountVectorizer(tokenizer=dummy, preprocessor=dummy,max_features=2000,ngram_range=(2, 3),min_df=0.02,max_df=0.4)
N_gram = N_gram_vectorizer.fit_transform(dataset['Text'].values)
N_gram = N_gram.toarray()

#generate X and y for the ML model
X = np.hstack((dataset['BagOfWords'].iloc[0],TFIDF[0],N_gram[0]))

for i,item in enumerate(TFIDF):
    row = np.hstack((dataset['BagOfWords'].iloc[i],item,N_gram[i]))
    X = np.vstack((X,row))

X = X[1:]
y = dataset['Cat_id']


print('Creating input matrix')
def feature_selection(X,y,k = 2000):

    # Score function Chi2 tells the feature to be selected using Chi Square
    test = SelectKBest(score_func=chi2, k=k)
    fit = test.fit(X, y)

    X_new=test.fit_transform(X, y)
    
    return X_new


#Setup parameters for random classificer
model = RandomForestClassifier(n_estimators=500)

if '-f' in sys.argv:
    
    print('testing features development set!')
    

    features = [50,250,500,1000,2000,3000,4000]
    stats = {}
    for param in features:
        
        #change parameters fand retrain the model
        X_new = feature_selection(X,y,k=param)
        # Split the data into test and train
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2)
        #Split the test set into test and development
        X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size = 0.5)

        model.fit(X_train,y_train)

        y_pred_dev = model.predict(X_dev) 
        accuracy = accuracy_score(y_dev,y_pred_dev)
        print('accuracy on dev set with number of features = {p} :{a}'.format(p=param,a=accuracy))
        stats[param] = accuracy
    
    print('Best performaning paramter {p} with accuracy: {a}'.format(p = max(stats,key=stats.get),a = stats[max(stats,key=stats.get)]) )
    # Change number of features to the best performing one.
    X = feature_selection(X,y,k=max(stats,key=stats.get))

if '-t' in sys.argv:
    print('testing features development set!')
    # Split the data into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    #Split the test set into test and development
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size = 0.5)

    trees = [10,50,250,500,1000]
    stats = {}
    for param in trees:
        model = RandomForestClassifier(n_estimators=param , random_state=0)

        model.fit(X_train,y_train)

        y_pred_dev = model.predict(X_dev) 
        accuracy = accuracy_score(y_dev,y_pred_dev)
        print('accuracy on dev set with number of trees = {p} :{a}'.format(p=param,a=accuracy))
        stats[param] = accuracy

    print('Best performaning paramter {p} with accuracy: {a}'.format(p = max(stats,key=stats.get),a = stats[max(stats,key=stats.get)]) )
    # Set best performing number of trees as the one used in the model.
    model = RandomForestClassifier(n_estimators= max(stats,key=stats.get) , random_state=0)

if '-t' not in sys.argv and '-f' not in sys.argv:
    X = feature_selection(X,y)

    # Split the data into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Train the model
print("Training Final Model")
model.fit(X_train,y_train)

#Training Accuracy
model.score(X_train, y_train)
#generate predictions on test set
y_pred_test = model.predict(X_test)
# Train predictons
y_pred_train = model.predict(X_train) 

print('Model Performance on Train Dataset: ')
print(classification_report(y_train, y_train,target_names=['entertainment', 'business','sport','politics','tech']))
print('Model Performance on Test Dataset: ')
print(classification_report(y_test, y_pred_test,target_names=['entertainment', 'business','sport','politics','tech']))