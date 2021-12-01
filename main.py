from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pandas as pd
import nltk
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pickle
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

app = Flask('__name__',template_folder='templates')

porter = PorterStemmer()
from nltk.tokenize import sent_tokenize, word_tokenize
def lower_text(text):
    return text.lower()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_numbers(text):
    text_cleaned = re.sub('[^a-zA-Z]',' ',text)
    return text_cleaned

def remove_stopwords(text):
    stoplist = stopwords.words('english')
    sps = stoplist
    return " ".join([word for word in str(text).split() if word not in sps])


@app.route("/",methods=['GET','POST'])
def predict_():
    if request.method == 'GET':
        return render_template("predict.html",predicted=False)
    title_text = request.form['titleInputForm']
    news_text = request.form['newsTextInputForm']
    author_text = request.form['authorInputForm']

    if (len(title_text) == 0 and len(author_text) == 0):
        print("Using text")
        # only use text
        news_text = remove_stopwords(news_text)
        news_text = remove_numbers(news_text)
        news_text = lemmatization(news_text)
        train_data = pd.read_csv("filtered_df.csv")
        train_text = list(train_data['text'])
        y_train = list(train_data['label'])

        X = train_text
        X.append(news_text)

        tf_vectorizer = CountVectorizer()
        X_vectorised = tf_vectorizer.fit_transform(X)

        X_train = X_vectorised[:-1]
        X_test = X_vectorised[-1]

        clf = PassiveAggressiveClassifier()
        clf.fit(X_train, y_train)
        y_pred_value = clf.predict(X_test)[0]

    elif (len(author_text) == 0):
        print("Using title and text")
        # use text and title 
        news_text = remove_stopwords(news_text)
        title_text = remove_stopwords(title_text)

        news_text = remove_numbers(news_text)
        title_text = remove_numbers(title_text)

        news_text = lemmatization(news_text)
        title_text = lemmatization(title_text)

        train_data = pd.read_csv("filtered_df.csv")
        train_text = list(train_data['titleplustext'])
        y_train = list(train_data['label'])

        X = train_text
        X.append(str(news_text + " "+ title_text))

        tf_vectorizer = CountVectorizer()
        X_vectorised = tf_vectorizer.fit_transform(X)

        X_train = X_vectorised[:-1]
        X_test = X_vectorised[-1]

        clf = PassiveAggressiveClassifier()
        clf.fit(X_train, y_train)
        y_pred_value = clf.predict(X_test)[0]

    elif ( len(title_text) == 0):
        print("Using author and text")
        # use authorname and text
        news_text = remove_stopwords(news_text)
        author_text = remove_stopwords(author_text)

        news_text = remove_numbers(news_text)
        author_text = remove_numbers(author_text)

        news_text = lemmatization(news_text)
        author_text = lemmatization(author_text)

        train_data = pd.read_csv("filtered_df.csv")
        train_text = list(train_data['authorplustext'])
        y_train = list(train_data['label'])

        X = train_text
        X.append(str(news_text+" " + author_text))

        tf_vectorizer = CountVectorizer()
        X_vectorised = tf_vectorizer.fit_transform(X)

        X_train = X_vectorised[:-1]
        X_test = X_vectorised[-1]

        clf = PassiveAggressiveClassifier()
        clf.fit(X_train, y_train)
        y_pred_value = clf.predict(X_test)[0]
    else:
        # all non-None
        print("ALL non None")
        news_text = remove_stopwords(news_text)
        title_text = remove_stopwords(title_text)
        author_text = remove_stopwords(author_text)

        news_text = remove_numbers(news_text)
        title_text = remove_numbers(title_text)
        author_text = remove_numbers(author_text)

        news_text = lemmatization(news_text)
        title_text = lemmatization(title_text)
        author_text = lemmatization(author_text)

        train_data = pd.read_csv("filtered_df.csv")
        train_text = list(train_data['titleauthortext'])
        y_train = list(train_data['label'])

        X = train_text
        X.append(str(news_text+" "+title_text+" "+ author_text))

        tf_vectorizer = CountVectorizer()
        X_vectorised = tf_vectorizer.fit_transform(X)

        X_train = X_vectorised[:-1]
        X_test = X_vectorised[-1]

        clf = PassiveAggressiveClassifier()
        clf.fit(X_train, y_train)
        y_pred_value = clf.predict(X_test)[0]

    if y_pred_value == 1:
        answer_results = ["Fake News"]
    else:
        answer_results = ["True News"]


    return render_template("predict.html", results_computed=True, result = answer_results)

if __name__ == '__main__':
    app.run(debug = True)
