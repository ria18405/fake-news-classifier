from flask import Flask, render_template, request, send_from_directory
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import nltk
import re
import nltk
# import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
# import torch
# from textblob import TextBlob, Word, Blobber
# import plotly.express as px
import pickle
# from tqdm.notebook import tqdm, trange
# from autocorrect import Speller
# from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask('__name__',template_folder='templates')

# from tqdm.notebook import tqdm, trange
# from nltk.stem import PorterStemmer
# from autocorrect import Speller
from autocorrect import Speller

spell = Speller(lang='en')

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

def autospell(text):
    return " ".join([spell(word) for word in text.split()])
 


# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )


@app.route("/",methods=['GET','POST'])
def predict_():
    if request.method == 'GET':
        return render_template("predict.html",predicted=False)
    title_text = request.form['titleInputForm']
    news_text = request.form['newsTextInputForm']
    author_text = request.form['authorInputForm']
    # print(title_text)
    # print(author_text)

    trained_model = pickle.load(open('models/onehotmodel2.sav', 'rb'))

    # preprocess test sentence:
    # test_df.isnull().sum().sum()
    # for c in test_df.columns:
    # test_df = test_df.dropna()

    news_text = remove_stopwords(news_text)
    # df['title'] = df['title'].apply(remove_stopwords)
    # df['author'] = df['author'].apply(remove_stopwords)

    news_text = remove_numbers(news_text)
    # df['title'] = df['title'].apply(remove_numbers)
    # df['author'] = df['author'].apply(remove_numbers)

    news_text = lemmatization(news_text)
    # df['title'] = df['title'].apply(lemmatization)
    # df['author'] = df['author'].apply(lemmatization)


    voc_size = 5000
    maxlen = 386
    onehot_rep_test = [one_hot(news_text,voc_size)]
    # print(len(onehot_rep_test))
    one_hot_embeddings_test = pad_sequences(onehot_rep_test,padding='pre',maxlen=maxlen)
    # print(len(one_hot_embeddings_test))
    X_test = one_hot_embeddings_test
    # print("X", len(X_test), len(X_test[0]))
    y_pred_value = trained_model.predict(X_test)[0]
    # print(y_pred_value)
    y_prob = trained_model.predict_proba(X_test)[0]
    # print(y_prob)
    y_pred_prob = round(y_prob[y_pred_value], 3)
    # y_pred_prob = 0.50

    if y_pred_value == 1:
        answer_results = ["Fake News",y_pred_prob]
    else:
        answer_results = ["True News",y_pred_prob]


    return render_template("predict.html", results_computed=True, result = answer_results)

if __name__ == '__main__':
    app.run(debug = True)
