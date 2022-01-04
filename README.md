
# Fake News Classification Tool

This project aims to build a fake news classifier that can accurately distinguish fake news from genuine news. We also developed a simple UI, to enable the users to efficiently verify news articles

<div align="center">
<img src= "https://imgur.com/Jdh1fRr.png" ><p>Figure 1: Overview of the approach</p>
</div>

## Abstract:
   * The advent of technology and the development of social media platforms has made it easier to share news updates with the masses.
    * The effects of fake news can be catastrophic.
    * Fact checking can prevent people from reacting and taking action on fake news.
    * Extremely useful for news houses to fact-check their news before they share it with the masses.

## Procedure:

1. Dataset: 20,800 samples with 10387 real news,10413 fake news. Dataset is publically available [here](https://www.kaggle.com/c/fake-news/data)

3. Preprocessing:
	1. Removing irrelevant texts and NAN values
	2. Stopwords
	3. Numerics and special characters
	4. Lemmatization
	5. Case folding
	6. Tokenization
	7. Padding
	
4. Vectorisation: 
	1. OneHot Encoding
	2. Count Vectoriser
	3. Hashing-Vectorizer
	4. TF-IDF
	5. GloVe Embedding
	6. Word2Vec Embedding
	7. BERT

5. Machine Learning / Deep Learning Algorithms for Fake news classification: 
	1. Naive Bayes (MultinomialNB)
	2. DecionTree
	3. AdaBoost Classification
	4. Logistic Regression
	5. Passive Aggressive
	6. Multilayer Perceptron
	7. LSTM
	8. BERT
<div align="center">
<img src= "https://imgur.com/Qou6VYy.png" ><p>Figure 2: Pipeline of the approach used</p>
</div>

## Pipeline & Output:
### Output 1: BERT Tokeniser with Bert model
<div align="center">
<img src= "https://imgur.com/OnbSNca.png" ><p>Figure 3: Pipeline for BERT Tokeniser with Bert model</p>
</div>

<div align="center">
<img src= "https://imgur.com/QLF41Ah.png" ><p>Figure 4: Performance of BERT Tokeniser with Bert model</p>
</div>

### 2: TF-IDF Tokenizer with Passive Aggressive Classifier
<div align="center"> 
<img src= "https://imgur.com/f6cgGfC.png" ><p>Figure 5: Pipeline for TF-IDF Tokenizer + Passive Aggressive Classifier</p>
</div>

<div align="center"> 
<img src= "https://imgur.com/RdRqvse.png" ><p>Figure 6: Performance of TF-IDF Tokenizer + Passive Aggressive Classifier</p>
</div>



## WebApp:

### [https://fake-news-detection-nlp.herokuapp.com/](https://fake-news-detection-nlp.herokuapp.com/)
- An end-to-end deployed tool which allows user to verify news articles in a click
- Efficient and accurate tool for fact checking
- A simple minimalistic user interface
- It allows user to input news text along with title and author name (both optional fields)
- ‘Load sample input’ button allows users to understand and test the app

## Technologies Used:
1. Flask, HTML, CSS, JS for building the webapp
2. Heroku for deploying the webapp
3. Python, along with Machine learning, deep learning frameworks. 

## Team Members:
This project has been completed as a course project of CSE556: [Natural Language Processing](http://techtree.iiitd.edu.in/viewDescription/filename?=CSE556). 

1. [Ria Gupta](https://github.com/ria18405)
2. [Ruhma Mehek Khan](https://github.com/ruhmamehek/)
3. [Prakriti Garg](mailto:prakriti19439@iiitd.ac.in)

## Report & Slides 

[Presentation Slides](https://github.com/ria18405/fake-news-classifier/blob/deployed_version/Presentation%20Slides.pdf)
[Project Report](https://github.com/ria18405/fake-news-classifier/blob/deployed_version/Report.pdf)
