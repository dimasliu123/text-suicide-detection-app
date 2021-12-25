from string import punctuation
import re
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from xgboost import XGBClassifier
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from flask import Flask, render_template, url_for, request
nltk.download('stopwords')
nltk.download('stem')
nltk.download('wordnet')

app = Flask(__name__)

vectorizer = joblib.load(open('model/vectorize.pkl', 'rb'))
model = joblib.load(open('model/xgboost.pkl', 'rb'))

def text_cleaning(text, remove_stopwords=True, stem_words=True, lemmatize_words=True):
    stop_words = {'a','about','above', 'after', 'again', 'against', 'ain', 'all', 'also', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'com', 'could', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'else', 'ever', 'few', 'for', 'from', 'further', 'get', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'hence', 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'however', 'http', 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'k', "let's", 'like', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'r', 're', 's', 'same', 'shall', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', "should've", 'shouldn', "shouldn't", 'since', 'so', 'some', 'such', 't', 'than', 'that', "that'll", "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'therefore', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', 'weren', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'will', 'with', 'won', "won't", 'would', 'wouldn', "wouldn't", 'www', 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 'im', 'yep', 'hey', 'heyy', 'heyyy'}

    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r"ur", " your ", text)
    text = re.sub(r" nd ", " and ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" c ", " can ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r'http\S+', ' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)  # remove numbers
    text = re.sub(r" u ", " you ", text)
    text = text.lower()  # set in lowercase
    
    text = ''.join([c for c in text if c not in punctuation])
    
    if remove_stopwords:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    if stem_words and lemmatize_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        lemmatizer = WordNetLemmatizer()
        stemmed_words = [stemmer.stem(word) for word in text]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]
        text = " ".join(lemmatized_words)
    return (text)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()
    clean_text = text_cleaning(to_predict_list['text'])
    vect = vectorizer.transform([clean_text])
    predict = model.predict(vect)
    return render_template('predict.html', prediction=predict)

@app.route('/index')
def return_page():
    return render_template("index.html")

if __name__ == "__main__":
    app.run()