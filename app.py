from string import punctuation
import re
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from xgboost import XGBClassifier
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

vectorizer = joblib.load(open('model/vectorize.pkl', 'rb'))
model = joblib.load(open('model/xgboost.pkl', 'rb'))

def text_cleaning(text, remove_stopwords=True, stem_words=True, lemmatize_words=True):
    stop_words = set(stopwords.words("english"))
    stop_words.update({'im', 'yep', 'hey', 'heyy', 'heyyy'})

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

if __name__ == "__main__":
    app.run()