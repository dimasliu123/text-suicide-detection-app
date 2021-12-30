from string import punctuation
from datetime import datetime
import sqlite3 as sql
import re
import numpy as np
import joblib
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from xgboost import XGBClassifier
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from flask import Flask, render_template, url_for, request, redirect

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('stem')
nltk.download('wordnet')

app = Flask(__name__)

# vectorizer = joblib.load(open('model/vectorize.pkl', 'rb'))
# model = joblib.load(open('model/xgboost.pkl', 'rb'))

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.15, **kwargs):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
                'embed_dim' : self.embed_dim,
                'num_heads' : self.num_heads,
                'ff_dim' : self.ff_dim,
                'rate' : self.rate,
                'att' : self.att, 
                'ffn' : self.ffn, 
                'layernorm1' : self.layernorm1,
                'layernorm2' : self.layernorm2,
                'dropout1' : self.dropout1,
                'dropout2' : self.dropout2
               })
        return config
    
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim, mask_zero=True)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super(TokenAndPositionEmbedding, self).get_config()
        config.update({
            'maxlen' : self.maxlen,
            'vocab_size' : self.vocab_size,
            'embed_dim' : self.embed_dim,
            'token_emb' : self.token_emb,
            'pos_emb' : self.pos_emb
        })
        return config

tokenizer = joblib.load(open('model/transformer_tokens.pkl', 'rb'))
model = tf.keras.models.load_model('model/transformer_model.h5', custom_objects = {"TokenAndPositionEmbedding" : TokenAndPositionEmbedding, "TransformerBlock" : TransformerBlock})

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

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        date = datetime.utcnow()
        to_predict_list = request.form.to_dict()
        clean_text = text_cleaning(to_predict_list['text'])
        # vect = vectorizer.transform([clean_text])
        # predict = model.predict(vect)
        tokens = tokenizer.texts_to_sequences([clean_text])
        seq = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen=100)
        prediction = model.predict(seq)
        predict = np.argmax(prediction, axis=-1)
        try :
            conn = sql.connect('user.db')
            curr = conn.cursor()
            curr.execute("INSERT INTO user (text, clean_text, ml_predictions, date_created) VALUES (?,?,?,?)",(to_predict_list['text'], clean_text, int(predict), date))
            conn.commit()
            print("Inserted Successfully")
        except :
            conn.rollback()
            print("Can't insert")
            print(date, to_predict_list['text'], clean_text, predict)
        finally :
            return render_template('predict.html', prediction=predict)
            conn.close()

        
@app.route('/index')
def return_page():
    return render_template("index.html")

if __name__ == "__main__":
    app.run()