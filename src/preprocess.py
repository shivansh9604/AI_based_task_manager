import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r'\W', ' ', text.lower())
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df['cleaned'] = df['description'].apply(clean_text)
    return df
