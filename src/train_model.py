import os
import joblib
from src.preprocess import preprocess_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

def train_and_save_model(data_path, model_path):
    df = preprocess_data(data_path)
    X = df['cleaned']
    y = LabelEncoder().fit_transform(df['priority'])

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])

    model.fit(X_train, y_train)

    # ✅ Ensure the directory exists before saving
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)

    return X_test, y_test, model
