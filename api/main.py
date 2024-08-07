import sys
from pathlib import Path

# sys.path.append(str(Path.cwd().parent))
sys.path.append(str(Path.cwd()))

from fastapi import FastAPI
import pickle
from src.preprocess import text_normalizer
from settings.params import MODEL_PARAMS

app = FastAPI()

# load the classifier
pickle_in = open("models/classifier.pkl", "rb")
classifier = pickle.load(pickle_in)
# load the vectorizer
pickle_in_vect = open("models/tfidf_vectorizer.pkl", "rb")
tfidf_vectorizer = pickle.load(pickle_in_vect)


@app.post("/predict")
def predict_label(article_description: str):
    # prediction
    text_norm = text_normalizer(article_description)
    text_tfidf = tfidf_vectorizer.transform([text_norm])
    predicted_label_index = classifier.predict(text_tfidf)[0]
    predicted_label = (
        ""
        if article_description == ""
        else MODEL_PARAMS["TARGET_LABELS"][predicted_label_index]
    )
    return {"predicted_label": predicted_label}


@app.get("/")
def read_root():
    return {"status": "ok"}
