"""ecommerce text classification API with FastAPI"""

import sys
from pathlib import Path
import pickle
from fastapi import FastAPI

sys.path.append(str(Path.cwd()))

#pylint: disable=wrong-import-position

from src.preprocess import text_normalizer
from settings.params import MODEL_PARAMS

#pylint: enable=wrong-import-position


app = FastAPI()

# Load the classifier
with open("models/classifier.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)

# Load the vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as pickle_in_vect:
    tfidf_vectorizer = pickle.load(pickle_in_vect)


@app.post("/predict")
def predict_label(article_description: str):
    """
    Predict the label of the given article description.

    Parameters:
    - article_description (str): The description of the article to be classified.

    Returns:
    - dict: A dictionary containing the predicted label of the article.
    """
    # Normalize the input text
    text_norm = text_normalizer(article_description)
    # Transform the normalized text using the loaded TF-IDF vectorizer
    text_tfidf = tfidf_vectorizer.transform([text_norm])
    # Predict the label index using the classifier
    predicted_label_index = classifier.predict(text_tfidf)[0]
    # Map the label index to the actual label
    predicted_label = (
        ""
        if article_description == ""
        else MODEL_PARAMS["TARGET_LABELS"][predicted_label_index]
    )
    return {"predicted_label": predicted_label}


@app.get("/")
def read_root():
    """
    Root endpoint to check the status of the API.

    Returns:
    - dict: A dictionary indicating the status of the API.
    """
    return {"status": "ok"}
