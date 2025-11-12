from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def build_pipeline():
    """
    Builds a TF-IDF + Logistic Regression text classification pipeline.
    This model will be used to classify medical transcription text
    into the correct medical specialty category.
    """
    return Pipeline
