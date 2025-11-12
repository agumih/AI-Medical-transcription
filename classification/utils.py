from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def build_pipeline():
    """
    Builds a TF-IDF + Logistic Regression text classification pipeline.
    This model will be used to classify medical transcription text
    into the correct medical specialty category.
    """
    vectorizer = TfidfVectorizer()
    classifier = LogisticRegression()

    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])

    return pipeline

# Main function to load data, train and evaluate the model
if __name__ == "__main__":
    # Load data - Replace this with the correct file path to your dataset
    # Assume clinical_notes.txt has 'text' and 'label' columns for demo purposes
    with open('clinical_notes.txt', 'r') as file:
        data = file.readlines()

    # Example: Split the data into 'text' and 'label' (Adjust based on actual data format)
    # Here we assume each line is a medical note with a category at the end, e.g., "Note text, category"
    # Modify this as needed based on how your data is structured
    X = [line.split(',')[0] for line in data]  # Text (medical notes)
    y = [line.split(',')[1].strip() for line in data]  # Labels (categories)

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate the model on test data
    predictions = pipeline.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")

    # Optionally, print some predictions
    print("Sample Predictions:", predictions[:10])  # Print the first 10 predictions
