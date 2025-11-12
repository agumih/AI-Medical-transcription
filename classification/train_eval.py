import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from classification import build_pipeline

def load_mtsamples():
    """
    Loads a sample medical transcription dataset from GitHub.
    No authentication or local download required.
    """
    url = "https://raw.githubusercontent.com/salgadev/medical-nlp/master/data/mtsamples.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={"transcription": "text", "medical_specialty": "label"})
    df = df[["text", "label"]].dropna().reset_index(drop=True)
    return df

def main():
    print("Loading data...")
    df = load_mtsamples()

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    clf = build_pipeline()
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n=== Victor's Model Results ===")
    print(f"Accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    main()

