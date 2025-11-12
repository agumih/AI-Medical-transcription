import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, Dense, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Dict, Any

# Configuration
DATA_FILE = 'spelling_detection_tags.csv'
MAX_SEQUENCE_LENGTH = 150  
EMBEDDING_DIM = 100
CONV_FILTERS = 128
CONV_KERNEL_SIZE = 5
BATCH_SIZE = 32
EPOCHS = 2


def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray, Tokenizer, LabelEncoder, int]:
    try:
        df = pd.read_csv(DATA_FILE)
        df = df.dropna()
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please ensure the augmentation script was run.")
        exit()
    
    X_data = [[word for word in seq.split() if word] for seq in df['words']]
    
    y_data = [
        [tag for tag in seq.strip().split() if tag] 
        for seq in df['tags']
    ]

    filtered_data = [(x, y) for x, y in zip(X_data, y_data) if len(x) == len(y) and len(x) > 0]
    
    if not filtered_data:
        raise ValueError("Data filtering resulted in an empty dataset. Check your CSV tokenization.")

    X_data, y_data = zip(*filtered_data)
    X_data, y_data = list(X_data), list(y_data)

    print(f"Loaded and filtered {len(X_data)} samples with matching word/tag counts.")
    
    word_tokenizer = Tokenizer(oov_token='<UNK>', lower=False)
    word_tokenizer.fit_on_texts(X_data)
    X_sequences = word_tokenizer.texts_to_sequences(X_data)

    tag_encoder = LabelEncoder()
    tag_encoder.fit(['O', 'B-ERROR'])
    N_TAGS = len(tag_encoder.classes_)
    O_TAG_ID = tag_encoder.transform(['O'])[0] 

    y_sequences = [[tag_encoder.transform([tag])[0] for tag in seq] for seq in y_data]

    X_padded = pad_sequences(X_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', value=0)
    y_padded = pad_sequences(y_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', value=O_TAG_ID)

    y_final = tf.keras.utils.to_categorical(y_padded, num_classes=N_TAGS)

    print(f"Data Ready. X={X_padded.shape}, Y={y_final.shape}. Vocab size: {len(word_tokenizer.word_index) + 1}")
    return X_padded, y_final, word_tokenizer, tag_encoder, N_TAGS


def build_and_train_model(X: np.ndarray, Y: np.ndarray, word_tokenizer: Tokenizer, N_TAGS: int) -> tf.keras.Model:
    VOCAB_SIZE = len(word_tokenizer.word_index) + 1
    
    model = Sequential([
        Embedding(
            input_dim=VOCAB_SIZE, 
            output_dim=EMBEDDING_DIM, 
            input_length=MAX_SEQUENCE_LENGTH,
            
        ),
        Conv1D(
            filters=CONV_FILTERS, 
            kernel_size=CONV_KERNEL_SIZE, 
            padding='same', 
            activation='relu'
        ),
        TimeDistributed(Dense(N_TAGS, activation='softmax'))
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print("\nStarting training.")
    model.fit(
        X_train, y_train, 
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        validation_data=(X_test, y_test)
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nCNN Accuracy Performance")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")
    return model


def check_spelling_errors(new_text: str, model: tf.keras.Model, word_tokenizer: Tokenizer, tag_encoder: LabelEncoder):
    new_sequence = word_tokenizer.texts_to_sequences([new_text.lower().split()])
    new_padded = pad_sequences(new_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', value=0)
    
    predictions = model.predict(new_padded, verbose=0)[0]

    predicted_indices = np.argmax(predictions, axis=1)
    predicted_tags = tag_encoder.inverse_transform(predicted_indices)

    original_words = new_text.split()
    results = list(zip(original_words, predicted_tags[:len(original_words)]))
    
    errors = [(word, tag) for word, tag in results if tag == 'B-ERROR']
    
    if errors:
        print("Spelling Errors Detected:")
        for word, _ in errors:
            print(f" - Word: {word}")
    else:
        print("No spelling errors detected.")
    
    return errors


if __name__ == "__main__":
    X, Y, word_tokenizer, tag_encoder, N_TAGS = load_and_preprocess_data()
    spelling_model = build_and_train_model(X, Y, word_tokenizer, N_TAGS)
    test_sentence_error = "The patinet tolereted the procdure without complicaton."

    print("\nDemonstrating error detection on sample text:")
    check_spelling_errors(test_sentence_error, spelling_model, word_tokenizer, tag_encoder)