import pandas as pd
import numpy as np
import random
import re
from datasets import load_dataset
from typing import List, Dict, Tuple

# Configuration
WORD_ERROR_RATE = 0.05 
OUTPUT_FILENAME= 'spelling_detection_tags.csv'
LETTERS = 'abcdefghijklmnopqrstuvwxyz'

def introduce_spelling_error(word: str) -> str:

    word = word.lower()
    if len(word) < 4 or not re.search('[a-z]', word):
        return word
    
    op = random.choice(['swap', 'delete', 'insert', 'substitute'])
    idx = random.randint(0, len(word) - 1)
    
    if op == 'delete':
        return word[:idx] + word[idx+1:]
    elif op == 'insert':
        char = random.choice(LETTERS)
        return word[:idx] + char + word[idx:]
    elif op == 'swap' and idx < len(word) - 1:
        return word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
    elif op == 'substitute':
        char = random.choice(LETTERS)
        return word[:idx] + char + word[idx+1:]
    
    return word


def generate_and_tag_spelling_data(df: pd.DataFrame, error_rate: float) -> pd.DataFrame:
    detection_data = []
    
    for _, row in df.iterrows():
        clean_text = row['clean_text']
        
        clean_words = clean_text.lower().split() 
        clean_words = [word for word in clean_words if word]
        corrupted_words = []
        tags = []
        error_introduced_in_report = False
        
        for i, word in enumerate(clean_words):
            
            original_word = word
            
            if random.random() < error_rate:
                corrupted_word = introduce_spelling_error(original_word)
                
                if corrupted_word != original_word:
                    corrupted_words.append(corrupted_word)
                    tags.append('B-ERROR')
                    error_introduced_in_report = True
                else:
                    corrupted_words.append(original_word)
                    tags.append('O')
            else:
                corrupted_words.append(original_word)
                tags.append('O')
        
        detection_data.append({
            'words': corrupted_words, 
            'tags': tags,             
            'clean_text': clean_text 
        })
        
    return pd.DataFrame(detection_data)


if __name__ == "__main__":
    print("Loading MTSamples dataset from Hugging Face...")

    try:
        dataset = load_dataset("harishnair04/mtsamples")
        df = dataset['train'].to_pandas()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    df = df.rename(columns={'transcription': 'clean_text'})
    df = df.dropna(subset=['clean_text'])
    df = df[df['clean_text'].str.len() > 100].reset_index(drop=True)
    
    print(f"Cleaned and loaded {len(df)} reports.")
    print(f"Generating synthetic spelling errors and tags at {WORD_ERROR_RATE*100}% word error rate...")

    detection_df = generate_and_tag_spelling_data(df, error_rate=WORD_ERROR_RATE)
    detection_df['words'] = detection_df['words'].apply(lambda x: ' '.join(x))
    detection_df['tags'] = detection_df['tags'].apply(lambda x: ' '.join(x))
    detection_df.to_csv(OUTPUT_FILENAME, index=False)
    
    print(f"\n Data augmentation complete. Saved {len(detection_df)} samples to {OUTPUT_FILENAME}")
    print("Output Columns: 'words' (Input Sequence) and 'tags' (Target Tag Sequence)")