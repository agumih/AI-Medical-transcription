import pandas as pd
import numpy as np
import random
import re
from datasets import load_dataset
from typing import List, Dict, Tuple

#  Configuration
TERMINOLOGY_ERROR_RATE = 0.05 
OUTPUT_FILENAME = 'terminology_detection_tags.csv'
MAX_SAMPLES = 5000

# Terminology Confusion Mapping
# Maps: Correct Medical Term to Possible Terminology Error
MEDICAL_TERM_PAIRS = {
    'infarction': 'infection',          
    'hypertension': 'hypotension',      
    'tachycardia': 'bradycardia',        
    'dyspnea': 'dysphagia',              
    'nephritis': 'hepatitis',           
    'intravenous': 'intra veinous',      
    'carotid': 'karotid',              
    'malignant': 'benign',              
    'diagnosis': 'diagnostics',
    'prognosis': 'prognosis report'
}

def introduce_terminology_error(word: str, error_map: Dict[str, str]) -> Tuple[str, bool]:
    word_lower = word.lower()
    
    if word_lower in error_map:
        if random.random() < 0.9: 
             return error_map[word_lower], True
    return word, False


def generate_and_tag_terminology_data(df: pd.DataFrame, error_rate: float, term_map: Dict[str, str]) -> pd.DataFrame:
    detection_data = []
    
    for _, row in df.iterrows():
        clean_text = row['clean_text']
        clean_words = re.findall(r'\b\w+\b', clean_text.lower())
        corrupted_words = []
        tags = []
        
        for word in clean_words:
            original_word = word
            
            # Check whether to replace the word
            if original_word in term_map.keys() and random.random() < error_rate:
                corrupted_word, error_flag = introduce_terminology_error(original_word, term_map)
                if error_flag:
                    corrupted_tokens = corrupted_word.split()
                    tags.append('B-ERROR')
                    tags.extend(['I-ERROR'] * (len(corrupted_tokens) - 1))
                    corrupted_words.extend(corrupted_tokens)
                else:
                    corrupted_words.append(original_word)
                    tags.append('O')
            else:
                corrupted_words.append(original_word)
                tags.append('O')

        if len(corrupted_words) != len(tags):
             print(f"Skipping row due to tokenization mismatch: {len(corrupted_words)} != {len(tags)}")
             continue

        if 'B-ERROR' in tags:
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
    df = df[df['clean_text'].str.len() > 100].head(MAX_SAMPLES).reset_index(drop=True)
    
    print(f"Cleaned and loaded {len(df)} reports.")

    print(f"Generating synthetic terminology errors and tags at {TERMINOLOGY_ERROR_RATE*100}% word error rate...")
    detection_df = generate_and_tag_terminology_data(df, error_rate=TERMINOLOGY_ERROR_RATE, term_map=MEDICAL_TERM_PAIRS)

    detection_df['words'] = detection_df['words'].apply(lambda x: ' '.join(x))
    detection_df['tags'] = detection_df['tags'].apply(lambda x: ' '.join(x))
    
    detection_df.to_csv(OUTPUT_FILENAME, index=False)
    
    print(f"\n Data augmentation complete. Saved {len(detection_df)} samples with errors to {OUTPUT_FILENAME}")
    print("Output Columns: 'words' (Input Sequence) and 'tags' (Target Tag Sequence: O, B-ERROR, I-ERROR)")
    
    if not detection_df.empty:
        sample_row = detection_df.iloc[0]
        print("\n--- Sample Output ---")
        print(f"Corrupted Words (Input): {sample_row['words']}")
        print(f"Tags (Target): {sample_row['tags']}")