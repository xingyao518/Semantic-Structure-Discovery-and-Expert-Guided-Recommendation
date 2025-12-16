"""
Preprocessing script for Kaggle fitness dataset.

This script loads, cleans, and filters fitness-related posts from a CSV file,
creating a corpus suitable for probabilistic topic modeling and advice generation.
"""

import pandas as pd
import json
import re
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load CSV file using pandas.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame
    """
    df = pd.read_csv(csv_path, encoding='utf-8')
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select required columns if they exist.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with selected columns
    """
    required_cols = ['text', 'title', 'score', 'datetime', 'post_id']
    available_cols = [col for col in required_cols if col in df.columns]
    
    if 'text' not in available_cols:
        raise ValueError("Required column 'text' not found in CSV")
    
    # Keep only available columns
    return df[available_cols]


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unusable rows (null text, deleted/removed posts).
    
    Args:
        df: Input DataFrame
        
    Returns:
        Filtered DataFrame
    """
    # Drop rows where text is null
    df = df.dropna(subset=['text'])
    
    # Remove deleted/removed posts
    deleted_patterns = ['deleted', 'removed', '[deleted]', '[removed]']
    mask = ~df['text'].str.lower().isin(deleted_patterns)
    df = df[mask]
    
    return df


def clean_text(text: str) -> str:
    """
    Clean text by removing markdown, URLs, and normalizing whitespace.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove markdown characters: >, *, _, ~, backticks
    text = re.sub(r'[>]', '', text)  # Remove blockquote markers
    text = re.sub(r'[*_]', '', text)  # Remove bold/italic markers
    text = re.sub(r'[~]', '', text)   # Remove strikethrough
    text = re.sub(r'`', '', text)     # Remove code backticks
    
    # Remove URLs using regex: r"http\S+|www\S+"
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
    
    # Strip extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def apply_text_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to all rows using vectorized operations.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with 'clean_text' column added
    """
    # Use vectorized string operations for faster processing
    df = df.copy()
    
    # Convert to string and lowercase (vectorized)
    print("    Converting to lowercase...")
    df['clean_text'] = df['text'].astype(str).str.lower()
    
    # Remove markdown characters (vectorized)
    print("    Removing markdown characters...")
    df['clean_text'] = df['clean_text'].str.replace(r'[>]', '', regex=True)
    df['clean_text'] = df['clean_text'].str.replace(r'[*_]', '', regex=True)
    df['clean_text'] = df['clean_text'].str.replace(r'[~]', '', regex=True)
    df['clean_text'] = df['clean_text'].str.replace(r'`', '', regex=True)
    
    # Remove URLs (vectorized)
    print("    Removing URLs...")
    df['clean_text'] = df['clean_text'].str.replace(r'http\S+|www\S+', '', regex=True)
    
    # Strip extra whitespace (vectorized)
    print("    Normalizing whitespace...")
    df['clean_text'] = df['clean_text'].str.replace(r'\s+', ' ', regex=True)
    df['clean_text'] = df['clean_text'].str.strip()
    
    return df


def filter_fitness_related(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter posts to include only fitness-related content.
    
    Args:
        df: Input DataFrame with 'clean_text' column
        
    Returns:
        Filtered DataFrame
    """
    fitness_keywords = [
        "run", "running", "injury", "recover", "knee", "ankle", "shin",
        "marathon", "race", "pace", "mileage", "training", "sprint",
        "treadmill", "stride", "coach", "training plan", "workout",
        "strength", "legs", "cardio", "aerobic", "distance", "speed"
    ]
    
    # Create pattern for matching any keyword (case-insensitive)
    # Escape special regex characters
    pattern = '|'.join([re.escape(kw) for kw in fitness_keywords])
    
    # Filter rows where clean_text contains any keyword
    mask = df['clean_text'].str.contains(pattern, case=False, na=False, regex=True)
    df_filtered = df[mask].copy()
    
    return df_filtered


def save_json(df: pd.DataFrame, output_path: str):
    """
    Save processed data to JSON format (as a list of documents).
    
    Args:
        df: Processed DataFrame
        output_path: Path to save JSON file
    """
    # Ensure directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to list of dictionaries
    documents = []
    for idx, row in df.iterrows():
        doc = {
            "id": int(idx),  # Use index as ID
            "text": str(row['clean_text']),
            "title": str(row.get('title', '')) if pd.notna(row.get('title')) else "",
            "score": int(row['score']) if pd.notna(row.get('score')) else 0
        }
        documents.append(doc)
    
    # Save as JSON with documents key (UTF-8 encoding and indent=2)
    output_data = {"documents": documents}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(documents)} documents to {output_path}")


def main():
    """
    Main preprocessing pipeline.
    """
    csv_path = "data/raw/data.csv"
    output_path = "data/processed/processed_data.json"
    
    print("=" * 60)
    print("Kaggle Fitness Dataset Preprocessing")
    print("=" * 60)
    
    # Step 1: Load CSV
    print("\nStep 1: Loading CSV...")
    if not Path(csv_path).exists():
        print(f"ERROR: {csv_path} not found!")
        print("Please ensure the Kaggle dataset is placed at data/raw/data.csv")
        return
    
    df = load_csv(csv_path)
    total_raw_rows = len(df)
    print(f"  Total raw rows: {total_raw_rows}")
    
    # Step 2: Select columns
    print("\nStep 2: Selecting columns...")
    df = select_columns(df)
    print(f"  Selected columns: {list(df.columns)}")
    
    # Step 3: Filter rows
    print("\nStep 3: Filtering rows (removing null/deleted)...")
    df = filter_rows(df)
    rows_after_cleaning = len(df)
    print(f"  Rows after cleaning: {rows_after_cleaning}")
    
    # Step 4: Clean text
    print("\nStep 4: Cleaning text...")
    print("  Processing text cleaning (this may take a few minutes)...")
    df = apply_text_cleaning(df)
    print("  Text cleaning completed")
    
    # Step 5: Filter fitness-related
    print("\nStep 5: Filtering fitness-related posts...")
    df = filter_fitness_related(df)
    rows_after_filtering = len(df)
    print(f"  Rows after filtering: {rows_after_filtering}")
    
    # Step 6: Save output
    print(f"\nStep 6: Saving to {output_path}...")
    save_json(df, output_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("Preprocessing Summary:")
    print(f"  Total raw rows: {total_raw_rows}")
    print(f"  Rows after cleaning: {rows_after_cleaning}")
    print(f"  Rows after filtering: {rows_after_filtering}")
    print(f"  Output path: {output_path}")
    print("=" * 60)
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
