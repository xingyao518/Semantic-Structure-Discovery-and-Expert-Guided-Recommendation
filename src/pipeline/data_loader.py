"""
Data Loader Module - Loads processed_data.json
"""

import json
import sys
import os
from typing import List, Dict, Any


def load_data(data_path: str = "data/processed/processed_data.json") -> Dict[str, Any]:
    """
    Load processed data from JSON file.
    
    Args:
        data_path: Path to processed_data.json
        
    Returns:
        Dictionary containing documents and metadata
    """
    print(f"[INFO] Loading data from: {data_path}")
    sys.stdout.flush()
    
    if not os.path.exists(data_path):
        print(f"[ERROR] File not found: {data_path}")
        sys.stdout.flush()
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different formats
        if isinstance(data, list):
            documents = data
        elif isinstance(data, dict) and 'documents' in data:
            documents = data['documents']
        else:
            documents = data
        
        num_docs = len(documents)
        print(f"[INFO] Data loaded: {num_docs} documents")
        sys.stdout.flush()
        
        return {
            'documents': documents,
            'num_documents': num_docs,
            'raw_data': data
        }
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        sys.stdout.flush()
        raise
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.stdout.flush()
        raise


def extract_texts(data: Dict[str, Any]) -> List[str]:
    """
    Extract text content from loaded data.
    
    Args:
        data: Loaded data dictionary
        
    Returns:
        List of text strings
    """
    documents = data['documents']
    texts = []
    
    for doc in documents:
        if isinstance(doc, dict):
            # Try common text field names
            text = doc.get('text') or doc.get('clean_text') or doc.get('content') or ''
        else:
            text = str(doc)
        texts.append(text)
    
    print(f"[INFO] Extracted {len(texts)} text documents")
    sys.stdout.flush()
    
    return texts


if __name__ == "__main__":
    print("Testing data_loader...")
    data = load_data()
    texts = extract_texts(data)
    print(f"Sample text: {texts[0][:100]}...")


