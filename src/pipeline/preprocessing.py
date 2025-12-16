"""
Preprocessing Module - Wraps TextPreprocessor for pipeline use
"""

import sys
import os
from typing import List, Dict, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.preprocess import TextPreprocessor


class PipelinePreprocessor:
    """
    Wrapper around TextPreprocessor for pipeline integration.
    """
    
    def __init__(self):
        """Initialize preprocessor."""
        print("[INFO] Initializing TextPreprocessor...")
        sys.stdout.flush()
        self.preprocessor = TextPreprocessor()
        self.vocab = None
        self.id_to_word = None
        self.tokenized_docs = None
        self.word_id_docs = None
    
    def preprocess(self, texts: List[str]) -> Tuple[List[List[int]], Dict[str, int], Dict[int, str]]:
        """
        Run full preprocessing pipeline.
        
        Args:
            texts: List of raw text documents
            
        Returns:
            Tuple of (word_id_docs, vocab, id_to_word)
        """
        print(f"[INFO] Preprocessing {len(texts)} documents...")
        sys.stdout.flush()
        
        # Tokenize
        print("[INFO] Tokenizing documents...")
        sys.stdout.flush()
        self.tokenized_docs = self.preprocessor.preprocess_documents(texts)
        
        # Build vocabulary
        print("[INFO] Building vocabulary...")
        sys.stdout.flush()
        self.vocab = self.preprocessor.build_vocabulary(self.tokenized_docs)
        
        # Create id_to_word mapping
        self.id_to_word = {v: k for k, v in self.vocab.items()}
        
        # Convert to word IDs
        print("[INFO] Converting to word IDs...")
        sys.stdout.flush()
        self.word_id_docs = self.preprocessor.docs_to_word_ids(self.tokenized_docs, self.vocab)
        
        # Filter empty documents
        non_empty_docs = [doc for doc in self.word_id_docs if len(doc) > 0]
        
        print(f"[INFO] Preprocessing finished: vocab size = {len(self.vocab)}")
        print(f"[INFO] Non-empty documents: {len(non_empty_docs)}/{len(self.word_id_docs)}")
        sys.stdout.flush()
        
        return non_empty_docs, self.vocab, self.id_to_word
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab) if self.vocab else 0


def run_preprocessing(texts: List[str]) -> Tuple[List[List[int]], Dict[str, int], Dict[int, str]]:
    """
    Convenience function to run preprocessing.
    
    Args:
        texts: List of raw text documents
        
    Returns:
        Tuple of (word_id_docs, vocab, id_to_word)
    """
    preprocessor = PipelinePreprocessor()
    return preprocessor.preprocess(texts)


if __name__ == "__main__":
    print("Testing preprocessing module...")
    sample_texts = [
        "I went running yesterday and my knee hurts.",
        "What's a good marathon training plan for beginners?",
        "Shin splints are common running injuries."
    ]
    word_id_docs, vocab, id_to_word = run_preprocessing(sample_texts)
    print(f"Sample vocab: {list(vocab.items())[:10]}")


