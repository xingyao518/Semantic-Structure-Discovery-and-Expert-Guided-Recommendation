"""
Text preprocessing for running Q&A data.

This module handles tokenization, cleaning, lemmatization, and
stopword removal to prepare text for probabilistic topic models.
"""

import re
import string
from typing import List, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextPreprocessor:
    """
    Preprocessor for running-related text data.
    
    Performs:
    - Lowercasing
    - Tokenization
    - Stopword removal
    - Lemmatization
    - Special character removal
    - Number removal (optional)
    """
    
    def __init__(self, 
                 remove_numbers: bool = True,
                 min_word_length: int = 2,
                 custom_stopwords: List[str] = None):
        """
        Initialize preprocessor.
        
        Args:
            remove_numbers: Whether to remove numeric tokens
            min_word_length: Minimum word length to keep
            custom_stopwords: Additional stopwords to remove
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add running-specific stopwords
        running_stopwords = ['run', 'running', 'runner', 'runners', 'ran']
        if custom_stopwords:
            running_stopwords.extend(custom_stopwords)
        
        self.stop_words.update(running_stopwords)
        self.remove_numbers = remove_numbers
        self.min_word_length = min_word_length
    
    def clean_text(self, text: str) -> str:
        """
        Clean raw text by removing URLs, special characters, etc.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove Reddit-specific patterns
        text = re.sub(r'\[deleted\]', '', text)
        text = re.sub(r'\[removed\]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of token strings
        """
        try:
            tokens = word_tokenize(text.lower())
        except:
            # Fallback to simple split
            tokens = text.lower().split()
        
        return tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using WordNet.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        lemmatized = []
        for token in tokens:
            # Use noun as default POS tag
            lemma = self.lemmatizer.lemmatize(token, pos='n')
            # Try verb if different
            lemma_verb = self.lemmatizer.lemmatize(token, pos='v')
            if len(lemma_verb) < len(lemma):
                lemma = lemma_verb
            lemmatized.append(lemma)
        return lemmatized
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by length, stopwords, and numbers.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered list of tokens
        """
        filtered = []
        for token in tokens:
            # Remove punctuation-only tokens
            if token in string.punctuation:
                continue
            
            # Remove short tokens
            if len(token) < self.min_word_length:
                continue
            
            # Remove numbers if requested
            if self.remove_numbers and token.isdigit():
                continue
            
            # Remove stopwords
            if token in self.stop_words:
                continue
            
            # Keep alphanumeric tokens
            if re.match(r'^[a-zA-Z]+$', token):
                filtered.append(token)
        
        return filtered
    
    def preprocess(self, text: str) -> List[str]:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Raw input text
            
        Returns:
            List of preprocessed tokens
        """
        # Clean
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Lemmatize
        lemmatized = self.lemmatize_tokens(tokens)
        
        # Filter
        filtered = self.filter_tokens(lemmatized)
        
        return filtered
    
    def preprocess_documents(self, documents: List[str]) -> List[List[str]]:
        """
        Preprocess a list of documents.
        
        Args:
            documents: List of raw text documents
            
        Returns:
            List of preprocessed token lists
        """
        import sys
        result = []
        total = len(documents)
        for i, doc in enumerate(documents):
            if i % 100 == 0:
                print(f"  [Preprocessing] {i}/{total} documents...")
                sys.stdout.flush()
            result.append(self.preprocess(doc))
        print(f"  [Preprocessing] {total}/{total} documents complete")
        sys.stdout.flush()
        return result
    
    def build_vocabulary(self, tokenized_docs: List[List[str]]) -> Dict[str, int]:
        """
        Build vocabulary mapping from tokenized documents.
        
        Args:
            tokenized_docs: List of token lists
            
        Returns:
            Dictionary mapping word -> word_id
        """
        vocab = {}
        word_id = 0
        
        for doc in tokenized_docs:
            for token in doc:
                if token not in vocab:
                    vocab[token] = word_id
                    word_id += 1
        
        return vocab
    
    def docs_to_word_ids(self, 
                        tokenized_docs: List[List[str]], 
                        vocab: Dict[str, int]) -> List[List[int]]:
        """
        Convert tokenized documents to word ID sequences.
        
        Args:
            tokenized_docs: List of token lists
            vocab: Vocabulary dictionary (word -> word_id)
            
        Returns:
            List of word ID sequences
        """
        word_id_docs = []
        for doc in tokenized_docs:
            word_ids = [vocab[token] for token in doc if token in vocab]
            word_id_docs.append(word_ids)
        return word_id_docs


if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    # Example
    sample_text = "I've been running for 3 months and my knees hurt. Should I rest?"
    tokens = preprocessor.preprocess(sample_text)
    print(f"Original: {sample_text}")
    print(f"Preprocessed: {tokens}")

