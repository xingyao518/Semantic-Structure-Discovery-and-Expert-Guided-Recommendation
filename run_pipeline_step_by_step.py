"""
Step-by-step pipeline execution with error handling
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
sys.path.insert(0, 'visualization')

print("=" * 80)
print("STEP-BY-STEP PIPELINE EXECUTION")
print("=" * 80)

try:
    # STEP 1: Load data
    print("\n[STEP 1] Loading data...")
    with open("data/processed/processed_data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        documents = [doc['text'] for doc in data[:500]]  # Small subset
    else:
        documents = [doc['text'] for doc in data['documents'][:500]]
    
    print(f"Loaded {len(documents)} documents")
    
    # STEP 2: Preprocess
    print("\n[STEP 2] Preprocessing...")
    from data.preprocess import TextPreprocessor  # pyright: ignore[reportMissingImports]
    preprocessor = TextPreprocessor()
    tokenized_docs = preprocessor.preprocess_documents(documents)
    vocab = preprocessor.build_vocabulary(tokenized_docs)
    word_id_docs = preprocessor.docs_to_word_ids(tokenized_docs, vocab)
    print(f"Vocabulary size: {len(vocab)}")
    
    # STEP 3: Train LDA Gibbs (small)
    print("\n[STEP 3] Training LDA Gibbs (10 iterations)...")
    from models.lda_gibbs import LDAGibbsSampler  # pyright: ignore[reportMissingImports]
    lda_gibbs = LDAGibbsSampler(num_topics=5, alpha=0.1, beta=0.01, random_seed=42)
    theta_gibbs, phi_gibbs = lda_gibbs.fit(
        documents=word_id_docs,
        vocab_size=len(vocab),
        num_iterations=10,
        burn_in=5
    )
    print(f"LDA Gibbs complete: theta={theta_gibbs.shape}, phi={phi_gibbs.shape}")
    
    # STEP 4: Generate visualizations
    print("\n[STEP 4] Generating visualizations...")
    os.makedirs("data/processed/visualizations/lda", exist_ok=True)
    
    from visualize_lda import plot_top_words, plot_topic_distributions  # pyright: ignore[reportMissingImports]
    plot_top_words(phi_gibbs, vocab, top_n=10, save_path="data/processed/visualizations/lda")
    plot_topic_distributions(theta_gibbs, save_path="data/processed/visualizations/lda")
    print("Visualizations saved!")
    
    print("\n" + "=" * 80)
    print("PIPELINE TEST COMPLETE!")
    print("=" * 80)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

