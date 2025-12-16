"""
LDA Gibbs Sampling Runner Module
"""

import sys
import os
import json
import numpy as np
from typing import List, Dict, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.lda_gibbs import LDAGibbsSampler


def run_lda_gibbs(
    word_id_docs: List[List[int]],
    vocab: Dict[str, int],
    id_to_word: Dict[int, str],
    num_topics: int = 10,
    num_iterations: int = 100,
    alpha: float = 0.1,
    beta: float = 0.01,
    output_path: str = "data/processed/lda_gibbs_model.json",
    metrics_path: str = "data/processed/metrics/lda_gibbs_metrics.json"
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Train LDA using Gibbs sampling.
    
    Args:
        word_id_docs: List of documents as word ID lists
        vocab: Word to ID mapping
        id_to_word: ID to word mapping
        num_topics: Number of topics
        num_iterations: Number of Gibbs iterations
        alpha: Document-topic prior
        beta: Topic-word prior
        output_path: Path to save model
        metrics_path: Path to save metrics
        
    Returns:
        Tuple of (theta, phi, loglik_history)
    """
    print("\n" + "=" * 50)
    print("  LDA GIBBS SAMPLING")
    print("=" * 50)
    sys.stdout.flush()
    
    vocab_size = len(vocab)
    num_docs = len(word_id_docs)
    
    print(f"[LDA-Gibbs] Configuration:")
    print(f"  - Documents: {num_docs}")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Topics: {num_topics}")
    print(f"  - Iterations: {num_iterations}")
    print(f"  - Alpha: {alpha}, Beta: {beta}")
    sys.stdout.flush()
    
    # Initialize model
    print("[LDA-Gibbs] Initializing sampler...")
    sys.stdout.flush()
    
    sampler = LDAGibbsSampler(
        num_topics=num_topics,
        alpha=alpha,
        beta=beta,
        random_seed=42
    )
    
    # Custom training loop with verbose output
    print("[LDA-Gibbs] Starting training...")
    sys.stdout.flush()
    
    # Initialize
    sampler._initialize(word_id_docs, vocab_size)
    sampler.V = vocab_size
    sampler.D = num_docs
    
    loglik_history = []
    
    for iteration in range(num_iterations):
        # Run one iteration
        for d, doc in enumerate(word_id_docs):
            for n, word_id in enumerate(doc):
                new_topic = sampler._sample_topic(d, n, word_id)
                sampler.z[d][n] = new_topic
        
        # Compute log likelihood
        ll = sampler._compute_log_joint()
        loglik_history.append(float(ll))
        
        # Print progress every iteration
        print(f"[LDA-Gibbs] Iter {iteration + 1}/{num_iterations}, loglik = {ll:.2f}")
        sys.stdout.flush()
    
    # Estimate final parameters
    sampler._estimate_parameters()
    sampler.loglik_history = loglik_history
    
    theta = sampler.theta
    phi = sampler.phi
    
    print(f"[LDA-Gibbs] Training complete!")
    print(f"  - Theta shape: {theta.shape}")
    print(f"  - Phi shape: {phi.shape}")
    sys.stdout.flush()
    
    # Save model
    print(f"[LDA-Gibbs] Saving model to {output_path}...")
    sys.stdout.flush()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model_data = {
        'theta': theta.tolist(),
        'phi': phi.tolist(),
        'vocab': vocab,
        'num_topics': num_topics,
        'alpha': alpha,
        'beta': beta,
        'model_type': 'gibbs',
        'loglik_history': loglik_history
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"[INFO] Model saved to: {output_path}")
    sys.stdout.flush()
    
    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    metrics = {
        'num_topics': num_topics,
        'num_iterations': num_iterations,
        'final_loglik': loglik_history[-1] if loglik_history else 0,
        'log_likelihoods': loglik_history
    }
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[INFO] Metrics saved to: {metrics_path}")
    print("[INFO] LDA-Gibbs completed")
    sys.stdout.flush()
    
    return theta, phi, loglik_history


if __name__ == "__main__":
    print("LDA Gibbs Runner - requires preprocessed data")


