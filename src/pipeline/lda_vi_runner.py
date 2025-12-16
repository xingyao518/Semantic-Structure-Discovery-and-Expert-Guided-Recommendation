"""
LDA Variational Inference Runner Module
"""

import sys
import os
import json
import numpy as np
from typing import List, Dict, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.lda_vi import LDAVariationalInference


def run_lda_vi(
    word_id_docs: List[List[int]],
    vocab: Dict[str, int],
    id_to_word: Dict[int, str],
    num_topics: int = 10,
    num_iterations: int = 50,
    alpha: float = 0.1,
    beta: float = 0.01,
    output_path: str = "data/processed/lda_vi_model.json",
    metrics_path: str = "data/processed/metrics/lda_vi_metrics.json"
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Train LDA using Variational Inference.
    
    Args:
        word_id_docs: List of documents as word ID lists
        vocab: Word to ID mapping
        id_to_word: ID to word mapping
        num_topics: Number of topics
        num_iterations: Number of VI iterations
        alpha: Document-topic prior
        beta: Topic-word prior
        output_path: Path to save model
        metrics_path: Path to save metrics
        
    Returns:
        Tuple of (theta, phi, elbo_history)
    """
    print("\n" + "=" * 50)
    print("  LDA VARIATIONAL INFERENCE")
    print("=" * 50)
    sys.stdout.flush()
    
    vocab_size = len(vocab)
    num_docs = len(word_id_docs)
    
    print(f"[LDA-VI] Configuration:")
    print(f"  - Documents: {num_docs}")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Topics: {num_topics}")
    print(f"  - Max iterations: {num_iterations}")
    print(f"  - Alpha: {alpha}, Beta: {beta}")
    sys.stdout.flush()
    
    # Initialize model
    print("[LDA-VI] Initializing model...")
    sys.stdout.flush()
    
    model = LDAVariationalInference(
        num_topics=num_topics,
        alpha=alpha,
        beta=beta,
        random_seed=42
    )
    
    # Train with verbose output
    print("[LDA-VI] Starting training...")
    sys.stdout.flush()
    
    # Initialize
    model._initialize(word_id_docs, vocab_size)
    
    prev_elbo = -np.inf
    elbo_history = []
    
    for iteration in range(num_iterations):
        # Update variational parameters
        for d in range(model.D):
            model._update_phi(d, word_id_docs)
            model._update_gamma(d, word_id_docs)
        
        model._update_lambda(word_id_docs)
        
        # Compute ELBO
        elbo = model._compute_elbo(word_id_docs)
        elbo_history.append(float(elbo))
        
        print(f"[LDA-VI] Iter {iteration + 1}/{num_iterations}, ELBO = {elbo:.2f}")
        sys.stdout.flush()
        
        # Check convergence
        if abs(elbo - prev_elbo) < 1e-4 and iteration > 10:
            print(f"[LDA-VI] Converged at iteration {iteration + 1}")
            sys.stdout.flush()
            break
        
        prev_elbo = elbo
    
    # Estimate final parameters
    model._estimate_parameters()
    
    theta = model.theta
    phi = model.phi_est
    
    print(f"[LDA-VI] Training complete!")
    print(f"  - Theta shape: {theta.shape}")
    print(f"  - Phi shape: {phi.shape}")
    sys.stdout.flush()
    
    # Save model
    print(f"[LDA-VI] Saving model to {output_path}...")
    sys.stdout.flush()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model_data = {
        'theta': theta.tolist(),
        'phi': phi.tolist(),
        'vocab': vocab,
        'num_topics': num_topics,
        'alpha': alpha,
        'beta': beta,
        'model_type': 'vi',
        'elbo_history': elbo_history
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"[INFO] Model saved to: {output_path}")
    sys.stdout.flush()
    
    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    metrics = {
        'num_topics': num_topics,
        'num_iterations': len(elbo_history),
        'final_elbo': elbo_history[-1] if elbo_history else 0,
        'elbo_history': elbo_history
    }
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[INFO] Metrics saved to: {metrics_path}")
    print("[INFO] LDA-VI completed")
    sys.stdout.flush()
    
    return theta, phi, elbo_history


if __name__ == "__main__":
    print("LDA VI Runner - requires preprocessed data")


