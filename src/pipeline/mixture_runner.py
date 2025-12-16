"""
Mixture Model Runner Module
"""

import sys
import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.bayesian_mixture import BayesianMixtureOfExperts


def run_mixture_model(
    theta: np.ndarray,
    num_experts: int = 5,
    num_iterations: int = 50,
    output_path: str = "data/processed/mixture_model.pkl",
    metrics_path: str = "data/processed/metrics/mixture_metrics.json"
) -> Tuple[np.ndarray, List[float]]:
    """
    Train Mixture-of-Experts model on LDA topic distributions.
    
    Args:
        theta: Document-topic distributions (N x K)
        num_experts: Number of mixture components
        num_iterations: Number of EM iterations
        output_path: Path to save model
        metrics_path: Path to save metrics
        
    Returns:
        Tuple of (responsibilities, loglik_history)
    """
    print("\n" + "=" * 50)
    print("  MIXTURE OF EXPERTS (GMM)")
    print("=" * 50)
    sys.stdout.flush()
    
    num_docs, num_topics = theta.shape
    
    print(f"[Mixture] Configuration:")
    print(f"  - Documents: {num_docs}")
    print(f"  - Feature dimension (topics): {num_topics}")
    print(f"  - Number of experts: {num_experts}")
    print(f"  - Max iterations: {num_iterations}")
    sys.stdout.flush()
    
    # Initialize model
    print("[Mixture] Initializing model...")
    sys.stdout.flush()
    
    model = BayesianMixtureOfExperts(
        num_experts=num_experts,
        feature_dim=num_topics,
        random_seed=42
    )
    
    # Train
    print("[Mixture] Starting EM training...")
    sys.stdout.flush()
    
    # Convert to list for training
    topic_mixtures = [theta[i] for i in range(theta.shape[0])]
    
    loglik_history = model.fit(topic_mixtures, num_iterations=num_iterations)
    
    responsibilities = model.responsibilities
    
    print(f"[Mixture] Training complete!")
    print(f"  - Responsibilities shape: {responsibilities.shape}")
    print(f"  - Final log-likelihood: {loglik_history[-1]:.2f}")
    print(f"  - Mixture weights: {model.pi}")
    sys.stdout.flush()
    
    # Save model
    print(f"[Mixture] Saving model to {output_path}...")
    sys.stdout.flush()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"[INFO] Model saved to: {output_path}")
    sys.stdout.flush()
    
    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    metrics = {
        'num_experts': num_experts,
        'num_iterations': len(loglik_history),
        'final_loglik': float(loglik_history[-1]) if loglik_history else 0,
        'loglik_history': [float(x) for x in loglik_history],
        'mixture_weights': model.pi.tolist()
    }
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[INFO] Metrics saved to: {metrics_path}")
    print("[INFO] Mixture model completed")
    sys.stdout.flush()
    
    return responsibilities, loglik_history, model.pi


if __name__ == "__main__":
    print("Mixture Runner - requires LDA theta matrix")

