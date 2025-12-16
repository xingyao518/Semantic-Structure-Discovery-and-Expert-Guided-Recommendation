"""
Recommendation System Runner Module
"""

import sys
import os
import json
import numpy as np
from typing import List, Dict, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.recommendation import RecommendationSystem, TFIDFRetriever, LDATopicRetriever


def run_recommendation(
    texts: List[str],
    theta: np.ndarray,
    test_queries: List[str] = None,
    output_path: str = "data/processed/retrieval_results.json"
) -> Dict:
    """
    Run recommendation system evaluation.
    
    Args:
        texts: Original text documents
        theta: Document-topic distributions
        test_queries: Test queries for evaluation
        output_path: Path to save results
        
    Returns:
        Dictionary of retrieval results
    """
    print("\n" + "=" * 50)
    print("  RECOMMENDATION SYSTEM")
    print("=" * 50)
    sys.stdout.flush()
    
    if test_queries is None:
        test_queries = [
            "How to recover from knee injury?",
            "Best marathon training plan for beginners",
            "Shin splints treatment and prevention",
            "Proper running form and technique",
            "Nutrition tips for long distance runners"
        ]
    
    print(f"[Recommendation] Configuration:")
    print(f"  - Documents: {len(texts)}")
    print(f"  - Test queries: {len(test_queries)}")
    sys.stdout.flush()
    
    # Initialize system
    print("[Recommendation] Initializing retrieval systems...")
    sys.stdout.flush()
    
    rec_system = RecommendationSystem()
    
    # Fit TF-IDF
    print("[Recommendation] Fitting TF-IDF retriever...")
    sys.stdout.flush()
    rec_system.fit_tfidf(texts)
    
    # Fit LDA retriever
    print("[Recommendation] Fitting LDA topic retriever...")
    sys.stdout.flush()
    rec_system.fit_lda(theta, texts)
    
    # Run evaluation
    print("[Recommendation] Running retrieval evaluation...")
    sys.stdout.flush()
    
    results = {
        'queries': [],
        'tfidf_results': [],
        'lda_results': []
    }
    
    for i, query in enumerate(test_queries):
        print(f"[Recommendation] Query {i+1}/{len(test_queries)}: {query[:50]}...")
        sys.stdout.flush()
        
        # TF-IDF retrieval
        tfidf_results = rec_system.retrieve(query, method='tfidf', top_k=5)
        
        # LDA retrieval (use average topic distribution as query)
        query_topic_dist = theta.mean(axis=0)  # Use corpus average as proxy
        lda_results = rec_system.retrieve(
            query, 
            query_topic_dist=query_topic_dist,
            method='lda', 
            top_k=5
        )
        
        results['queries'].append(query)
        results['tfidf_results'].append([
            {'doc_idx': int(idx), 'score': float(score)} 
            for idx, score in tfidf_results
        ])
        results['lda_results'].append([
            {'doc_idx': int(idx), 'score': float(score)} 
            for idx, score in lda_results
        ])
        
        # Print top result
        if tfidf_results:
            top_idx, top_score = tfidf_results[0]
            print(f"  TF-IDF top: doc {top_idx}, score={top_score:.3f}")
        if lda_results:
            top_idx, top_score = lda_results[0]
            print(f"  LDA top: doc {top_idx}, score={top_score:.3f}")
        sys.stdout.flush()
    
    # Save results
    print(f"[Recommendation] Saving results to {output_path}...")
    sys.stdout.flush()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Results saved to: {output_path}")
    print("[INFO] Recommendation completed")
    sys.stdout.flush()
    
    return results


if __name__ == "__main__":
    print("Recommendation Runner - requires texts and theta")


