"""
Visualization functions for retrieval system evaluation.

This module provides visualization tools for comparing different
retrieval methods (TF-IDF, LDA-topic similarity, Matrix Factorization).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import seaborn as sns

def plot_top_k_scores(retrieved_results: List[Tuple[int, float]],
                      top_k: int = 10,
                      method_name: str = "Retrieval Method",
                      save_path: Optional[str] = None):
    """
    Plot top-K retrieval scores as bar chart.
    
    Args:
        retrieved_results: List of (document_index, score) tuples
        top_k: Number of top results to display
        method_name: Name of retrieval method for title
        save_path: Optional path to save figure
    """
    # Sort by score and take top-k
    sorted_results = sorted(retrieved_results, key=lambda x: x[1], reverse=True)[:top_k]
    doc_indices = [r[0] for r in sorted_results]
    scores = [r[1] for r in sorted_results]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(scores)), scores, color='steelblue')
    plt.yticks(range(len(doc_indices)), [f'Doc {idx}' for idx in doc_indices])
    plt.xlabel('Similarity Score')
    plt.ylabel('Document Index')
    plt.title(f'Top-{top_k} Retrieval Scores: {method_name}')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def compare_retrieval_methods(method_results: Dict[str, List[Tuple[int, float]]],
                             top_k: int = 10,
                             save_path: Optional[str] = None):
    """
    Compare multiple retrieval methods side-by-side.
    
    Args:
        method_results: Dictionary mapping method_name -> list of (doc_idx, score) tuples
        top_k: Number of top results to compare
        save_path: Optional path to save figure
    """
    num_methods = len(method_results)
    fig, axes = plt.subplots(1, num_methods, figsize=(5 * num_methods, 6))
    
    if num_methods == 1:
        axes = [axes]
    
    for idx, (method_name, results) in enumerate(method_results.items()):
        ax = axes[idx]
        
        # Get top-k
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        doc_indices = [r[0] for r in sorted_results]
        scores = [r[1] for r in sorted_results]
        
        ax.barh(range(len(scores)), scores, color=f'C{idx}')
        ax.set_yticks(range(len(doc_indices)))
        ax.set_yticklabels([f'Doc {idx}' for idx in doc_indices])
        ax.set_xlabel('Score')
        ax.set_title(method_name)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'Retrieval Method Comparison (Top-{top_k})', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_similarity_heatmap(similarity_matrix: np.ndarray,
                            query_labels: List[str] = None,
                            doc_labels: List[str] = None,
                            save_path: Optional[str] = None):
    """
    Plot similarity matrix as heatmap.
    
    Args:
        similarity_matrix: Matrix of similarity scores (num_queries x num_docs)
        query_labels: Optional labels for queries
        doc_labels: Optional labels for documents
        save_path: Optional path to save figure
    """
    num_queries, num_docs = similarity_matrix.shape
    
    if query_labels is None:
        query_labels = [f'Query {i}' for i in range(num_queries)]
    if doc_labels is None:
        doc_labels = [f'Doc {i}' for i in range(num_docs)]
    
    plt.figure(figsize=(max(10, num_docs * 0.5), max(6, num_queries * 0.5)))
    sns.heatmap(similarity_matrix,
                xticklabels=doc_labels,
                yticklabels=query_labels,
                cmap='YlOrRd',
                cbar_kws={'label': 'Similarity Score'},
                annot=False)
    plt.title('Query-Document Similarity Heatmap')
    plt.xlabel('Documents')
    plt.ylabel('Queries')
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_retrieval_summary(results_dict: Dict,
                          save_path: Optional[str] = None):
    """
    Wrapper function to generate comprehensive retrieval visualizations.
    
    Args:
        results_dict: Dictionary containing:
            - 'methods': Dict mapping method_name -> list of (doc_idx, score) tuples
            - 'similarity_matrix': Optional similarity matrix (num_queries x num_docs)
            - 'query_labels': Optional query labels
            - 'doc_labels': Optional document labels
            - 'top_k': Number of top results to display (default: 10)
        save_path: Optional directory path to save all figures
    """
    methods = results_dict.get('methods', {})
    top_k = results_dict.get('top_k', 10)
    
    import os
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # Plot individual method scores
    for method_name, results in methods.items():
        # Save individual method plots with proper naming
        method_save_path = f"{save_path}/{method_name}_scores.png" if save_path else None
        plot_top_k_scores(
            results, top_k=top_k, method_name=method_name,
            save_path=method_save_path
        )
    
    # Compare methods
    if len(methods) > 1:
        compare_retrieval_methods(
            methods, top_k=top_k,
            save_path=f"{save_path}/method_comparison.png" if save_path else None
        )
    
    # Plot similarity heatmap if available
    if 'similarity_matrix' in results_dict:
        plot_similarity_heatmap(
            results_dict['similarity_matrix'],
            query_labels=results_dict.get('query_labels'),
            doc_labels=results_dict.get('doc_labels'),
            save_path=f"{save_path}/similarity_heatmap.png" if save_path else None
        )


if __name__ == "__main__":
    import argparse
    import json
    import sys
    
    parser = argparse.ArgumentParser(description='Generate Retrieval Visualizations')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots')
    parser.add_argument('--results_path', type=str, default='data/processed/retrieval_results.json', help='Path to retrieval results json')
    
    args = parser.parse_args()
    
    print(f"Generating retrieval visualizations in {args.output_dir}...")
    
    try:
        # Load Results
        with open(args.results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Adapt structure if needed
        viz_data = data
        if 'methods' not in data and 'tfidf_results' in data:
            viz_data = {
                'methods': {
                    'TF-IDF': data.get('tfidf_results', []),
                    'LDA': data.get('lda_results', [])
                },
                'query': data.get('query', ''),
                'top_k': 10
            }
            
        plot_retrieval_summary(viz_data, save_path=args.output_dir)
        print("  Saved retrieval visualizations.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


