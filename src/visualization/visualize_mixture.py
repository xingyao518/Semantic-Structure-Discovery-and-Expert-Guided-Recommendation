"""
Visualization functions for Bayesian Mixture-of-Experts model.

This module provides visualization tools for inspecting mixture component
weights, expert selection, and posterior responsibilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import seaborn as sns

def plot_mixture_weights(mixture_weights: np.ndarray,
                        expert_names: List[str] = None,
                        save_path: Optional[str] = None):
    """
    Plot mixture component weights as bar chart.
    
    Args:
        mixture_weights: Mixture weights array (E,) where E is number of experts
        expert_names: Optional names for experts (default: Expert 0, Expert 1, ...)
        save_path: Optional path to save figure
    """
    E = len(mixture_weights)
    if expert_names is None:
        expert_names = [f'Expert {i}' for i in range(E)]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(E), mixture_weights, color='steelblue', alpha=0.7)
    plt.xticks(range(E), expert_names, rotation=45, ha='right')
    plt.ylabel('Mixture Weight')
    plt.xlabel('Expert')
    plt.title('Mixture-of-Experts Component Weights')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, weight) in enumerate(zip(bars, mixture_weights)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{weight:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_advice_template_selection(expert_assignments: List[int],
                                   num_experts: int = None,
                                   save_path: Optional[str] = None):
    """
    Visualize which expert templates were selected for each query/advice.
    
    Args:
        expert_assignments: List of expert indices assigned to each query
        num_experts: Total number of experts (if None, inferred from assignments)
        save_path: Optional path to save figure
    """
    if num_experts is None:
        num_experts = max(expert_assignments) + 1 if expert_assignments else 1
    
    # Count assignments
    assignment_counts = np.zeros(num_experts)
    for assignment in expert_assignments:
        if 0 <= assignment < num_experts:
            assignment_counts[assignment] += 1
    
    expert_names = [f'Expert {i}' for i in range(num_experts)]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(num_experts), assignment_counts, 
                   color='forestgreen', alpha=0.7)
    plt.xticks(range(num_experts), expert_names, rotation=45, ha='right')
    plt.ylabel('Number of Selections')
    plt.xlabel('Expert')
    plt.title('Advice Template Selection Frequency')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, assignment_counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{int(count)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_posterior_responsibilities(responsibilities: np.ndarray,
                                   query_labels: List[str] = None,
                                   expert_names: List[str] = None,
                                   save_path: Optional[str] = None):
    """
    Plot posterior responsibilities (mixture weights) as heatmap.
    
    Args:
        responsibilities: Responsibility matrix (N x E) where responsibilities[i, e]
            is the posterior probability of expert e for query i
        query_labels: Optional labels for queries
        expert_names: Optional names for experts
        save_path: Optional path to save figure
    """
    N, E = responsibilities.shape
    
    if query_labels is None:
        query_labels = [f'Query {i}' for i in range(N)]
    if expert_names is None:
        expert_names = [f'Expert {i}' for i in range(E)]
    
    plt.figure(figsize=(max(10, E * 1.5), max(6, N * 0.3)))
    sns.heatmap(responsibilities,
                xticklabels=expert_names,
                yticklabels=query_labels,
                cmap='YlOrRd',
                cbar_kws={'label': 'Posterior Responsibility'},
                annot=True, fmt='.2f')
    plt.title('Posterior Responsibilities (Mixture Weights)')
    plt.xlabel('Experts')
    plt.ylabel('Queries')
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_cluster_assignments(cluster_labels: np.ndarray,
                             save_path: Optional[str] = None):
    """
    Plot cluster assignment distribution.
    
    Args:
        cluster_labels: Array of cluster assignments (N,)
        save_path: Optional path to save figure
    """
    import os
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique_labels, counts, color='steelblue', alpha=0.7)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Assignments')
    plt.title('Cluster Assignment Distribution')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.01,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/mixture_clusters.png", 
                   dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_cluster_centroids(centroids: np.ndarray,
                          feature_names: List[str] = None,
                          save_path: Optional[str] = None):
    """
    Plot cluster centroids as heatmap.
    
    Args:
        centroids: Centroid matrix (K x D) where K is num clusters, D is num features
        feature_names: Optional feature names
        save_path: Optional path to save figure
    """
    import os
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    K, D = centroids.shape
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(D)]
    
    plt.figure(figsize=(max(10, D * 0.8), max(6, K * 0.8)))
    sns.heatmap(centroids,
                xticklabels=feature_names,
                yticklabels=[f'Cluster {i}' for i in range(K)],
                cmap='RdYlBu_r',
                center=0,
                cbar_kws={'label': 'Centroid Value'},
                annot=True, fmt='.2f')
    plt.title('Cluster Centroids')
    plt.xlabel('Features')
    plt.ylabel('Clusters')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/mixture_centroids.png", 
                   dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def visualize_mixture_results(mixture_output: Dict,
                             save_path: Optional[str] = None):
    """
    Wrapper function to generate all mixture-of-experts visualizations.
    
    Args:
        mixture_output: Dictionary containing:
            - 'mixture_weights': Array of mixture weights (E,) or (N x E) for multiple queries
            - 'expert_assignments': Optional list of expert indices assigned to queries
            - 'responsibilities': Optional responsibility matrix (N x E)
            - 'expert_names': Optional expert names
            - 'query_labels': Optional query labels
        save_path: Optional directory path to save all figures
    """
    import os
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    mixture_weights = mixture_output.get('mixture_weights')
    expert_assignments = mixture_output.get('expert_assignments')
    responsibilities = mixture_output.get('responsibilities')
    expert_names = mixture_output.get('expert_names')
    query_labels = mixture_output.get('query_labels')
    
    # Plot mixture weights
    if mixture_weights is not None:
        # Handle both single vector and matrix
        if mixture_weights.ndim == 1:
            plot_mixture_weights(
                mixture_weights, expert_names=expert_names,
                save_path=f"{save_path}/mixture_weights.png" if save_path else None
            )
        elif mixture_weights.ndim == 2:
            # Plot average weights
            avg_weights = mixture_weights.mean(axis=0)
            plot_mixture_weights(
                avg_weights, expert_names=expert_names,
                save_path=f"{save_path}/avg_mixture_weights.png" if save_path else None
            )
    
    # Plot expert assignments
    if expert_assignments is not None:
        num_experts = len(expert_names) if expert_names else None
        plot_advice_template_selection(
            expert_assignments, num_experts=num_experts,
            save_path=f"{save_path}/expert_selection.png" if save_path else None
        )
    
    # Plot responsibilities
    if responsibilities is not None:
        plot_posterior_responsibilities(
            responsibilities, query_labels=query_labels,
            expert_names=expert_names,
            save_path=f"{save_path}/responsibilities.png" if save_path else None
        )


if __name__ == "__main__":
    import argparse
    import pickle
    import sys
    import os
    # Add src to path to allow unpickling legacy models
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    
    parser = argparse.ArgumentParser(description='Generate Mixture-of-Experts Visualizations')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots')
    parser.add_argument('--model_path', type=str, default='data/processed/mixture_model.pkl', help='Path to MoE model pickle')
    
    args = parser.parse_args()
    
    print(f"Generating MoE visualizations in {args.output_dir}...")
    
    try:
        # Load Model
        with open(args.model_path, 'rb') as f:
            moe = pickle.load(f)
            
        viz_data = {}
        
        # Extract data
        if hasattr(moe, 'pi'):
            viz_data['mixture_weights'] = moe.pi
        
        if hasattr(moe, 'responsibilities') and moe.responsibilities is not None:
            # Visualize subset if too large
            viz_data['responsibilities'] = moe.responsibilities[:50]
        
        if hasattr(moe, 'E'):
             viz_data['expert_names'] = [f'Expert {i}' for i in range(moe.E)]
             
        # Call visualization wrapper
        visualize_mixture_results(viz_data, save_path=args.output_dir)
        print("  Saved MoE visualizations.")
        
        # Plot Log-Likelihood if available
        if hasattr(moe, 'loglik_history') and moe.loglik_history:
             plt.figure()
             plt.plot(moe.loglik_history)
             plt.title("MoE EM Log-Likelihood")
             plt.xlabel("Iteration")
             plt.ylabel("Log Likelihood")
             os.makedirs(args.output_dir, exist_ok=True)
             plt.savefig(os.path.join(args.output_dir, "moe_loglik_fixed.png"))
             plt.close()
             print("  Saved moe_loglik_fixed.png")
             
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


