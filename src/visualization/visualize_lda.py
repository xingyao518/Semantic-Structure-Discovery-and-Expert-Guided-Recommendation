"""
Visualization functions for Latent Dirichlet Allocation (LDA) models.

This module provides visualization tools for both Gibbs sampling and
Variational Inference implementations of LDA.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from wordcloud import WordCloud
import seaborn as sns
import os

def plot_topic_word_distributions(phi: np.ndarray,
                                 vocab,
                                 top_n: int = 10,
                                 num_topics: int = None,
                                 save_path: Optional[str] = None):
    """
    Plot top words for each topic as bar charts.
    """
    K, V = phi.shape
    if num_topics is None:
        num_topics = K
    
    # Convert vocab to id->word mapping if needed
    if vocab and isinstance(vocab, dict):
        first_key = next(iter(vocab.keys()))
        if isinstance(first_key, str):
            id_to_word = {v: k for k, v in vocab.items()}
        else:
            id_to_word = vocab
    else:
        id_to_word = {}
    
    # Create subplots
    cols = 3
    rows = (num_topics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_topics > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for k in range(min(num_topics, K)):
        ax = axes[k]
        
        # Get top words for topic k
        top_indices = np.argsort(phi[k, :])[::-1][:top_n]
        top_words = [id_to_word.get(idx, f"word_{idx}") for idx in top_indices]
        top_probs = phi[k, top_indices]
        
        # Bar plot
        ax.barh(range(len(top_words)), top_probs)
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words)
        ax.set_xlabel('Probability')
        ax.set_title(f'Topic {k}')
        ax.invert_yaxis()
    
    # Hide unused subplots
    for k in range(num_topics, len(axes)):
        axes[k].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_topic_wordclouds(phi: np.ndarray,
                         vocab,
                         num_topics: int = None,
                         save_path: Optional[str] = None):
    """
    Generate word clouds for each topic.
    """
    K, V = phi.shape
    if num_topics is None:
        num_topics = K
    
    # Convert vocab to id->word mapping if needed
    if vocab and isinstance(vocab, dict):
        first_key = next(iter(vocab.keys()))
        if isinstance(first_key, str):
            id_to_word = {v: k for k, v in vocab.items()}
        else:
            id_to_word = vocab
    else:
        id_to_word = {}
    
    # Create word frequency dictionaries for each topic
    for k in range(min(num_topics, K)):
        word_freq = {}
        for v in range(V):
            word = id_to_word.get(v, f"word_{v}")
            if phi[k, v] > 0:
                word_freq[word] = phi[k, v]
        
        if not word_freq:
            continue
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white',
                            max_words=50).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {k} Word Cloud', fontsize=16)
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/topic_{k}_wordcloud.png", 
                       dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def plot_document_topic_distributions(theta: np.ndarray,
                                     num_docs: int = 20,
                                     topic_labels: Optional[List[str]] = None,
                                     save_path: Optional[str] = None):
    """
    Plot document-topic distributions as heatmap.
    """
    D, K = theta.shape
    num_docs = min(num_docs, D)
    
    # Select subset of documents
    theta_subset = theta[:num_docs, :]
    
    # Generate labels
    try:
        import json
        with open("data/processed/processed_data.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        doc_labels = []
        if topic_labels is None:
            topic_labels = [f"Topic {j}" for j in range(theta.shape[1])]
        
        if isinstance(data, list):
            # List of dicts
            for i in range(min(len(data), num_docs)):
                # Try shortest meaningful field
                d = data[i]
                title = d.get('title') or d.get('question') or d.get('text') or f'Doc {i}'
                # Truncate if too long (40 chars)
                if len(title) > 40: title = title[:37] + "..."
                doc_labels.append(title)
        elif isinstance(data, dict):
            # Dict with metadata (patch format)
            full_docs = data.get("doc_titles", [])
            doc_labels = full_docs[:num_docs] if full_docs else [f"Doc {i}" for i in range(num_docs)]
            # If topic_labels were passed, use them, otherwise check data
            if "topic_names" in data and topic_labels is None:
                topic_labels = data["topic_names"]
        
        # Ensure lengths match
        if len(doc_labels) < num_docs:
            doc_labels.extend([f"Doc {i}" for i in range(len(doc_labels), num_docs)])
            
    except Exception as e:
        print(f"Warning loading labels: {e}")
        doc_labels = [f"Doc {d}" for d in range(num_docs)]
        if topic_labels is None:
            topic_labels = [f"Topic {k}" for k in range(theta.shape[1])]

    # Create heatmap
    plt.figure(figsize=(12, max(6, num_docs * 0.5)))
    sns.heatmap(theta_subset, 
                cmap='YlOrRd',
                xticklabels=topic_labels,
                yticklabels=doc_labels,
                cbar_kws={'label': 'Topic Probability'})
    plt.title('Document-Topic Distributions')
    plt.xlabel('Topics')
    plt.ylabel('Documents')
    # Rotate topic labels if they are long
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_gibbs_convergence(log_likelihoods: List[float],
                          burn_in: int = 0,
                          save_path: Optional[str] = None):
    """
    Plot Gibbs sampling convergence (log-likelihood over iterations).
    """
    iterations = range(len(log_likelihoods))
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, log_likelihoods, 'b-', linewidth=1.5, label='Log-likelihood')
    
    if burn_in > 0 and burn_in < len(log_likelihoods):
        plt.axvline(x=burn_in, color='r', linestyle='--', 
                   label=f'Burn-in ({burn_in} iterations)')
    
    plt.xlabel('Iteration')
    plt.ylabel('Log-likelihood')
    plt.title('Gibbs Sampling Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ensure x-axis shows full range
    plt.xlim(0, len(log_likelihoods) - 1)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_vi_convergence(elbo_values: List[float],
                        save_path: Optional[str] = None):
    """
    Plot Variational Inference convergence (ELBO over iterations).
    """
    iterations = range(len(elbo_values))
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, elbo_values, 'g-', linewidth=1.5, label='ELBO')
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.title('Variational Inference Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_top_words(phi: np.ndarray,
                   vocab,
                   top_n: int = 10,
                   save_path: Optional[str] = None):
    """
    Plot top words for each topic (wrapper for plot_topic_word_distributions).
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # Convert vocab to id->word mapping if needed
    if vocab and isinstance(vocab, dict):
        # Check if vocab is word->id or id->word
        first_key = next(iter(vocab.keys()))
        if isinstance(first_key, str):
            # vocab is word->id, need to invert it
            id_to_word = {v: k for k, v in vocab.items()}
        else:
            # vocab is id->word
            id_to_word = vocab
    else:
        id_to_word = {}
    
    K = phi.shape[0]
    for k in range(K):
        # Get top words for topic k
        top_indices = np.argsort(phi[k, :])[::-1][:top_n]
        # Map indices to actual words
        top_words = [id_to_word.get(idx, f"word_{idx}") for idx in top_indices]
        top_probs = phi[k, top_indices]
        
        # Plot single topic
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_words)), top_probs, color='steelblue', alpha=0.7)
        plt.yticks(range(len(top_words)), top_words)
        plt.xlabel('Probability')
        plt.title(f'Top {top_n} Words for Topic {k}')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/lda_top_words_topic{k}.png", 
                       dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def plot_topic_distributions(theta: np.ndarray,
                            save_path: Optional[str] = None):
    """
    Plot topic distribution statistics across documents.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Average topic distribution across all documents
    avg_topic_dist = theta.mean(axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(avg_topic_dist)), avg_topic_dist, 
           color='coral', alpha=0.7)
    plt.xlabel('Topic')
    plt.ylabel('Average Probability')
    plt.title('Average Topic Distribution Across All Documents')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/lda_topic_distribution.png", 
                   dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_document_topic_heatmap(theta: np.ndarray,
                               num_docs: int = 50,
                               save_path: Optional[str] = None):
    """
    Plot document-topic heatmap.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plot_document_topic_distributions(theta, num_docs=num_docs, 
                                     save_path=f"{save_path}/lda_doc_topic_heatmap.png" if save_path else None)


def visualize_all_topics(model_output: Dict,
                         save_path: Optional[str] = None):
    """
    Wrapper function to generate all LDA visualizations.
    """
    phi = model_output['phi']
    theta = model_output['theta']
    vocab = model_output['vocab']
    model_type = model_output.get('model_type', 'gibbs')
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # Construct Topic Labels from Top Words
    topic_labels = []
    if vocab and isinstance(vocab, dict):
        try:
            # Handle word->id or id->word
            first_key = next(iter(vocab.keys()))
            if isinstance(first_key, str):
                id_to_word = {v: k for k, v in vocab.items()}
            else:
                id_to_word = vocab
            
            for k in range(phi.shape[0]):
                top_indices = np.argsort(phi[k, :])[::-1][:3] # Top 3 words
                words = [id_to_word.get(idx, f"w{idx}") for idx in top_indices]
                topic_labels.append(f"T{k}: {', '.join(words)}")
        except Exception:
            topic_labels = [f"Topic {k}" for k in range(phi.shape[0])]
    else:
        topic_labels = [f"Topic {k}" for k in range(phi.shape[0])]

    # Plot topic word distributions
    plot_topic_word_distributions(
        phi, vocab, top_n=10, 
        save_path=f"{save_path}/topic_words.png" if save_path else None
    )
    
    # Plot word clouds
    plot_topic_wordclouds(
        phi, vocab,
        save_path=save_path
    )
    
    # Plot document-topic distributions
    # Pass constructed topic_labels
    plot_document_topic_distributions(
        theta, num_docs=20, topic_labels=topic_labels,
        save_path=f"{save_path}/doc_topic_heatmap.png" if save_path else None
    )
    
    # Check for Gibbs convergence history from metrics if missing
    if model_type == 'gibbs':
        loglik = model_output.get('loglik_history')
        if not loglik:
            try:
                import json
                metrics_path = "data/processed/metrics/lda_gibbs_metrics.json"
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                        # Check possible keys
                        for key in ['log_likelihoods', 'log_lik_history', 'gibbs_loglik', 'log_likelihood_history']:
                            if key in metrics and isinstance(metrics[key], list):
                                loglik = metrics[key]
                                break
            except Exception as e:
                print(f"Warning loading metrics: {e}")
        
        if loglik:
            plot_gibbs_convergence(
                loglik,
                burn_in=model_output.get('burn_in', 0),
                save_path=f"{save_path}/gibbs_convergence.png" if save_path else None
            )
    elif model_type == 'vi' and 'elbo_history' in model_output:
        plot_vi_convergence(
            model_output['elbo_history'],
            save_path=f"{save_path}/vi_convergence.png" if save_path else None
        )

if __name__ == "__main__":
    import argparse
    import json
    import sys
    
    parser = argparse.ArgumentParser(description='Generate LDA Visualizations')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots')
    parser.add_argument('--model_path', type=str, default='data/processed/lda_gibbs_model.json', help='Path to LDA model json')
    parser.add_argument('--skip_gibbs', type=bool, default=False, help='Skip Gibbs convergence plot')
    
    args = parser.parse_args()
    
    print(f"Generating LDA visualizations in {args.output_dir}...")
    
    try:
        # Load Model
        with open(args.model_path, 'r', encoding='utf-8') as f:
            lda_data = json.load(f)
            
        # Reconstruct numpy arrays
        phi = np.array(lda_data['phi'])
        theta = np.array(lda_data['theta'])
        vocab = lda_data.get('vocab', {})
        # If vocab is stored as string keys (json), convert to int if possible or keep as is
        # My plot functions handle dict.
        
        # Construct model output dict
        model_output = {
            'phi': phi,
            'theta': theta,
            'vocab': vocab,
            'model_type': lda_data.get('model_type', 'gibbs'),
            'loglik_history': lda_data.get('loglik_history', [])
        }
        
        if args.skip_gibbs:
            # Remove history to skip plot
            if 'loglik_history' in model_output:
                del model_output['loglik_history']
                
        visualize_all_topics(model_output, save_path=args.output_dir)
        print("  Saved LDA visualizations.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


