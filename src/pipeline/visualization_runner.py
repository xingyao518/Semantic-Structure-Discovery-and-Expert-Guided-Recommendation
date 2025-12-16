"""
Visualization Runner Module - Generates all visualizations
"""

import sys
import os
import json
import pickle
import numpy as np
from typing import Dict, List, Optional

# Force non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'visualization'))


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def run_lda_visualizations(
    theta: np.ndarray,
    phi: np.ndarray,
    vocab: Dict[str, int],
    loglik_history: List[float],
    output_dir: str = "data/processed/visualizations/lda",
    model_type: str = "gibbs"
):
    """
    Generate LDA visualizations.
    
    Args:
        theta: Document-topic distributions (D x K)
        phi: Topic-word distributions (K x V)
        vocab: Word to ID mapping
        loglik_history: Training history
        output_dir: Output directory
        model_type: 'gibbs' or 'vi'
    """
    print("\n[Visualization] Generating LDA visualizations...")
    sys.stdout.flush()
    
    ensure_dir(output_dir)
    
    # Create id_to_word mapping
    id_to_word = {v: k for k, v in vocab.items()}
    
    # 1. Topic Words Bar Chart
    print("[Visualization] Creating topic words plot...")
    sys.stdout.flush()
    
    K = phi.shape[0]
    top_n = 10
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for k in range(min(K, 10)):
        ax = axes[k]
        top_indices = np.argsort(phi[k, :])[::-1][:top_n]
        words = [id_to_word.get(idx, f"w{idx}") for idx in top_indices]
        probs = phi[k, top_indices]
        
        ax.barh(range(len(words)), probs, color='steelblue')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel('Probability')
        ax.set_title(f'Topic {k}')
        ax.invert_yaxis()
    
    for k in range(K, 10):
        axes[k].axis('off')
    
    plt.tight_layout()
    save_path = f"{output_dir}/topic_words.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {save_path}")
    sys.stdout.flush()
    
    # 2. Document-Topic Heatmap
    print("[Visualization] Creating doc-topic heatmap...")
    sys.stdout.flush()
    
    num_docs = min(30, theta.shape[0])
    
    plt.figure(figsize=(12, 8))
    plt.imshow(theta[:num_docs, :], aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='Topic Probability')
    plt.xlabel('Topics')
    plt.ylabel('Documents')
    plt.title('Document-Topic Distributions')
    plt.tight_layout()
    save_path = f"{output_dir}/doc_topic_heatmap.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {save_path}")
    sys.stdout.flush()
    
    # 3. Convergence Plot
    if loglik_history:
        print("[Visualization] Creating convergence plot...")
        sys.stdout.flush()
        
        plt.figure(figsize=(10, 6))
        plt.plot(loglik_history, 'b-', linewidth=1.5)
        plt.xlabel('Iteration')
        ylabel = 'Log-Likelihood' if model_type == 'gibbs' else 'ELBO'
        plt.ylabel(ylabel)
        plt.title(f'LDA {model_type.upper()} Convergence')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        suffix = 'gibbs_convergence' if model_type == 'gibbs' else 'vi_convergence'
        save_path = f"{output_dir}/{suffix}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved {save_path}")
        sys.stdout.flush()
    
    # 4. Word Clouds (simplified - just top words as text)
    print("[Visualization] Creating word clouds...")
    sys.stdout.flush()
    
    try:
        from wordcloud import WordCloud
        
        for k in range(min(K, 10)):
            word_freq = {}
            for v in range(phi.shape[1]):
                word = id_to_word.get(v, f"w{v}")
                if phi[k, v] > 0.001:  # Filter low probability words
                    word_freq[word] = phi[k, v]
            
            if word_freq:
                wc = WordCloud(width=800, height=400, background_color='white', max_words=50)
                wc.generate_from_frequencies(word_freq)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Topic {k} Word Cloud')
                
                save_path = f"{output_dir}/topic_{k}_wordcloud.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[INFO] Saved {save_path}")
                sys.stdout.flush()
    except ImportError:
        print("[WARNING] wordcloud not installed, skipping word clouds")
        sys.stdout.flush()
    
    print("[INFO] LDA visualizations completed")
    sys.stdout.flush()


def run_mixture_visualizations(
    responsibilities: np.ndarray,
    loglik_history: List[float],
    mixture_weights: np.ndarray,
    output_dir: str = "data/processed/visualizations/mixture"
):
    """
    Generate mixture model visualizations.
    
    Args:
        responsibilities: Document-expert responsibilities (N x E)
        loglik_history: Training history
        mixture_weights: Mixture weights (E,)
        output_dir: Output directory
    """
    print("\n[Visualization] Generating Mixture visualizations...")
    sys.stdout.flush()
    
    ensure_dir(output_dir)
    
    # 1. Responsibilities Heatmap
    print("[Visualization] Creating responsibilities heatmap...")
    sys.stdout.flush()
    
    num_docs = min(50, responsibilities.shape[0]) if responsibilities is not None else 0
    if num_docs == 0:
        print("[WARNING] No responsibilities available for heatmap")
        sys.stdout.flush()
        return
    
    plt.figure(figsize=(10, 8))
    plt.imshow(responsibilities[:num_docs, :], aspect='auto', cmap='Blues')
    plt.colorbar(label='Responsibility')
    plt.xlabel('Expert')
    plt.ylabel('Document')
    plt.title('Document-Expert Responsibilities')
    plt.tight_layout()
    save_path = f"{output_dir}/responsibilities_heatmap.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {save_path}")
    sys.stdout.flush()
    
    # 2. Mixture Weights
    if mixture_weights is not None and getattr(mixture_weights, "size", 0) > 0:
        print("[Visualization] Creating mixture weights plot...")
        sys.stdout.flush()
        
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(mixture_weights)), mixture_weights, color='coral')
        plt.xlabel('Expert')
        plt.ylabel('Weight')
        plt.title('Mixture Weights (π)')
        plt.xticks(range(len(mixture_weights)))
        plt.tight_layout()
        save_path = f"{output_dir}/mixture_weights.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved {save_path}")
        sys.stdout.flush()
    else:
        print("[WARNING] Skipping mixture weights plot - no weights available")
        sys.stdout.flush()
    
    # 3. Convergence Plot
    if loglik_history:
        print("[Visualization] Creating EM convergence plot...")
        sys.stdout.flush()
        
        plt.figure(figsize=(10, 6))
        plt.plot(loglik_history, 'g-', linewidth=1.5)
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.title('Mixture Model EM Convergence')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = f"{output_dir}/em_convergence.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved {save_path}")
        sys.stdout.flush()
    
    # 4. Expert Assignment Distribution
    print("[Visualization] Creating expert assignment distribution...")
    sys.stdout.flush()
    
    assignments = np.argmax(responsibilities, axis=1)
    unique, counts = np.unique(assignments, return_counts=True)
    
    plt.figure(figsize=(8, 5))
    plt.bar(unique, counts, color='teal')
    plt.xlabel('Expert')
    plt.ylabel('Number of Documents')
    plt.title('Document Assignments by Expert')
    plt.xticks(unique)
    plt.tight_layout()
    save_path = f"{output_dir}/expert_assignments.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {save_path}")
    sys.stdout.flush()
    
    print("[INFO] Mixture visualizations completed")
    sys.stdout.flush()


def run_retrieval_visualizations(
    retrieval_results: Dict,
    output_dir: str = "data/processed/visualizations/retrieval"
):
    """
    Generate retrieval visualizations.
    
    Args:
        retrieval_results: Retrieval results dictionary
        output_dir: Output directory
    """
    print("\n[Visualization] Generating Retrieval visualizations...")
    sys.stdout.flush()
    
    ensure_dir(output_dir)
    
    # 1. Score Comparison
    print("[Visualization] Creating retrieval score comparison...")
    sys.stdout.flush()
    
    queries = retrieval_results.get('queries', [])
    tfidf_results = retrieval_results.get('tfidf_results', [])
    lda_results = retrieval_results.get('lda_results', [])
    
    if queries and tfidf_results:
        # Average scores per query
        tfidf_avg = []
        lda_avg = []
        
        for i in range(len(queries)):
            if i < len(tfidf_results) and tfidf_results[i]:
                scores = [r['score'] for r in tfidf_results[i]]
                tfidf_avg.append(np.mean(scores) if scores else 0)
            else:
                tfidf_avg.append(0)
            
            if i < len(lda_results) and lda_results[i]:
                scores = [r['score'] for r in lda_results[i]]
                lda_avg.append(np.mean(scores) if scores else 0)
            else:
                lda_avg.append(0)
        
        x = np.arange(len(queries))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, tfidf_avg, width, label='TF-IDF', color='steelblue')
        plt.bar(x + width/2, lda_avg, width, label='LDA', color='coral')
        plt.xlabel('Query')
        plt.ylabel('Average Score')
        plt.title('Retrieval Score Comparison')
        plt.xticks(x, [f'Q{i+1}' for i in range(len(queries))])
        plt.legend()
        plt.tight_layout()
        save_path = f"{output_dir}/retrieval_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved {save_path}")
        sys.stdout.flush()
    
    # 2. Top-k score distribution
    print("[Visualization] Creating score distribution plot...")
    sys.stdout.flush()
    
    all_tfidf_scores = []
    all_lda_scores = []
    
    for results in tfidf_results:
        if results:
            all_tfidf_scores.extend([r['score'] for r in results])
    
    for results in lda_results:
        if results:
            all_lda_scores.extend([r['score'] for r in results])
    
    if all_tfidf_scores or all_lda_scores:
        plt.figure(figsize=(10, 5))
        
        if all_tfidf_scores:
            plt.hist(all_tfidf_scores, bins=20, alpha=0.5, label='TF-IDF', color='steelblue')
        if all_lda_scores:
            plt.hist(all_lda_scores, bins=20, alpha=0.5, label='LDA', color='coral')
        
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Retrieval Score Distribution')
        plt.legend()
        plt.tight_layout()
        save_path = f"{output_dir}/score_distribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved {save_path}")
        sys.stdout.flush()
    
    print("[INFO] Retrieval visualizations completed")
    sys.stdout.flush()


def run_all_visualizations(
    theta_gibbs: np.ndarray = None,
    phi_gibbs: np.ndarray = None,
    vocab: Dict[str, int] = None,
    gibbs_history: List[float] = None,
    theta_vi: np.ndarray = None,
    phi_vi: np.ndarray = None,
    vi_history: List[float] = None,
    responsibilities: np.ndarray = None,
    mixture_history: List[float] = None,
    mixture_weights: np.ndarray = None,
    retrieval_results: Dict = None,
    base_output_dir: str = "data/processed/visualizations"
):
    """
    Run all visualizations.
    """
    print("\n" + "=" * 50)
    print("  VISUALIZATION GENERATION")
    print("=" * 50)
    sys.stdout.flush()
    
    # Ensure base output directory exists
    os.makedirs(base_output_dir, exist_ok=True)
    
    # LDA Gibbs visualizations
    if theta_gibbs is not None and phi_gibbs is not None and vocab is not None:
        run_lda_visualizations(
            theta_gibbs, phi_gibbs, vocab, gibbs_history or [],
            output_dir=f"{base_output_dir}/lda",
            model_type="gibbs"
        )
    
    # LDA VI visualizations (if different from Gibbs)
    if theta_vi is not None and phi_vi is not None and vocab is not None:
        run_lda_visualizations(
            theta_vi, phi_vi, vocab, vi_history or [],
            output_dir=f"{base_output_dir}/lda_vi",
            model_type="vi"
        )
    
    # Mixture visualizations
    # Ensure mixture_weights is valid and not causing ambiguous numpy truth errors
    if mixture_weights is None:
        mixture_weights = np.array([])
    
    if responsibilities is not None and getattr(responsibilities, "size", 0) > 0:
        run_mixture_visualizations(
            responsibilities,
            mixture_history or [],
            mixture_weights,
            output_dir=f"{base_output_dir}/mixture"
        )
    
    # Retrieval visualizations
    if retrieval_results is not None:
        run_retrieval_visualizations(
            retrieval_results,
            output_dir=f"{base_output_dir}/retrieval"
        )
    
    print("\n[INFO] All visualizations completed!")
    sys.stdout.flush()


if __name__ == "__main__":
    print("Visualization Runner - requires model outputs")

