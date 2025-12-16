"""
Safe regeneration of missing visualizations only.
"""
import sys
import os
import json
import pickle
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add root to path
ROOT = os.path.abspath(".")
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))

# Import visualization modules (try root first)
try:
    from visualization.visualize_lda import visualize_all_topics
    from visualization.visualize_logistic import plot_logistic_diagnostics, plot_coefficients
    from visualization.visualize_mixture import visualize_mixture_results
    from visualization.visualize_retrieval import plot_retrieval_summary
except ImportError:
    # Try src prefix
    from src.visualization.visualize_lda import visualize_all_topics
    from src.visualization.visualize_logistic import plot_logistic_diagnostics, plot_coefficients
    from src.visualization.visualize_mixture import visualize_mixture_results
    from src.visualization.visualize_retrieval import plot_retrieval_summary

base = Path("data/processed/visualizations")

expected_files = {
    "retrieval": [
        "LDA-topic_scores.png",
        "method_comparison.png",
        "method_comparison_precision.png",
        "similarity_heatmap.png",
        "TF-IDF_scores.png",
        "tfidf_topk.png",
    ],
    "lda": [
        "gibbs_convergence.png",
        "doc_topic_heatmap.png", # Mapped from lda_topic_distribution
        "topic_words.png",
        # Wordclouds are generated dynamically, check at least one
        "topic_0_wordcloud.png"
    ],
    "logistic": [
        "logistic_coefficients.png",
        "confusion_matrix.png", 
        "calibration_curve.png",
    ],
    "moe": [
        "moe_loglik_fixed.png",
        "moe_responsibilities_fixed.png",
        "mixture_clusters.png"
    ]
}

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_pickle(path):
    if not os.path.exists(path): return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    # ---- STEP 1: Create directories if missing ----
    for module in expected_files.keys():
        folder = base / module
        folder.mkdir(parents=True, exist_ok=True)

    # ---- STEP 2: Identify missing categories ----
    missing_cats = []
    for module, files in expected_files.items():
        folder = base / module
        # If any expected file is missing, mark category for regeneration
        is_missing = False
        for f in files:
            if not (folder / f).exists():
                is_missing = True
                print(f"Missing: {module}/{f}")
                break
        if is_missing:
            missing_cats.append(module)

    print(f"\nCategories to regenerate: {missing_cats}")

    # ---- STEP 3: Regenerate ----

    if "lda" in missing_cats:
        print("\nRegenerating LDA visualizations...")
        path = "data/processed/lda_gibbs_model.json"
        lda_data = load_json(path)
        if lda_data:
            # Convert to numpy
            phi = np.array(lda_data['phi'])
            theta = np.array(lda_data['theta'])
            vocab = None
            if 'vocab' in lda_data:
                 vocab = {int(k): v for k, v in lda_data['vocab'].items()}
            
            viz_data = {
                'phi': phi,
                'theta': theta,
                'vocab': vocab,
                'loglik_history': lda_data.get('loglik_history', []),
                'model_type': 'gibbs'
            }
            try:
                visualize_all_topics(viz_data, save_path=str(base / "lda"))
                print("LDA Done.")
            except Exception as e:
                print(f"Error LDA: {e}")

    if "logistic" in missing_cats:
        print("\nRegenerating Logistic Regression visualizations...")
        path = "data/processed/logistic_model.pkl"
        model = load_pickle(path)
        if model and hasattr(model, 'weights'):
             save_dir = base / "logistic"
             save_path = save_dir / "logistic_coefficients.png"
             try:
                 plot_coefficients(model.weights, save_path=str(save_path))
                 # Cannot generate confusion matrix without test data
                 print("Logistic Coefficients Done.")
             except Exception as e:
                 print(f"Error Logistic: {e}")

    if "moe" in missing_cats:
        print("\nRegenerating MoE visualizations...")
        path = "data/processed/mixture_model.pkl"
        moe = load_pickle(path)
        if moe:
            save_path = base / "moe"
            try:
                if hasattr(moe, 'loglik_history') and moe.loglik_history:
                    plt.figure()
                    plt.plot(moe.loglik_history)
                    plt.title("MoE Log-Likelihood")
                    plt.savefig(save_path / "moe_loglik_fixed.png")
                    plt.close()
                
                if hasattr(moe, 'responsibilities') and moe.responsibilities is not None:
                     viz_data = {
                        'mixture_weights': moe.pi,
                        'responsibilities': moe.responsibilities[:50],
                        'expert_names': [f'Expert {i}' for i in range(moe.E)]
                    }
                     visualize_mixture_results(viz_data, save_path=str(save_path))
                print("MoE Done.")
            except Exception as e:
                print(f"Error MoE: {e}")

    if "retrieval" in missing_cats:
        print("\nRegenerating Retrieval visualizations...")
        path = "data/processed/retrieval_results.json"
        ret_data = load_json(path)
        if ret_data:
            save_path = base / "retrieval"
            
            # Adapt structure if needed
            viz_ret_data = ret_data
            if 'methods' not in ret_data and 'tfidf_results' in ret_data:
                viz_ret_data = {
                    'methods': {
                        'TF-IDF': ret_data.get('tfidf_results', []),
                        'LDA': ret_data.get('lda_results', [])
                    },
                    'query': ret_data.get('query', ''),
                    'top_k': 10
                }
                
            try:
                plot_retrieval_summary(viz_ret_data, save_path=str(save_path))
                print("Retrieval Done.")
            except Exception as e:
                print(f"Error Retrieval: {e}")

    print("\n=== SAFE REGENERATION COMPLETE ===")

if __name__ == "__main__":
    main()

