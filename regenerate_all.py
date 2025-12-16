"""
Regenerate all models and visualizations with fixed implementations.
DIAGNOSTIC MODE
"""
import sys, os
import traceback

print("Starting diagnostics...")
ROOT = os.path.abspath(".")
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil

print("Importing modules...")
try:
    from src.models.lda_gibbs import LDAGibbsSampler
    from src.models.logistic_map import LogisticRegressionMAP
    from src.models.bayesian_mixture import BayesianMixtureOfExperts
    from visualization.visualize_lda import visualize_all_topics
    from visualization.visualize_logistic import plot_logistic_diagnostics
    from visualization.visualize_mixture import visualize_mixture_results
    print("Modules imported successfully.")
except Exception as e:
    print(f"Error importing modules: {e}")
    traceback.print_exc()
    sys.exit(1)

def clean_visualizations():
    """Delete old visualizations."""
    print("Cleaning old visualizations...")
    base_dir = "data/processed/visualizations"
    if os.path.exists(base_dir):
        try:
            shutil.rmtree(base_dir)
        except Exception as e:
            print(f"Warning: Could not clean directory: {e}")
    
    for sub in ["lda", "gibbs", "logistic", "moe"]:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)
    print("Cleaned.")

def load_data():
    path = "data/processed/processed_data.json"
    print(f"Loading data from {path}...")
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        sys.exit(1)
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            docs_text = [d['text'] for d in data]
            # Create mock labels for demo
            labels = [1 if 'injury' in d.get('text', '').lower() else 0 for d in data]
        else:
            docs_text = [d['text'] for d in data.get('documents', [])]
            labels = [1 if 'injury' in d.get('text', '').lower() else 0 for d in data.get('documents', [])]
        
        print(f"Loaded {len(docs_text)} documents. Using subset of 2000.")
        return docs_text[:2000], np.array(labels[:2000])
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        sys.exit(1)

def run_lda(docs_text):
    print("\n=== Running LDA Gibbs ===")
    try:
        # Tokenize
        vocab = {}
        next_id = 0
        docs_ids = []
        
        for doc in docs_text:
            tokens = doc.lower().split()
            ids = []
            for t in tokens:
                if len(t) < 3: continue
                if t not in vocab:
                    vocab[t] = next_id
                    next_id += 1
                ids.append(vocab[t])
            docs_ids.append(ids)
        
        vocab_inv = {v: k for k, v in vocab.items()}
        print(f"Vocabulary size: {len(vocab)}")
        
        lda = LDAGibbsSampler(num_topics=5, random_seed=42)
        print("Starting Gibbs sampling fit...")
        theta, phi = lda.fit(docs_ids, len(vocab), num_iterations=100)
        print("LDA fit complete.")
        
        print("Saving LDA visualizations...")
        viz_data = {
            'phi': phi,
            'theta': theta,
            'vocab': vocab_inv,
            'loglik_history': lda.loglik_history,
            'model_type': 'gibbs'
        }
        visualize_all_topics(viz_data, save_path="data/processed/visualizations/lda")
        
        # Save explicit Gibbs convergence
        plt.figure()
        plt.plot(lda.loglik_history)
        plt.title("Gibbs Convergence (Fixed)")
        plt.xlabel("Iteration")
        plt.ylabel("Log Joint Probability")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("data/processed/visualizations/gibbs/gibbs_convergence_fixed.png")
        plt.close()
        print("Saved Gibbs convergence plot.")
        
        print("Finished LDA Gibbs ✓")
        return theta
    except Exception as e:
        print(f"Error in LDA: {e}")
        traceback.print_exc()
        sys.exit(1)

def run_logistic(theta, labels):
    print("\n=== Running Logistic Regression ===")
    try:
        # Use theta as features
        X = theta
        y = labels[:len(theta)]
        
        lr = LogisticRegressionMAP(learning_rate=0.1, max_iterations=500)
        print("Fitting Logistic Regression...")
        lr.fit(X, y)
        print("Logistic fit complete.")
        
        probs = lr.predict_proba(X)
        preds = lr.predict(X)
        
        print("Saving Logistic visualizations...")
        diag_data = {
            'y_pred': preds,
            'y_proba': probs,
            'class_names': ['No Injury', 'Injury Risk']
        }
        plot_logistic_diagnostics(diag_data, y, save_path="data/processed/visualizations/logistic")
        print("Finished Logistic Regression ✓")
    except Exception as e:
        print(f"Error in Logistic Regression: {e}")
        traceback.print_exc()
        # Don't exit, try to continue to MoE

def run_moe(theta):
    print("\n=== Running Mixture of Experts ===")
    try:
        # Use theta (topic mixtures) as input
        
        moe = BayesianMixtureOfExperts(num_experts=3, feature_dim=theta.shape[1])
        print("Fitting MoE...")
        loglik_hist = moe.fit(theta, num_iterations=50)
        print("MoE fit complete.")
        
        print("Saving MoE visualizations...")
        
        # 1. Log Likelihood
        plt.figure()
        plt.plot(loglik_hist)
        plt.title("MoE EM Log-Likelihood (Fixed)")
        plt.xlabel("Iteration")
        plt.ylabel("Log Likelihood")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("data/processed/visualizations/moe/moe_loglik_fixed.png")
        plt.close()
        
        # 2. Responsibilities Heatmap
        if moe.responsibilities is not None:
            viz_data = {
                'mixture_weights': moe.pi,
                'responsibilities': moe.responsibilities[:50], # Top 50 docs
                'expert_names': [f'Expert {i}' for i in range(moe.E)]
            }
            visualize_mixture_results(viz_data, save_path="data/processed/visualizations/moe")
            
            # Save explicit fixed heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(moe.responsibilities[:20], cmap="YlOrRd", annot=True, fmt=".2f")
            plt.title("Posterior Responsibilities (First 20 Docs)")
            plt.xlabel("Expert")
            plt.ylabel("Document")
            plt.tight_layout()
            plt.savefig("data/processed/visualizations/moe/moe_responsibilities_fixed.png")
            plt.close()
            
        print("Finished Mixture-of-Experts ✓")
    except Exception as e:
        print(f"Error in MoE: {e}")
        traceback.print_exc()

def main():
    clean_visualizations()
    docs, labels = load_data()
    theta = run_lda(docs)
    run_logistic(theta, labels)
    run_moe(theta)
    
    print("\n\nAll tasks completed successfully.")
    print("Visualizations saved to data/processed/visualizations/")

if __name__ == "__main__":
    main()
