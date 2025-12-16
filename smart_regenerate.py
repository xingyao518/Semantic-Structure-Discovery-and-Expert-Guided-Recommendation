"""
Regenerate MoE and Logistic visualizations using existing LDA theta.
Fast execution.
"""
import sys, os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.abspath(".")
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))

from src.models.bayesian_mixture import BayesianMixtureOfExperts
from src.models.logistic_map import LogisticRegressionMAP
from visualization.visualize_mixture import visualize_mixture_results
from visualization.visualize_logistic import plot_logistic_diagnostics

def load_theta():
    path = "data/processed/lda_gibbs_model.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return np.array(data['theta'])

def load_labels(num_samples):
    path = "data/processed/processed_data.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        labels = [1 if 'injury' in d.get('text', '').lower() else 0 for d in data]
    else:
        labels = [1 if 'injury' in d.get('text', '').lower() else 0 for d in data.get('documents', [])]
    
    return np.array(labels[:num_samples])

def run_moe(theta):
    print("Running MoE...")
    moe = BayesianMixtureOfExperts(num_experts=3, feature_dim=theta.shape[1])
    loglik_hist = moe.fit(theta, num_iterations=50)
    
    save_path = "data/processed/visualizations/moe"
    os.makedirs(save_path, exist_ok=True)
    
    # Log Likelihood
    plt.figure()
    plt.plot(loglik_hist)
    plt.title("MoE EM Log-Likelihood")
    plt.savefig(os.path.join(save_path, "moe_loglik_fixed.png"))
    plt.close()
    
    # Responsibilities
    viz_data = {
        'mixture_weights': moe.pi,
        'responsibilities': moe.responsibilities[:50],
        'expert_names': [f'Expert {i}' for i in range(moe.E)]
    }
    visualize_mixture_results(viz_data, save_path=save_path)
    print("MoE Done.")

def run_logistic(theta):
    print("Running Logistic...")
    labels = load_labels(len(theta))
    # Ensure lengths match (LDA might have dropped short docs or used subset)
    # If theta is from full run, labels should match.
    # If len(theta) != len(labels), truncate labels
    if len(labels) > len(theta):
        labels = labels[:len(theta)]
    elif len(labels) < len(theta):
        # Truncate theta
        theta = theta[:len(labels)]
    
    X = theta
    y = labels
    
    lr = LogisticRegressionMAP(learning_rate=0.1, max_iterations=200)
    lr.fit(X, y)
    
    probs = lr.predict_proba(X)
    preds = lr.predict(X)
    
    diag_data = {
        'y_pred': preds,
        'y_proba': probs,
        'class_names': ['No Injury', 'Injury Risk']
    }
    save_path = "data/processed/visualizations/logistic"
    os.makedirs(save_path, exist_ok=True)
    plot_logistic_diagnostics(diag_data, y, save_path=save_path)
    print("Logistic Done.")

def main():
    try:
        theta = load_theta()
        print(f"Loaded theta with shape {theta.shape}")
        
        run_moe(theta)
        run_logistic(theta)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


