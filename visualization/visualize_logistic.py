"""
Visualization functions for Logistic Regression with MAP estimation.

This module provides diagnostic plots for evaluating logistic regression
model performance and calibration.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, precision_recall_curve
import seaborn as sns
import os

def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: List[str] = None,
                          save_path: Optional[str] = None):
    """
    Plot confusion matrix for binary classification.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = ['Negative', 'Positive']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_precision_recall_f1(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_proba: np.ndarray = None,
                             save_path: Optional[str] = None):
    """
    Plot precision, recall, F1 summary and precision-recall curve.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot of metrics
    metrics = ['Precision', 'Recall', 'F1']
    values = [precision, recall, f1]
    colors = ['steelblue', 'forestgreen', 'coral']
    
    axes[0].bar(metrics, values, color=colors, alpha=0.7)
    axes[0].set_ylim([0, 1.1]) # Slight buffer for text
    axes[0].set_ylabel('Score')
    axes[0].set_title('Classification Metrics Summary')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (metric, value) in enumerate(zip(metrics, values)):
        axes[0].text(i, value + 0.02, f'{value:.3f}', 
                    ha='center', va='bottom')
    
    # Precision-Recall curve (if probabilities available)
    if y_proba is not None:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
        axes[1].plot(recall_curve, precision_curve, 'b-', linewidth=2)
        axes[1].axhline(y=precision, color='r', linestyle='--', 
                       label=f'Precision={precision:.3f}')
        axes[1].axvline(x=recall, color='g', linestyle='--', 
                       label=f'Recall={recall:.3f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Probabilities not available\nfor PR curve',
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Precision-Recall Curve')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_calibration_curve(y_true: np.ndarray,
                           y_proba: np.ndarray,
                           n_bins: int = 10,
                           save_path: Optional[str] = None):
    """
    Plot calibration curve (reliability diagram).
    """
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 
            's-', label='Model', linewidth=2, markersize=8)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_coefficients(coeff_vector: np.ndarray,
                     feature_names: List[str] = None,
                     save_path: Optional[str] = None):
    """
    Plot logistic regression coefficients.
    """
    D = len(coeff_vector)
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(D)]
    
    # Sort by absolute value
    sorted_indices = np.argsort(np.abs(coeff_vector))[::-1]
    sorted_coeffs = coeff_vector[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    # Limit to top features if too many
    if len(sorted_coeffs) > 20:
        sorted_coeffs = sorted_coeffs[:20]
        sorted_names = sorted_names[:20]
    
    plt.figure(figsize=(10, max(6, len(sorted_coeffs) * 0.3)))
    colors = ['red' if c < 0 else 'blue' for c in sorted_coeffs]
    plt.barh(range(len(sorted_coeffs)), sorted_coeffs, color=colors, alpha=0.7)
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Coefficient Value')
    plt.title('Logistic Regression Coefficients (MAP)')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.grid(axis='x', alpha=0.3)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_prediction_accuracy(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            save_path: Optional[str] = None):
    """
    Plot prediction accuracy visualization.
    """
    from sklearn.metrics import accuracy_score
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy bar
    axes[0].bar(['Accuracy'], [accuracy], color='steelblue', alpha=0.7)
    axes[0].set_ylim([0, 1.1])
    axes[0].set_ylabel('Score')
    axes[0].set_title(f'Prediction Accuracy: {accuracy:.3f}')
    axes[0].text(0, accuracy + 0.02, f'{accuracy:.3f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Correct vs Incorrect predictions
    correct = (y_true == y_pred).sum()
    incorrect = len(y_true) - correct
    axes[1].bar(['Correct', 'Incorrect'], [correct, incorrect], 
               color=['green', 'red'], alpha=0.7)
    axes[1].set_ylabel('Count')
    axes[1].set_title('Prediction Results')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    axes[1].text(0, correct + max(correct, incorrect) * 0.01, f'{correct}',
                ha='center', va='bottom')
    axes[1].text(1, incorrect + max(correct, incorrect) * 0.01, f'{incorrect}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_logistic_diagnostics(preds: Dict,
                              labels: np.ndarray,
                              save_path: Optional[str] = None):
    """
    Wrapper function to generate all logistic regression diagnostic plots.
    """
    y_pred = preds['y_pred']
    y_proba = preds.get('y_proba', None)
    class_names = preds.get('class_names', None)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # Confusion matrix
    plot_confusion_matrix(
        labels, y_pred, class_names=class_names,
        save_path=f"{save_path}/confusion_matrix.png" if save_path else None
    )
    
    # Precision/Recall/F1 summary
    plot_precision_recall_f1(
        labels, y_pred, y_proba=y_proba,
        save_path=f"{save_path}/prf_summary.png" if save_path else None
    )
    
    # Calibration curve (if probabilities available)
    if y_proba is not None:
        plot_calibration_curve(
            labels, y_proba,
            save_path=f"{save_path}/calibration_curve.png" if save_path else None
        )
    
    # Prediction accuracy
    plot_prediction_accuracy(
        labels, y_pred,
        save_path=f"{save_path}/logistic_accuracy.png" if save_path else None
    )

if __name__ == "__main__":
    import argparse
    import pickle
    import json
    import sys
    import os
    # Add src to path to allow unpickling legacy models
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    
    parser = argparse.ArgumentParser(description='Generate Logistic Regression Visualizations')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots')
    parser.add_argument('--model_path', type=str, default='data/processed/logistic_model.pkl', help='Path to logistic model pickle')
    parser.add_argument('--lda_path', type=str, default='data/processed/lda_gibbs_model.json', help='Path to LDA model json (for theta features)')
    parser.add_argument('--data_path', type=str, default='data/processed/processed_data.json', help='Path to processed data json (for labels)')
    
    args = parser.parse_args()
    
    print(f"Generating logistic visualizations in {args.output_dir}...")
    
    try:
        # Load Model
        with open(args.model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Plot Coefficients (only needs model)
        if hasattr(model, 'weights'):
            os.makedirs(args.output_dir, exist_ok=True)
            plot_coefficients(model.weights, save_path=os.path.join(args.output_dir, "logistic_coefficients.png"))
            print("  Saved logistic_coefficients.png")
            
        # Try to load data for diagnostics
        if os.path.exists(args.lda_path) and os.path.exists(args.data_path):
            # Load features (theta)
            with open(args.lda_path, 'r', encoding='utf-8') as f:
                lda_data = json.load(f)
                theta = np.array(lda_data['theta'])
            
            # Load labels
            with open(args.data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            if isinstance(raw_data, list):
                labels = [1 if 'injury' in d.get('text', '').lower() else 0 for d in raw_data]
            else:
                doc_list = raw_data.get('documents', [])
                labels = [1 if 'injury' in d.get('text', '').lower() else 0 for d in doc_list]
                
            labels = np.array(labels)
            
            # Align lengths
            n_samples = min(len(theta), len(labels))
            X = theta[:n_samples]
            y = labels[:n_samples]
            
            # Run inference
            if hasattr(model, 'predict_proba') and hasattr(model, 'predict'):
                probs = model.predict_proba(X)
                preds = model.predict(X)
                
                diag_data = {
                    'y_pred': preds,
                    'y_proba': probs,
                    'class_names': ['No Injury', 'Injury Risk']
                }
                
                plot_logistic_diagnostics(diag_data, y, save_path=args.output_dir)
                print("  Saved diagnostic plots (confusion matrix, calibration, etc.)")
            else:
                 print("  Model missing predict/predict_proba methods. Skipping diagnostics.")
        else:
            print("  LDA model or Data not found. Skipping diagnostics requiring inference.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
