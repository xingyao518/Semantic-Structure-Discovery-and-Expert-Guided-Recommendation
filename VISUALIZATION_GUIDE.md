# Visualization Guide

This document describes all visualization components and their usage.

## Directory Structure

All visualizations are saved to:

```
data/processed/
├── retrieval_visualizations/    # Retrieval method comparisons
│   ├── TF-IDF_scores.png
│   ├── LDA-topic_scores.png
│   ├── method_comparison.png
│   └── similarity_heatmap.png
└── visualizations/
    ├── lda/                      # LDA topic model visualizations
    │   ├── lda_top_words_topic0.png
    │   ├── lda_top_words_topic1.png
    │   ├── ... (one per topic)
    │   ├── lda_topic_distribution.png
    │   ├── lda_doc_topic_heatmap.png
    │   └── topic_*_wordcloud.png
    ├── mixture/                  # Mixture-of-experts visualizations
    │   ├── mixture_clusters.png
    │   ├── mixture_centroids.png
    │   ├── mixture_weights.png
    │   └── responsibilities.png
    └── logistic/                 # Logistic regression visualizations
        ├── logistic_coefficients.png
        ├── logistic_accuracy.png
        ├── confusion_matrix.png
        ├── prf_summary.png
        └── calibration_curve.png
```

## Visualization Modules

### 1. `visualization/visualize_lda.py`

**Functions:**
- `plot_top_words(phi, vocab, top_n, save_path)` - Top words per topic
- `plot_topic_distributions(theta, save_path)` - Average topic distribution
- `plot_document_topic_heatmap(theta, num_docs, save_path)` - Document-topic heatmap
- `plot_topic_word_distributions(phi, vocab, top_n, save_path)` - Bar charts
- `plot_topic_wordclouds(phi, vocab, save_path)` - Word clouds
- `plot_gibbs_convergence(log_likelihoods, burn_in, save_path)` - Convergence curve
- `plot_vi_convergence(elbo_values, save_path)` - ELBO convergence
- `visualize_all_topics(model_output, save_path)` - Comprehensive wrapper

**Usage in notebooks:**
- `notebooks/03_lda_gibbs.ipynb` - Gibbs sampling visualizations
- `notebooks/04_lda_vi.ipynb` - Variational inference visualizations

### 2. `visualization/visualize_retrieval.py`

**Functions:**
- `plot_top_k_scores(retrieved_results, top_k, method_name, save_path)` - Bar chart of scores
- `compare_retrieval_methods(method_results, top_k, save_path)` - Side-by-side comparison
- `plot_similarity_heatmap(similarity_matrix, query_labels, doc_labels, save_path)` - Heatmap
- `plot_retrieval_summary(results_dict, save_path)` - Comprehensive wrapper

**Usage in notebooks:**
- `notebooks/06_retrieval_eval.ipynb` - Retrieval evaluation

### 3. `visualization/visualize_logistic.py`

**Functions:**
- `plot_coefficients(coeff_vector, feature_names, save_path)` - Coefficient visualization
- `plot_prediction_accuracy(y_true, y_pred, save_path)` - Accuracy metrics
- `plot_confusion_matrix(y_true, y_pred, class_names, save_path)` - Confusion matrix
- `plot_precision_recall_f1(y_true, y_pred, y_proba, save_path)` - PR/F1 metrics
- `plot_calibration_curve(y_true, y_proba, n_bins, save_path)` - Calibration curve
- `plot_logistic_diagnostics(preds, labels, save_path)` - Comprehensive wrapper

**Usage in notebooks:**
- `notebooks/05_logistic_map.ipynb` - Logistic regression diagnostics

### 4. `visualization/visualize_mixture.py`

**Functions:**
- `plot_cluster_assignments(cluster_labels, save_path)` - Assignment distribution
- `plot_cluster_centroids(centroids, feature_names, save_path)` - Centroids heatmap
- `plot_mixture_weights(mixture_weights, expert_names, save_path)` - Weight bar chart
- `plot_advice_template_selection(expert_assignments, num_experts, save_path)` - Selection frequency
- `plot_posterior_responsibilities(responsibilities, query_labels, expert_names, save_path)` - Responsibility heatmap
- `visualize_mixture_results(mixture_output, save_path)` - Comprehensive wrapper

**Usage in notebooks:**
- `notebooks/06_retrieval_eval.ipynb` - Mixture-of-experts visualization

## Running Visualizations

### Step 1: Preprocess Data
```bash
# Run notebook 02 to preprocess Kaggle dataset
jupyter notebook notebooks/02_clean_and_label.ipynb
```

### Step 2: Train Models and Generate Visualizations

**LDA Gibbs Sampling:**
```bash
jupyter notebook notebooks/03_lda_gibbs.ipynb
# Generates: data/processed/visualizations/lda/*.png
```

**LDA Variational Inference:**
```bash
jupyter notebook notebooks/04_lda_vi.ipynb
# Generates: data/processed/visualizations/lda/*.png
```

**Logistic Regression:**
```bash
jupyter notebook notebooks/05_logistic_map.ipynb
# Generates: data/processed/visualizations/logistic/*.png
```

**Retrieval and Mixture:**
```bash
jupyter notebook notebooks/06_retrieval_eval.ipynb
# Generates: 
#   - data/processed/retrieval_visualizations/*.png
#   - data/processed/visualizations/mixture/*.png
```

## Expected Output Files

After running all notebooks, you should have:

### Retrieval Visualizations
- `TF-IDF_scores.png` - Top-K TF-IDF retrieval scores
- `LDA-topic_scores.png` - Top-K LDA-topic retrieval scores (if LDA trained)
- `method_comparison.png` - Side-by-side method comparison
- `similarity_heatmap.png` - Query-document similarity matrix

### LDA Visualizations
- `lda_top_words_topic0.png` through `lda_top_words_topic9.png` - Top words per topic
- `lda_topic_distribution.png` - Average topic distribution
- `lda_doc_topic_heatmap.png` - Document-topic assignments
- `topic_*_wordcloud.png` - Word clouds for each topic
- `gibbs_convergence.png` or `vi_convergence.png` - Convergence curves

### Logistic Regression Visualizations
- `logistic_coefficients.png` - Coefficient values
- `logistic_accuracy.png` - Prediction accuracy metrics
- `confusion_matrix.png` - Classification confusion matrix
- `prf_summary.png` - Precision/Recall/F1 summary
- `calibration_curve.png` - Probability calibration

### Mixture-of-Experts Visualizations
- `mixture_clusters.png` - Expert assignment distribution
- `mixture_centroids.png` - Average mixture weights
- `mixture_weights.png` - Component weights bar chart
- `expert_selection.png` - Template selection frequency
- `responsibilities.png` - Posterior responsibility heatmap

## Notes

- All directories are automatically created using `os.makedirs(save_path, exist_ok=True)`
- All plots are saved with 300 DPI resolution
- Functions handle missing data gracefully (optional parameters)
- Visualizations use consistent color schemes and styling

