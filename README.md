# Semantic Structure Discovery and Expert-Guided Recommendation

This project studies interpretable recommendation systems by combining topic modeling, Bayesian mixture models, and retrieval-based methods.  
The focus is on **semantic structure discovery and explainable advice generation**, rather than end-to-end black-box optimization.

---

## Core Components

- **Topic Modeling**  
  Latent Dirichlet Allocation (LDA) implemented using Gibbs sampling and variational inference.

- **Bayesian Mixture-of-Experts**  
  Topic-conditioned mixture model for selecting expert advice templates.

- **Logistic Regression (MAP)**  
  Bayesian logistic regression for predicting risk-related categories.

- **Retrieval-Based Recommendation**  
  Similarity-based retrieval using TF-IDF, topic-space distance, and matrix factorization.

- **Evaluation & Visualization**  
  Quantitative metrics, qualitative scoring, and visualization tools for interpretability.

---

## Repository Structure

```text
.
├── data/                 # Raw and processed data
├── notebooks/            # Exploratory analysis and experiments
├── src/                  # Core implementation
│   ├── data/             # Data processing
│   ├── models/           # Probabilistic models
│   ├── recommendation/   # Retrieval and advice logic
│   ├── evaluation/       # Evaluation metrics
│   └── pipeline/         # Step-by-step execution
├── visualization/        # Visualization utilities
├── requirements.txt
└── README.md
```

---

## Methods Overview

- LDA provides low-dimensional semantic representations of text.
- Topic mixtures condition expert selection in a Bayesian mixture-of-experts model.
- Logistic regression with MAP estimation predicts risk-related categories.
- Multiple retrieval strategies are compared for recommendation quality.

---

## Pipeline

1. Text preprocessing and labeling  
2. Topic inference for queries  
3. Risk category prediction  
4. Similar case retrieval  
5. Expert-guided advice generation  
6. Quantitative and qualitative evaluation  

---

## Evaluation

The system is evaluated using:

- **Relevance metrics** for retrieval quality  
- **Compliance checks** for safety and constraint adherence  
- **Qualitative scoring** for clarity, safety, and personalization  

Visualization modules are provided to inspect topic structure, model behavior, and recommendation outcomes.

---

## Usage

Install dependencies:

```bash
pip install -r requirements.txt


---

## Methods Overview

- LDA provides low-dimensional semantic representations of text.
- Topic mixtures condition expert selection in a Bayesian mixture-of-experts model.
- Logistic regression with MAP estimation predicts risk-related categories.
- Multiple retrieval strategies are compared for recommendation quality.

---

## Pipeline

1. Text preprocessing and labeling  
2. Topic inference for queries  
3. Risk category prediction  
4. Similar case retrieval  
5. Expert-guided advice generation  
6. Quantitative and qualitative evaluation  

---

## Evaluation

The system is evaluated using:

- **Relevance metrics** for retrieval quality  
- **Compliance checks** for safety and constraint adherence  
- **Qualitative scoring** for clarity, safety, and personalization  

Visualization modules are provided to inspect topic structure, model behavior, and recommendation outcomes.

---

## Usage

Install dependencies:

```bash
pip install -r requirements.txt

Run the pipeline step by step:

python run_pipeline_step_by_step.py

Exploratory analyses and experiments are documented in the notebooks/ directory.

```
