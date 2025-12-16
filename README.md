##  Project Overview

1. **Topic Modeling**: LDA via Gibbs sampling and Variational Inference
2. **Bayesian Mixture-of-Experts**: Generates structured advice based on topic mixtures
3. **Logistic Regression with MAP**: Predicts risk-factor categories
4. **Recommendation System**: Multiple retrieval methods (TF-IDF, LDA-topic similarity, matrix factorization)
5. **Evaluation Modules**: Compliance checks, relevance metrics, and qualitative scoring

##  Project Structure

```
.
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data and models
├── notebooks/            # Jupyter notebooks for development
│   ├── 01_download_data.ipynb
│   ├── 02_clean_and_label.ipynb
│   ├── 03_lda_gibbs.ipynb
│   ├── 04_lda_vi.ipynb
│   ├── 05_logistic_map.ipynb
│   └── 06_retrieval_eval.ipynb
├── src/
│   ├── data/             # Data processing modules
│   │   ├── fetch_reddit.py
│   │   ├── preprocess.py
│   │   └── labeling.py
│   ├── models/           # Probabilistic models
│   │   ├── lda_gibbs.py
│   │   ├── lda_vi.py
│   │   ├── bayesian_mixture.py
│   │   ├── logistic_map.py
│   │   └── recommendation.py
│   ├── evaluation/       # Evaluation modules
│   │   ├── compliance_check.py
│   │   ├── relevance_metrics.py
│   │   └── qualitative_eval.py
│   └── app/              # Application interface
│       └── run_cli.py
├── visualization/        # Visualization modules
│   ├── visualize_lda.py
│   ├── visualize_retrieval.py
│   ├── visualize_logistic.py
│   └── visualize_mixture.py
├── requirements.txt
└── README.md
```

##  Statistical Models Used

### 1. Latent Dirichlet Allocation (LDA)

- **Gibbs Sampling** (`lda_gibbs.py`): Collapsed Gibbs sampler
  - Mathematical formulation: `P(z_{dn} = k | z_{-dn}, w) ∝ (n_{dk} + α) * (n_{kv} + β) / (n_k + V*β)`
  - Integrates out multinomial parameters θ and φ
  - Samples only topic assignments z

- **Variational Inference** (`lda_vi.py`): Mean-field variational inference
  - Variational distribution: `q(θ) = Dir(γ), q(φ) = Dir(λ), q(z) = Mult(φ)`
  - Coordinate ascent updates for γ and λ
  - Maximizes ELBO (Evidence Lower BOund)

### 2. Bayesian Mixture-of-Experts (`bayesian_mixture.py`)

- Mixture weights depend on LDA topic mixture: `π_e = softmax(Σ_k topic_mixture[k] * W[k][e])`
- Gibbs sampling for component assignment: `P(z = e | z_{-i}, topics) ∝ (n_e + α) * π_e`
- MAP updates for topic-expert weight matrix

### 3. Logistic Regression with MAP (`logistic_map.py`)

- Likelihood: `p(y|x,w) = σ(w^T x)^y * (1-σ(w^T x))^(1-y)`
- Prior: `p(w) = N(0, σ^2 I)`
- MAP estimate: `w* = argmax_w [log p(D|w) + log p(w)]`
- Optimization via gradient ascent or Newton's method

### 4. Recommendation System (`recommendation.py`)

Three retrieval methods:

1. **TF-IDF**: Cosine similarity on TF-IDF vectors
2. **LDA-topic similarity**: Hellinger distance between topic distributions
3. **Matrix Factorization**: ALS or MAP-based factorization `R ≈ U V^T`


### Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download NLTK data (if not already downloaded):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### Usage

#### Command Line Interface

Run the interactive CLI:

```bash
python src/app/run_cli.py --interactive
```

Or process a single query:

```bash
python src/app/run_cli.py --query "I have knee pain after running. What should I do?"
```

#### Jupyter Notebooks

Run the notebooks in order:

1. `01_download_data.ipynb` - Download running data from Reddit
2. `02_clean_and_label.ipynb` - Preprocess and label data
3. `03_lda_gibbs.ipynb` - Train LDA with Gibbs sampling
4. `04_lda_vi.ipynb` - Train LDA with Variational Inference
5. `05_logistic_map.ipynb` - Train logistic regression models
6. `06_retrieval_eval.ipynb` - Evaluate retrieval and advice quality

## 📊 Pipeline

The system processes queries through the following pipeline:

1. **Text Preprocessing**: Tokenization, cleaning, lemmatization, stopword removal
2. **Risk Factor Labeling**: Rule-based keyword matching for injury, heat, nutrition risks
3. **Topic Inference**: LDA topic mixture inference for the query
4. **Risk Prediction**: MAP logistic regression predicts risk categories
5. **Document Retrieval**: Find similar cases using TF-IDF, LDA, or matrix factorization
6. **Advice Generation**: Mixture-of-experts model generates structured advice
7. **Evaluation**: Compliance checks, relevance metrics, and qualitative scoring

##  Evaluation

The system includes three evaluation modules:

### 1. Compliance Check (`compliance_check.py`)

Checks whether advice:
- Respects injury constraints (rest recommendations)
- Respects weather constraints (heat warnings)
- Is appropriate for experience level
- Follows safe progression (10% rule)

### 2. Relevance Metrics (`relevance_metrics.py`)

Computes:
- Cosine similarity (TF-IDF)
- KL divergence between topic mixtures
- Hellinger distance
- Precision, Recall, F1, NDCG for retrieval

### 3. Qualitative Evaluation (`qualitative_eval.py`)

Rubric-based scoring:
- **Clarity**: Clear language, specific recommendations, actionable steps
- **Safety**: No dangerous recommendations, appropriate for risk level
- **Personalization**: References user's situation, experience, goals
- **Correctness**: Factually correct, follows best practices

##  Visualization and Evaluation

The project includes comprehensive visualization and evaluation modules for all statistical models.

### LDA Visualizations (`visualize_lda.py`)

Visualization tools for both Gibbs sampling and Variational Inference LDA models:

- **Topic Word Distributions**: Bar charts showing top words for each topic
- **Topic Word Clouds**: Visual word clouds for each topic
- **Document-Topic Distributions**: Heatmaps showing topic assignments across documents
- **Convergence Curves**: 
  - Gibbs sampling: Log-likelihood over iterations
  - Variational Inference: ELBO (Evidence Lower BOund) over iterations

These visualizations help understand:
- What topics the model discovered
- How topics are distributed across documents
- Whether the model has converged

### Retrieval Visualizations (`visualize_retrieval.py`)

Tools for evaluating and comparing retrieval methods:

- **Top-K Scores**: Bar plots showing similarity scores for retrieved documents
- **Method Comparison**: Side-by-side comparison of TF-IDF, LDA-topic, and Matrix Factorization
- **Similarity Heatmaps**: Query-document similarity matrices

These visualizations measure:
- Retrieval performance across different methods
- Which method works best for different query types
- Similarity patterns between queries and documents

### Logistic Model Evaluation (`visualize_logistic.py`)

Diagnostic plots for logistic regression with MAP estimation:

- **Confusion Matrix**: Classification performance visualization
- **Precision/Recall/F1 Summary**: Bar charts of classification metrics
- **Precision-Recall Curve**: Trade-off between precision and recall
- **Calibration Curve**: Reliability diagram showing probability calibration

These plots help assess:
- Model accuracy and error types
- Class imbalance issues
- Probability calibration quality

### Mixture-of-Experts Inspection (`visualize_mixture.py`)

Visualization tools for the Bayesian mixture-of-experts model:

- **Mixture Component Weights**: Bar charts showing expert weights
- **Advice Template Selection**: Frequency of expert template usage
- **Posterior Responsibilities**: Heatmap of mixture weights across queries

These visualizations show:
- Which experts are most influential
- How expert selection varies with topic mixtures
- Posterior distribution over experts

### Compliance, Relevance, and Qualitative Evaluation

Extended evaluation functions:

- **Compliance Checks**: Injury constraints, weather constraints, progression safety, experience alignment
- **Relevance Metrics**: Topic KL distance, retrieval score summaries, method comparisons
- **Qualitative Ratings**: Clarity, safety, personalization, and overall quality scores

##  Mathematical Formulations

All models include detailed mathematical formulations in their docstrings:

- **LDA Gibbs**: Collapsed Gibbs sampling formula
- **LDA VI**: Variational updates and ELBO calculation
- **Mixture-of-Experts**: Topic-dependent mixture weights and Gibbs sampling
- **Logistic MAP**: Log-posterior, gradient, and Hessian
- **Matrix Factorization**: ALS updates and MAP estimation

##  Development

This is a starter codebase designed for incremental development:

- All modules have clean skeletons with proper structure
- Mathematical formulations are documented
- Models can be trained and evaluated independently
- Data processing pipeline is modular

## Contributing

This is a course project. Extend and modify as needed for your specific requirements.


