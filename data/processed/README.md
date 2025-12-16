# Processed Fitness Dataset

## Dataset Source

This dataset is derived from the **Kaggle r/Fitness** dataset, containing posts and discussions from the Reddit r/Fitness community.

## Data Cleaning Rules

### 1. Column Selection
- Kept columns: `text`, `title`, `score`, `datetime`, `post_id`
- Removed all other columns

### 2. Row Filtering
- **Removed null text**: Dropped rows where `text` is null
- **Removed deleted posts**: Filtered out posts with text equal to:
  - "deleted"
  - "removed"
  - "[deleted]"
  - "[removed]"

### 3. Text Cleaning
- **Lowercasing**: All text converted to lowercase
- **Markdown removal**: Stripped markdown characters (`>`, `*`, `_`, backticks)
- **URL removal**: Removed URLs using regex pattern
- **Whitespace normalization**: Collapsed multiple spaces to single space, trimmed edges

### 4. Fitness-Related Filtering

A post is included if it contains **ANY** of the following case-insensitive substrings (root-based matching):

**Training & Exercise:**
- train, workout, exercise, gym, routine

**Strength & Weight:**
- strength, lift, weight, muscl, cardio, run

**Health & Recovery:**
- injur, pain, sore, recover

**Nutrition & Diet:**
- diet, nutri, fat, calor, bulk, cut, protein

**Advice & Help:**
- advice, help, tip

**Note**: Root-based matching means "injury", "training", "muscles", "lifting", "workouts" all match their respective root keywords.

### 5. Additional Rules
- **Minimum length**: Keep posts where cleaned text length > 20 characters
- **ID assignment**: Reset index and assign sequential ID field

## Final Dataset Structure

The processed dataset is saved as `processed_data.json` with the following structure:

```json
{
  "documents": [
    {
      "id": <int>,
      "text": "<clean_text>",
      "title": "<title or empty string>",
      "score": <int>
    },
    ...
  ],
  "tokenized_docs": [...],
  "vocab": {...},
  "labels": [...]
}
```

## Dataset Statistics

- **Original rows**: Varies by source CSV
- **After cleaning**: Removed null/deleted posts
- **After fitness filter**: Only fitness-related content
- **Final dataset size**: See notebook `02_clean_and_label.ipynb` for current statistics

## Why This Corpus is Ideal for Probabilistic Models

### 1. **Diverse Topics for LDA**
The fitness domain naturally contains multiple coherent topics:
- Training routines and programs
- Injury prevention and recovery
- Nutrition and diet planning
- Strength training vs. cardio
- Beginner advice vs. advanced techniques
- Equipment and gear discussions

This diversity allows LDA (both Gibbs sampling and Variational Inference) to discover meaningful topic structures.

### 2. **Mixture Clusters**
The dataset contains distinct clusters that map well to mixture-of-experts models:
- **Injury/Recovery experts**: Posts about pain, soreness, rehabilitation
- **Training experts**: Workout routines, program design
- **Nutrition experts**: Diet, macros, supplements
- **General advice experts**: Beginner questions, general tips

These clusters enable the Bayesian mixture-of-experts model to learn topic-dependent expert weights.

### 3. **MAP Logistic Regression**
The `score` field provides a natural target for logistic regression:
- **High-score posts**: Likely contain valuable advice (positive class)
- **Low-score posts**: May contain less useful or controversial content (negative class)

This allows training MAP logistic regression models to predict post quality or risk factors (injury risk, nutrition concerns, etc.) based on topic distributions or text features.

### 4. **Retrieval Tasks**
The dataset supports multiple retrieval scenarios:
- **Query**: "I have knee pain, what should I do?"
- **Retrieval methods**:
  - **TF-IDF**: Find posts with similar word frequencies
  - **LDA-topic similarity**: Find posts with similar topic distributions
  - **Matrix Factorization**: Learn latent user-item (query-document) similarities

The diversity of topics and advice types makes retrieval evaluation meaningful.

## Usage

1. **Preprocessing**: Run `src/data/preprocess_kaggle.py` or use `notebooks/02_clean_and_label.ipynb`
2. **Exploration**: See `notebooks/03_explore_kaggle.ipynb` for data exploration
3. **Modeling**: Use processed data in:
   - `notebooks/03_lda_gibbs.ipynb` - LDA with Gibbs sampling
   - `notebooks/04_lda_vi.ipynb` - LDA with Variational Inference
   - `notebooks/05_logistic_map.ipynb` - Logistic regression with MAP
   - `notebooks/06_retrieval_eval.ipynb` - Retrieval system evaluation

## File Locations

- **Raw data**: `data/raw/data.csv`
- **Processed data**: `data/processed/processed_data.json`
- **Retrieval visualizations**: `data/processed/retrieval_visualizations/`

