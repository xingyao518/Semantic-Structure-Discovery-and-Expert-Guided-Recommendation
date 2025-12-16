"""
Recommendation Module using Multiple Retrieval Methods.

This module implements three retrieval methods:
1. TF-IDF cosine similarity
2. LDA topic similarity (Hellinger distance)
3. Matrix factorization (ALS or MAP-based)

Mathematical formulations:
- TF-IDF: tf-idf(t,d) = tf(t,d) * log(N / df(t))
- Cosine similarity: cos(x,y) = (x · y) / (||x|| ||y||)
- Hellinger distance: H(P,Q) = (1/√2) * √(Σ_i (√p_i - √q_i)^2)
- Matrix factorization: R ≈ U V^T, where R is user-item matrix
  - ALS: Alternating Least Squares
  - MAP: Maximum A Posteriori with Gaussian priors
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from scipy.stats import entropy


class TFIDFRetriever:
    """
    TF-IDF based retrieval using cosine similarity.
    """
    
    def __init__(self, max_features: int = 5000):
        """
        Initialize TF-IDF retriever.
        
        Args:
            max_features: Maximum number of features
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features, 
                                         stop_words='english',
                                         lowercase=True)
        self.tfidf_matrix = None
        self.documents = None
    
    def fit(self, documents: List[str]):
        """
        Fit TF-IDF vectorizer on documents.
        
        Args:
            documents: List of text documents
        """
        self.documents = documents
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve top-k similar documents.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of (document_index, similarity_score) tuples
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarities
        similarities = (self.tfidf_matrix @ query_vector.T).toarray().flatten()
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(idx, float(similarities[idx])) for idx in top_indices]
        return results


class LDATopicRetriever:
    """
    LDA topic-based retrieval using Hellinger distance.
    
    Uses topic distributions to find similar documents.
    """
    
    def __init__(self):
        """Initialize LDA topic retriever."""
        self.topic_distributions = None
        self.documents = None
    
    def fit(self, topic_distributions: np.ndarray, documents: List[str]):
        """
        Fit retriever with topic distributions.
        
        Args:
            topic_distributions: Topic distributions matrix (N x K)
            documents: List of documents
        """
        self.topic_distributions = topic_distributions
        self.documents = documents
    
    def _hellinger_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Hellinger distance between two probability distributions.
        
        H(P,Q) = (1/√2) * √(Σ_i (√p_i - √q_i)^2)
        
        Args:
            p: First distribution
            q: Second distribution
            
        Returns:
            Hellinger distance
        """
        sqrt_p = np.sqrt(p + 1e-10)
        sqrt_q = np.sqrt(q + 1e-10)
        distance = np.sqrt(np.sum((sqrt_p - sqrt_q) ** 2))
        return distance / np.sqrt(2)
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute KL divergence: KL(P||Q) = Σ_i p_i log(p_i / q_i).
        
        Args:
            p: First distribution
            q: Second distribution
            
        Returns:
            KL divergence
        """
        return entropy(p + 1e-10, q + 1e-10)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def retrieve(self, 
                query_topic_dist: np.ndarray, 
                top_k: int = 10,
                metric: str = 'cosine') -> List[Tuple[int, float]]:
        """
        Retrieve top-k similar documents based on topic similarity.
        
        Args:
            query_topic_dist: Topic distribution for query (K,)
            top_k: Number of documents to retrieve
            metric: Similarity metric ('hellinger' or 'kl')
            
        Returns:
            List of (document_index, similarity_score) tuples
        """
        if self.topic_distributions is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        scores = []
        for i, doc_topic_dist in enumerate(self.topic_distributions):
            if metric == 'hellinger':
                dist = self._hellinger_distance(query_topic_dist, doc_topic_dist)
                similarity = 1.0 / (1.0 + dist)
            elif metric == 'kl':
                dist = self._kl_divergence(query_topic_dist, doc_topic_dist)
                similarity = 1.0 / (1.0 + dist)
            elif metric == 'cosine':
                similarity = self._cosine_similarity(query_topic_dist, doc_topic_dist)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append((i, similarity))
        
        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]


class MatrixFactorization:
    """
    Matrix Factorization for recommendation.
    
    Implements Alternating Least Squares (ALS) or MAP estimation.
    Factorizes R ≈ U V^T where R is document-query similarity matrix.
    """
    
    def __init__(self, 
                 num_factors: int = 50,
                 method: str = 'als',
                 lambda_reg: float = 0.1,
                 max_iterations: int = 100,
                 random_seed: int = 42):
        """
        Initialize matrix factorization model.
        
        Args:
            num_factors: Number of latent factors
            method: Factorization method ('als' or 'map')
            lambda_reg: Regularization parameter
            max_iterations: Maximum iterations
            random_seed: Random seed
        """
        self.num_factors = num_factors
        self.method = method
        self.lambda_reg = lambda_reg
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.U = None  # Document factors (N x F)
        self.V = None  # Query factors (M x F)
        self.R = None  # Original matrix (N x M)
    
    def _als_update_U(self, R: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Update U using ALS: U = R V (V^T V + λI)^(-1).
        
        Args:
            R: Rating matrix
            V: Query factors
            
        Returns:
            Updated document factors
        """
        VTV = V.T @ V
        regularization = self.lambda_reg * np.eye(self.num_factors)
        U = R @ V @ np.linalg.inv(VTV + regularization)
        return U
    
    def _als_update_V(self, R: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Update V using ALS: V = R^T U (U^T U + λI)^(-1).
        
        Args:
            R: Rating matrix
            U: Document factors
            
        Returns:
            Updated query factors
        """
        UTU = U.T @ U
        regularization = self.lambda_reg * np.eye(self.num_factors)
        V = R.T @ U @ np.linalg.inv(UTU + regularization)
        return V
    
    def _map_update_U(self, R: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Update U using MAP estimation with Gaussian prior.
        
        Maximizes: log p(R|U,V) + log p(U)
        where p(U) = N(0, σ^2 I)
        
        Args:
            R: Rating matrix
            V: Query factors
            
        Returns:
            Updated document factors
        """
        # MAP update is similar to ALS but with different regularization
        # This is a simplified version
        VTV = V.T @ V
        regularization = self.lambda_reg * np.eye(self.num_factors)
        U = R @ V @ np.linalg.inv(VTV + regularization)
        return U
    
    def fit(self, R: np.ndarray):
        """
        Fit matrix factorization model.
        
        Args:
            R: Document-query similarity matrix (N x M)
        """
        self.R = R
        N, M = R.shape
        
        # Initialize factors
        self.U = np.random.randn(N, self.num_factors) * 0.1
        self.V = np.random.randn(M, self.num_factors) * 0.1
        
        prev_error = np.inf
        
        for iteration in range(self.max_iterations):
            # Update U
            if self.method == 'als':
                self.U = self._als_update_U(self.R, self.V)
            else:  # map
                self.U = self._map_update_U(self.R, self.V)
            
            # Update V
            if self.method == 'als':
                self.V = self._als_update_V(self.R, self.U)
            else:  # map
                self.V = self._als_update_V(self.R, self.U)  # Same for V
            
            # Compute reconstruction error
            R_pred = self.U @ self.V.T
            error = np.mean((self.R - R_pred) ** 2)
            
            if abs(error - prev_error) < 1e-6:
                break
            
            prev_error = error
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}, Error: {error:.4f}")
    
    def predict(self, query_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Predict top-k documents for a query.
        
        Args:
            query_idx: Query index
            top_k: Number of documents to retrieve
            
        Returns:
            List of (document_index, score) tuples
        """
        if self.V is None or self.U is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Compute scores: U @ V[query_idx]^T
        query_factor = self.V[query_idx, :]
        scores = self.U @ query_factor
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [(idx, float(scores[idx])) for idx in top_indices]
        return results


class ExpertResponsibilityRetriever:
    """
    Expert-responsibility based retrieval using cosine similarity.
    
    Uses mixture-of-experts responsibilities γ as document embeddings.
    """
    
    def __init__(self):
        """Initialize expert responsibility retriever."""
        self.responsibilities = None
        self.documents = None
    
    def fit(self, responsibilities: np.ndarray, documents: List[str]):
        """
        Fit retriever with expert responsibilities.
        
        Args:
            responsibilities: Responsibility matrix (N x M) where M is num experts
            documents: List of documents
        """
        self.responsibilities = responsibilities
        self.documents = documents
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def retrieve(self, query_responsibilities: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve top-k similar documents based on expert responsibility similarity.
        
        Args:
            query_responsibilities: Expert responsibilities for query (M,)
            top_k: Number of documents to retrieve
            
        Returns:
            List of (document_index, similarity_score) tuples
        """
        if self.responsibilities is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        similarities = []
        for i, doc_resp in enumerate(self.responsibilities):
            sim = self._cosine_similarity(query_responsibilities, doc_resp)
            similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class RecommendationSystem:
    """
    Unified recommendation system combining multiple retrieval methods.
    """
    
    def __init__(self):
        """Initialize recommendation system."""
        self.tfidf_retriever = TFIDFRetriever()
        self.lda_retriever = LDATopicRetriever()
        self.expert_retriever = ExpertResponsibilityRetriever()
        self.mf_model = None
    
    def fit_tfidf(self, documents: List[str]):
        """Fit TF-IDF retriever."""
        self.tfidf_retriever.fit(documents)
    
    def fit_lda(self, topic_distributions: np.ndarray, documents: List[str]):
        """Fit LDA topic retriever."""
        self.lda_retriever.fit(topic_distributions, documents)
    
    def fit_expert(self, responsibilities: np.ndarray, documents: List[str]):
        """Fit expert responsibility retriever."""
        self.expert_retriever.fit(responsibilities, documents)
    
    def fit_matrix_factorization(self, similarity_matrix: np.ndarray):
        """Fit matrix factorization model."""
        self.mf_model = MatrixFactorization()
        self.mf_model.fit(similarity_matrix)
    
    def retrieve(self, 
                query: str,
                query_topic_dist: Optional[np.ndarray] = None,
                query_responsibilities: Optional[np.ndarray] = None,
                method: str = 'tfidf',
                top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve documents using specified method.
        
        Args:
            query: Query text
            query_topic_dist: Topic distribution for query (for LDA methods)
            query_responsibilities: Expert responsibilities for query (for expert method)
            method: Retrieval method ('tfidf', 'lda', 'lda_kl', 'expert_cosine', 'ensemble')
            top_k: Number of documents to retrieve
            
        Returns:
            List of (document_index, score) tuples
        """
        if method == 'tfidf':
            return self.tfidf_retriever.retrieve(query, top_k)
        elif method == 'lda':
            if query_topic_dist is None:
                raise ValueError("query_topic_dist required for LDA method")
            return self.lda_retriever.retrieve(query_topic_dist, top_k, metric='cosine')
        elif method == 'lda_kl':
            if query_topic_dist is None:
                raise ValueError("query_topic_dist required for LDA-KL method")
            return self.lda_retriever.retrieve(query_topic_dist, top_k, metric='kl')
        elif method == 'expert_cosine':
            if query_responsibilities is None:
                raise ValueError("query_responsibilities required for expert_cosine method")
            return self.expert_retriever.retrieve(query_responsibilities, top_k)
        elif method == 'mf':
            raise NotImplementedError("MF retrieval requires query index mapping")
        elif method == 'ensemble':
            # Combine multiple methods (simple average)
            results_tfidf = self.tfidf_retriever.retrieve(query, top_k * 2)
            combined = {}
            for idx, score in results_tfidf:
                combined[idx] = combined.get(idx, 0) + 0.33 * score
            
            if query_topic_dist is not None:
                results_lda = self.lda_retriever.retrieve(query_topic_dist, top_k * 2)
                for idx, score in results_lda:
                    combined[idx] = combined.get(idx, 0) + 0.33 * score
            
            if query_responsibilities is not None and self.expert_retriever.responsibilities is not None:
                results_expert = self.expert_retriever.retrieve(query_responsibilities, top_k * 2)
                for idx, score in results_expert:
                    combined[idx] = combined.get(idx, 0) + 0.34 * score
            
            sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            return sorted_results[:top_k]
        else:
            raise ValueError(f"Unknown method: {method}")


def compute_precision_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """
    Compute Precision@k.
    
    Args:
        retrieved: List of retrieved document indices
        relevant: List of relevant document indices
        k: Number of top documents to consider
        
    Returns:
        Precision@k score
    """
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    num_relevant = sum(1 for doc in retrieved_k if doc in relevant_set)
    return num_relevant / k if k > 0 else 0.0


def compute_recall_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """
    Compute Recall@k.
    
    Args:
        retrieved: List of retrieved document indices
        relevant: List of relevant document indices
        k: Number of top documents to consider
        
    Returns:
        Recall@k score
    """
    if len(relevant) == 0:
        return 0.0
    
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    num_relevant = sum(1 for doc in retrieved_k if doc in relevant_set)
    return num_relevant / len(relevant)


def compute_average_precision(retrieved: List[int], relevant: List[int]) -> float:
    """
    Compute Average Precision (AP).
    
    Args:
        retrieved: List of retrieved document indices
        relevant: List of relevant document indices
        
    Returns:
        Average Precision score
    """
    if len(relevant) == 0:
        return 0.0
    
    relevant_set = set(relevant)
    ap = 0.0
    num_relevant_found = 0
    
    for i, doc in enumerate(retrieved):
        if doc in relevant_set:
            num_relevant_found += 1
            precision_at_i = num_relevant_found / (i + 1)
            ap += precision_at_i
    
    return ap / len(relevant)


def compute_map_at_k(all_retrieved: List[List[int]], 
                     all_relevant: List[List[int]], 
                     k: int) -> float:
    """
    Compute Mean Average Precision @ k.
    
    Args:
        all_retrieved: List of retrieved document lists for each query
        all_relevant: List of relevant document lists for each query
        k: Number of top documents to consider
        
    Returns:
        MAP@k score
    """
    if len(all_retrieved) == 0:
        return 0.0
    
    total_ap = 0.0
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        retrieved_k = retrieved[:k]
        ap = compute_average_precision(retrieved_k, relevant)
        total_ap += ap
    
    return total_ap / len(all_retrieved)


if __name__ == "__main__":
    print("Recommendation System initialized")
    print("Methods:")
    print("  1. TF-IDF cosine similarity")
    print("  2. LDA topic Hellinger distance")
    print("  3. Matrix factorization (ALS/MAP)")

