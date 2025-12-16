"""
Relevance Metrics for Retrieval Evaluation.

This module computes various relevance metrics:
- Cosine similarity (TF-IDF)
- KL divergence between topic mixtures
- Retrieval accuracy for held-out queries
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from sklearn.metrics import precision_recall_fscore_support, ndcg_score


class RelevanceMetrics:
    """
    Computes relevance metrics for retrieval evaluation.
    """
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        cos(x, y) = (x · y) / (||x|| ||y||)
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0 to 1)
        """
        # Handle sparse vectors
        if hasattr(vec1, 'toarray'):
            vec1 = vec1.toarray().flatten()
        if hasattr(vec2, 'toarray'):
            vec2 = vec2.toarray().flatten()
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute KL divergence: KL(P||Q) = Σ_i p_i log(p_i / q_i).
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            KL divergence (non-negative)
        """
        # Add small epsilon to avoid log(0)
        p = p + 1e-10
        q = q + 1e-10
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        return entropy(p, q)
    
    @staticmethod
    def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Hellinger distance: H(P,Q) = (1/√2) * √(Σ_i (√p_i - √q_i)^2).
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            Hellinger distance (0 to 1)
        """
        # Normalize
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)
        
        sqrt_p = np.sqrt(p)
        sqrt_q = np.sqrt(q)
        
        distance = np.sqrt(np.sum((sqrt_p - sqrt_q) ** 2))
        return distance / np.sqrt(2)
    
    @staticmethod
    def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Jensen-Shannon divergence.
        
        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        where M = 0.5 * (P + Q)
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            JS divergence (0 to 1)
        """
        # Normalize
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)
        
        m = 0.5 * (p + q)
        js = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
        return js
    
    def compute_topic_similarity(self, 
                                topic_dist1: np.ndarray, 
                                topic_dist2: np.ndarray,
                                metric: str = 'kl') -> float:
        """
        Compute similarity between two topic distributions.
        
        Args:
            topic_dist1: First topic distribution
            topic_dist2: Second topic distribution
            metric: Similarity metric ('kl', 'hellinger', 'js', 'cosine')
            
        Returns:
            Similarity/distance value
        """
        if metric == 'kl':
            return self.kl_divergence(topic_dist1, topic_dist2)
        elif metric == 'hellinger':
            return self.hellinger_distance(topic_dist1, topic_dist2)
        elif metric == 'js':
            return self.jensen_shannon_divergence(topic_dist1, topic_dist2)
        elif metric == 'cosine':
            return 1.0 - self.cosine_similarity(topic_dist1, topic_dist2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def compute_retrieval_metrics(self, 
                                 retrieved_indices: List[int],
                                 relevant_indices: List[int],
                                 k: int = None) -> Dict:
        """
        Compute retrieval metrics (precision, recall, F1, NDCG).
        
        Args:
            retrieved_indices: List of retrieved document indices
            relevant_indices: List of relevant document indices (ground truth)
            k: Cutoff for top-k evaluation (if None, use all retrieved)
            
        Returns:
            Dictionary with metrics
        """
        if k is not None:
            retrieved_indices = retrieved_indices[:k]
        
        # Convert to binary relevance
        retrieved_set = set(retrieved_indices)
        relevant_set = set(relevant_indices)
        
        # True positives: retrieved and relevant
        tp = len(retrieved_set & relevant_set)
        
        # Precision: tp / (tp + fp) = tp / |retrieved|
        precision = tp / len(retrieved_indices) if len(retrieved_indices) > 0 else 0.0
        
        # Recall: tp / (tp + fn) = tp / |relevant|
        recall = tp / len(relevant_indices) if len(relevant_indices) > 0 else 0.0
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # NDCG (simplified - assumes binary relevance)
        # Create relevance vector
        relevance = [1 if idx in relevant_set else 0 for idx in retrieved_indices]
        if len(relevance) > 0:
            # NDCG@k
            ndcg = ndcg_score([relevance], [relevance], k=k or len(relevance))
        else:
            ndcg = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ndcg': ndcg,
            'num_retrieved': len(retrieved_indices),
            'num_relevant': len(relevant_indices),
            'num_relevant_retrieved': tp
        }
    
    def compute_average_precision(self, 
                                 retrieved_indices: List[int],
                                 relevant_indices: List[int]) -> float:
        """
        Compute Average Precision (AP).
        
        AP = (1/|R|) * Σ_k P@k * rel(k)
        where R is set of relevant documents, rel(k) is relevance at rank k
        
        Args:
            retrieved_indices: List of retrieved document indices
            relevant_indices: List of relevant document indices
            
        Returns:
            Average precision
        """
        relevant_set = set(relevant_indices)
        
        if len(relevant_set) == 0:
            return 0.0
        
        precision_sum = 0.0
        num_relevant_so_far = 0
        
        for k, idx in enumerate(retrieved_indices, 1):
            if idx in relevant_set:
                num_relevant_so_far += 1
                precision_at_k = num_relevant_so_far / k
                precision_sum += precision_at_k
        
        ap = precision_sum / len(relevant_set)
        return ap
    
    def compute_mean_reciprocal_rank(self, 
                                    retrieved_indices: List[int],
                                    relevant_indices: List[int]) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        MRR = 1 / rank of first relevant document
        
        Args:
            retrieved_indices: List of retrieved document indices
            relevant_indices: List of relevant document indices
            
        Returns:
            Reciprocal rank
        """
        relevant_set = set(relevant_indices)
        
        for rank, idx in enumerate(retrieved_indices, 1):
            if idx in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    def evaluate_retrieval(self,
                          query_results: List[Tuple[List[int], List[int]]],
                          metric: str = 'precision') -> Dict:
        """
        Evaluate retrieval performance across multiple queries.
        
        Args:
            query_results: List of (retrieved_indices, relevant_indices) tuples
            metric: Metric to compute ('precision', 'recall', 'f1', 'ap', 'mrr')
            
        Returns:
            Dictionary with average metrics
        """
        metrics_list = []
        
        for retrieved, relevant in query_results:
            if metric == 'precision':
                metrics_list.append(self.compute_retrieval_metrics(retrieved, relevant)['precision'])
            elif metric == 'recall':
                metrics_list.append(self.compute_retrieval_metrics(retrieved, relevant)['recall'])
            elif metric == 'f1':
                metrics_list.append(self.compute_retrieval_metrics(retrieved, relevant)['f1'])
            elif metric == 'ap':
                metrics_list.append(self.compute_average_precision(retrieved, relevant))
            elif metric == 'mrr':
                metrics_list.append(self.compute_mean_reciprocal_rank(retrieved, relevant))
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        return {
            'mean': np.mean(metrics_list),
            'std': np.std(metrics_list),
            'min': np.min(metrics_list),
            'max': np.max(metrics_list),
            'all_values': metrics_list
        }
    
    def topic_kl_distance(self, theta_query: np.ndarray, theta_doc: np.ndarray) -> float:
        """
        Compute KL divergence between query and document topic distributions.
        
        KL(P_query || P_doc) = Σ_k P_query(k) * log(P_query(k) / P_doc(k))
        
        Args:
            theta_query: Topic distribution for query (K,)
            theta_doc: Topic distribution for document (K,)
            
        Returns:
            KL divergence value
        """
        return self.kl_divergence(theta_query, theta_doc)
    
    def retrieval_score_summary(self, results_dict: Dict) -> Dict:
        """
        Generate summary statistics for retrieval results.
        
        Args:
            results_dict: Dictionary containing:
                - 'method_scores': Dict mapping method_name -> list of scores
                - 'queries': Optional list of query identifiers
                
        Returns:
            Dictionary with summary statistics (mean, std, min, max per method)
        """
        method_scores = results_dict.get('method_scores', {})
        summary = {}
        
        for method_name, scores in method_scores.items():
            scores_array = np.array(scores)
            summary[method_name] = {
                'mean': float(np.mean(scores_array)),
                'std': float(np.std(scores_array)),
                'min': float(np.min(scores_array)),
                'max': float(np.max(scores_array)),
                'median': float(np.median(scores_array))
            }
        
        return summary
    
    def compare_retrieval_methods(self, method_scores: Dict[str, List[float]]) -> Dict:
        """
        Compare multiple retrieval methods using statistical tests.
        
        Args:
            method_scores: Dictionary mapping method_name -> list of scores
            
        Returns:
            Dictionary with comparison results including:
            - mean scores per method
            - pairwise comparisons (if applicable)
        """
        comparison = {}
        
        # Calculate statistics for each method
        for method_name, scores in method_scores.items():
            scores_array = np.array(scores)
            comparison[method_name] = {
                'mean': float(np.mean(scores_array)),
                'std': float(np.std(scores_array)),
                'n': len(scores_array)
            }
        
        # Find best method by mean score
        best_method = max(comparison.items(), key=lambda x: x[1]['mean'])
        comparison['best_method'] = {
            'name': best_method[0],
            'mean_score': best_method[1]['mean']
        }
        
        return comparison


if __name__ == "__main__":
    metrics = RelevanceMetrics()
    
    # Example
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.4, 0.4, 0.2])
    
    kl = metrics.kl_divergence(p, q)
    hellinger = metrics.hellinger_distance(p, q)
    
    print(f"KL divergence: {kl:.4f}")
    print(f"Hellinger distance: {hellinger:.4f}")

