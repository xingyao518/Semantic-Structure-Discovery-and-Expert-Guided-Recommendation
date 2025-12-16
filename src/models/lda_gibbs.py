"""
Collapsed Gibbs Sampling for Latent Dirichlet Allocation (LDA).

This module implements the collapsed Gibbs sampler for LDA as described in:
Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics.
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import random
from scipy.special import gammaln

class LDAGibbsSampler:
    """
    Collapsed Gibbs Sampler for LDA.
    """
    
    def __init__(self, 
                 num_topics: int = 10,
                 alpha: float = 0.1,
                 beta: float = 0.01,
                 random_seed: int = 42):
        """
        Initialize LDA Gibbs sampler.
        """
        self.K = num_topics
        self.alpha = alpha
        self.beta = beta
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Model parameters
        self.V = None
        self.D = None
        
        # Count matrices
        self.n_dk = None  # Document-topic counts
        self.n_kv = None  # Topic-word counts
        self.n_k = None   # Topic totals
        
        # Topic assignments
        self.z = None
        
        # Posterior parameters
        self.theta = None
        self.phi = None
        
        # History
        self.loglik_history = []
    
    def _initialize(self, documents: List[List[int]], vocab_size: int):
        """Initialize topic assignments randomly."""
        self.V = vocab_size
        self.D = len(documents)
        
        self.n_dk = np.zeros((self.D, self.K), dtype=int)
        self.n_kv = np.zeros((self.K, self.V), dtype=int)
        self.n_k = np.zeros(self.K, dtype=int)
        
        self.z = []
        for d, doc in enumerate(documents):
            doc_assignments = []
            for word_id in doc:
                k = random.randint(0, self.K - 1)
                doc_assignments.append(k)
                
                self.n_dk[d, k] += 1
                self.n_kv[k, word_id] += 1
                self.n_k[k] += 1
            
            self.z.append(doc_assignments)
        
        self.loglik_history = []
    
    def _sample_topic(self, d: int, n: int, word_id: int) -> int:
        """Sample topic assignment for word n in document d."""
        current_topic = self.z[d][n]
        self.n_dk[d, current_topic] -= 1
        self.n_kv[current_topic, word_id] -= 1
        self.n_k[current_topic] -= 1
        
        if word_id < 0 or word_id >= self.V:
            new_topic = random.randint(0, self.K - 1)
            self.n_dk[d, new_topic] += 1
            self.n_kv[new_topic, min(word_id, self.V - 1)] += 1
            self.n_k[new_topic] += 1
            return new_topic
        
        # Compute unnormalized probabilities
        # (n_{dk} + alpha) * (n_{kv} + beta) / (n_k + V*beta)
        
        # Optimized computation using numpy arrays
        # n_dk[d, :] is (K,)
        # n_kv[:, word_id] is (K,)
        # n_k is (K,)
        
        doc_topic_term = self.n_dk[d, :] + self.alpha
        topic_word_term = (self.n_kv[:, word_id] + self.beta) / (self.n_k + self.V * self.beta)
        
        probs = doc_topic_term * topic_word_term
        
        # Sample
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            probs = np.ones(self.K) / self.K
            
        new_topic = np.random.choice(self.K, p=probs)
        
        self.n_dk[d, new_topic] += 1
        self.n_kv[new_topic, word_id] += 1
        self.n_k[new_topic] += 1
        
        return new_topic
    
    def _compute_log_joint(self) -> float:
        """
        Compute log joint probability log P(w, z).
        Used for monitoring convergence.
        """
        # log P(w|z)
        # Sum over k: [ Sum over v: gammaln(n_kv + beta) ] - gammaln(n_k + V*beta)
        # Constant terms omitted for trend monitoring
        
        log_pw_z = 0
        # Term 1: Sum over k, v log gamma(n_kv + beta)
        log_pw_z += np.sum(gammaln(self.n_kv + self.beta))
        # Term 2: Sum over k log gamma(n_k + V*beta)
        log_pw_z -= np.sum(gammaln(self.n_k + self.V * self.beta))
        
        # log P(z)
        # Sum over d: [ Sum over k: gammaln(n_dk + alpha) ] - gammaln(n_d + K*alpha)
        
        log_pz = 0
        # Term 1: Sum over d, k log gamma(n_dk + alpha)
        log_pz += np.sum(gammaln(self.n_dk + self.alpha))
        # Term 2: Sum over d log gamma(n_d + K*alpha)
        doc_lengths = self.n_dk.sum(axis=1)
        log_pz -= np.sum(gammaln(doc_lengths + self.K * self.alpha))
        
        return log_pw_z + log_pz

    def fit(self, 
            documents: List[List[int]], 
            vocab_size: int,
            num_iterations: int = 100,
            burn_in: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train LDA model using Gibbs sampling.
        """
        self._initialize(documents, vocab_size)
        
        for iteration in range(num_iterations):
            for d, doc in enumerate(documents):
                for n, word_id in enumerate(doc):
                    new_topic = self._sample_topic(d, n, word_id)
                    self.z[d][n] = new_topic
            
            # Compute log likelihood
            ll = self._compute_log_joint()
            self.loglik_history.append(ll)
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, LL: {ll:.2f}")
        
        self._estimate_parameters()
        
        return self.theta, self.phi
    
    def _estimate_parameters(self):
        """Estimate theta and phi."""
        alpha_sum = self.K * self.alpha
        doc_totals = self.n_dk.sum(axis=1, keepdims=True)
        self.theta = (self.n_dk + self.alpha) / (doc_totals + alpha_sum)
        
        beta_sum = self.V * self.beta
        topic_totals = self.n_k.reshape(-1, 1)
        self.phi = (self.n_kv + self.beta) / (topic_totals + beta_sum)
    
    def infer_topics(self, document: List[int], num_iterations: int = 20) -> np.ndarray:
        """Infer topic distribution for a new document."""
        # Initialize
        doc_n_dk = np.zeros(self.K, dtype=int)
        doc_assignments = []
        
        for word_id in document:
            if word_id >= self.V:
                continue
            k = random.randint(0, self.K - 1)
            doc_assignments.append(k)
            doc_n_dk[k] += 1
            
        # Sample
        for _ in range(num_iterations):
            for n, word_id in enumerate(document):
                if word_id >= self.V:
                    continue
                
                current_topic = doc_assignments[n]
                doc_n_dk[current_topic] -= 1
                
                # Probs
                # P(z=k) \propto (n_dk + alpha) * phi_{k,v}
                # We use the trained phi here
                if self.phi is not None:
                    topic_word_prob = self.phi[:, word_id]
                else:
                    # Fallback if fit not called (shouldn't happen in usage)
                    topic_word_prob = np.ones(self.K) / self.K
                
                doc_topic_term = doc_n_dk + self.alpha
                probs = doc_topic_term * topic_word_prob
                
                probs = probs / (probs.sum() + 1e-10)
                new_topic = np.random.choice(self.K, p=probs)
                
                doc_assignments[n] = new_topic
                doc_n_dk[new_topic] += 1
        
        theta_new = (doc_n_dk + self.alpha) / (doc_n_dk.sum() + self.K * self.alpha)
        return theta_new

    def get_top_words(self, vocab: Dict[int, str], top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        if self.phi is None:
            raise ValueError("Model not trained.")
        
        top_words = {}
        for k in range(self.K):
            topic_probs = self.phi[k, :]
            top_indices = np.argsort(topic_probs)[::-1][:top_n]
            
            top_words[k] = [
                (vocab[i], topic_probs[i]) 
                for i in top_indices 
                if i in vocab
            ]
        return top_words
