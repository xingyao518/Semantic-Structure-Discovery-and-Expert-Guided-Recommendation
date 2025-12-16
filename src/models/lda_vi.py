"""
Mean-Field Variational Inference for Latent Dirichlet Allocation (LDA).

This module implements mean-field variational inference for LDA as described in:
Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation.

Mathematical formulation:
- Variational distribution: q(z, θ, φ) = q(θ) * q(φ) * q(z)
- q(θ_d) = Dir(γ_d), q(φ_k) = Dir(λ_k), q(z_{dn}) = Mult(φ_{dn})
- Coordinate ascent updates:
  - γ_{dk} = α_k + Σ_n φ_{dnk}
  - λ_{kv} = β_v + Σ_d Σ_n φ_{dnk} * [w_{dn} = v]
  - φ_{dnk} ∝ exp(E[log θ_{dk}] + E[log φ_{kw_{dn}}])
- ELBO: E[log p(w|z,φ)] + E[log p(z|θ)] + E[log p(θ|α)] + E[log p(φ|β)] - E[log q]
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import random


class LDAVariationalInference:
    """
    Mean-Field Variational Inference for LDA.
    
    Uses coordinate ascent to optimize the variational parameters
    γ (document-topic) and λ (topic-word) by maximizing the ELBO.
    """
    
    def __init__(self, 
                 num_topics: int = 10,
                 alpha: float = 0.1,
                 beta: float = 0.01,
                 random_seed: int = 42):
        """
        Initialize LDA VI model.
        
        Args:
            num_topics: Number of topics K
            alpha: Dirichlet prior parameter for document-topic distributions
            beta: Dirichlet prior parameter for topic-word distributions
            random_seed: Random seed for reproducibility
        """
        self.K = num_topics
        self.alpha = alpha
        self.beta = beta
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Model parameters
        self.V = None  # Vocabulary size
        self.D = None  # Number of documents
        
        # Variational parameters
        self.gamma = None  # Document-topic variational parameters: γ[d][k]
        self.lambda_param = None  # Topic-word variational parameters: λ[k][v]
        self.phi = None  # Word-topic variational parameters: φ[d][n][k]
        
        # Posterior estimates
        self.theta = None  # Document-topic distributions: θ[d][k] = E[θ_d] = γ_d / sum(γ_d)
        self.phi_est = None  # Topic-word distributions: φ[k][v] = E[φ_k] = λ_k / sum(λ_k)
    
    def _digamma(self, x: np.ndarray) -> np.ndarray:
        """
        Compute digamma function (derivative of log Gamma).
        
        Uses approximation for numerical stability.
        
        Args:
            x: Input array
            
        Returns:
            Digamma values
        """
        from scipy.special import digamma
        return digamma(x)
    
    def _initialize(self, documents: List[List[int]], vocab_size: int):
        """
        Initialize variational parameters.
        
        Args:
            documents: List of documents (word ID lists)
            vocab_size: Vocabulary size
        """
        self.V = vocab_size
        self.D = len(documents)
        
        # Initialize γ: document-topic parameters
        # γ_{dk} = α_k + random initialization
        self.gamma = np.random.gamma(100, 1.0/100, size=(self.D, self.K)) + self.alpha
        
        # Initialize λ: topic-word parameters
        # λ_{kv} = β_v + random initialization
        self.lambda_param = np.random.gamma(100, 1.0/100, size=(self.K, self.V)) + self.beta
        
        # Initialize φ: word-topic variational parameters
        self.phi = []
        for d, doc in enumerate(documents):
            doc_phi = np.random.dirichlet([1.0/self.K] * self.K, size=len(doc))
            self.phi.append(doc_phi)
    
    def _update_phi(self, d: int, documents: List[List[int]]):
        """
        Update word-topic variational parameters φ for document d.
        
        φ_{dnk} ∝ exp(E[log θ_{dk}] + E[log φ_{kw_{dn}}])
        where:
        - E[log θ_{dk}] = digamma(γ_{dk}) - digamma(sum_k γ_{dk})
        - E[log φ_{kw_{dn}}] = digamma(λ_{kw_{dn}}) - digamma(sum_v λ_{kv})
        
        Args:
            d: Document index
            documents: List of all documents
        """
        doc = documents[d]
        N_d = len(doc)
        
        # Compute E[log θ_{dk}] for all topics
        gamma_sum = self.gamma[d, :].sum()
        E_log_theta = self._digamma(self.gamma[d, :]) - self._digamma(gamma_sum)
        
        # Update φ for each word in document
        for n, word_id in enumerate(doc):
            if word_id >= self.V:
                continue
            
            # Compute E[log φ_{kw_{dn}}] for all topics
            E_log_phi = self._digamma(self.lambda_param[:, word_id]) - \
                       self._digamma(self.lambda_param.sum(axis=1))
            
            # Compute unnormalized φ
            log_phi_dn = E_log_theta + E_log_phi
            
            # Normalize (softmax)
            # Subtract max for numerical stability
            log_phi_dn = log_phi_dn - log_phi_dn.max()
            phi_dn = np.exp(log_phi_dn)
            phi_dn = phi_dn / (phi_dn.sum() + 1e-10)
            
            self.phi[d][n, :] = phi_dn
    
    def _update_gamma(self, d: int, documents: List[List[int]]):
        """
        Update document-topic variational parameters γ for document d.
        
        γ_{dk} = α_k + Σ_n φ_{dnk}
        
        Args:
            d: Document index
            documents: List of all documents
        """
        doc = documents[d]
        # Sum over all words in document
        self.gamma[d, :] = self.alpha + self.phi[d].sum(axis=0)
    
    def _update_lambda(self, documents: List[List[int]]):
        """
        Update topic-word variational parameters λ.
        
        λ_{kv} = β_v + Σ_d Σ_n φ_{dnk} * [w_{dn} = v]
        
        Args:
            documents: List of all documents
        """
        # Reset λ
        self.lambda_param = np.full((self.K, self.V), self.beta)
        
        # Accumulate counts from all documents
        for d, doc in enumerate(documents):
            for n, word_id in enumerate(doc):
                if word_id >= self.V:
                    continue
                # Add φ_{dnk} to λ_{kw_{dn}} for all topics k
                self.lambda_param[:, word_id] += self.phi[d][n, :]
    
    def _compute_elbo(self, documents: List[List[int]]) -> float:
        """
        Compute Evidence Lower BOund (ELBO).
        
        ELBO = E[log p(w|z,φ)] + E[log p(z|θ)] + E[log p(θ|α)] + 
               E[log p(φ|β)] - E[log q(θ)] - E[log q(φ)] - E[log q(z)]
        
        Args:
            documents: List of all documents
            
        Returns:
            ELBO value
        """
        from scipy.special import gammaln
        
        elbo = 0.0
        
        # E[log p(w|z,φ)] + E[log p(z|θ)]
        for d, doc in enumerate(documents):
            for n, word_id in enumerate(doc):
                if word_id >= self.V:
                    continue
                
                # E[log p(w_{dn}|z_{dn}, φ)] = Σ_k φ_{dnk} * E[log φ_{kw_{dn}}]
                E_log_phi_kw = self._digamma(self.lambda_param[:, word_id]) - \
                              self._digamma(self.lambda_param.sum(axis=1))
                elbo += np.sum(self.phi[d][n, :] * E_log_phi_kw)
                
                # E[log p(z_{dn}|θ_d)] = Σ_k φ_{dnk} * E[log θ_{dk}]
                gamma_sum = self.gamma[d, :].sum()
                E_log_theta_d = self._digamma(self.gamma[d, :]) - self._digamma(gamma_sum)
                elbo += np.sum(self.phi[d][n, :] * E_log_theta_d)
        
        # E[log p(θ|α)] - E[log q(θ)]
        for d in range(self.D):
            gamma_sum = self.gamma[d, :].sum()
            elbo += gammaln(self.K * self.alpha) - np.sum(gammaln(self.alpha))
            elbo -= gammaln(gamma_sum) - np.sum(gammaln(self.gamma[d, :]))
            elbo += np.sum((self.alpha - self.gamma[d, :]) * 
                          (self._digamma(self.gamma[d, :]) - self._digamma(gamma_sum)))
        
        # E[log p(φ|β)] - E[log q(φ)]
        for k in range(self.K):
            lambda_sum = self.lambda_param[k, :].sum()
            elbo += gammaln(self.V * self.beta) - np.sum(gammaln(self.beta))
            elbo -= gammaln(lambda_sum) - np.sum(gammaln(self.lambda_param[k, :]))
            elbo += np.sum((self.beta - self.lambda_param[k, :]) *
                          (self._digamma(self.lambda_param[k, :]) - self._digamma(lambda_sum)))
        
        # -E[log q(z)]
        for d, doc in enumerate(documents):
            for n in range(len(doc)):
                # Entropy of multinomial: -Σ_k φ_{dnk} * log φ_{dnk}
                phi_dn = self.phi[d][n, :]
                elbo += np.sum(phi_dn * np.log(phi_dn + 1e-10))
        
        return elbo
    
    def fit(self, 
            documents: List[List[int]], 
            vocab_size: int,
            num_iterations: int = 100,
            convergence_threshold: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Train LDA model using variational inference.
        
        Args:
            documents: List of documents (word ID lists)
            vocab_size: Vocabulary size
            num_iterations: Maximum number of iterations
            convergence_threshold: Convergence threshold for ELBO change
            
        Returns:
            Tuple of (theta, phi, elbo_history):
                - theta: Document-topic distributions (D x K)
                - phi: Topic-word distributions (K x V)
                - elbo_history: List of ELBO values per iteration
        """
        # Initialize
        self._initialize(documents, vocab_size)
        
        prev_elbo = -np.inf
        elbo_history = []
        
        # Coordinate ascent iterations
        for iteration in range(num_iterations):
            # Update variational parameters
            for d in range(self.D):
                self._update_phi(d, documents)
                self._update_gamma(d, documents)
            
            self._update_lambda(documents)
            
            # Compute ELBO
            elbo = self._compute_elbo(documents)
            elbo_history.append(elbo)
            elbo_change = elbo - prev_elbo
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, ELBO: {elbo:.2f}, Change: {elbo_change:.4f}")
            
            # Check convergence
            if abs(elbo_change) < convergence_threshold and iteration > 10:
                print(f"Converged at iteration {iteration + 1}")
                break
            
            prev_elbo = elbo
        
        # Estimate posterior parameters
        self._estimate_parameters()
        
        return self.theta, self.phi_est, elbo_history
    
    def _estimate_parameters(self):
        """
        Estimate document-topic and topic-word distributions from
        variational parameters.
        
        θ_{dk} = γ_{dk} / sum_k γ_{dk}
        φ_{kv} = λ_{kv} / sum_v λ_{kv}
        """
        # Document-topic distributions
        self.theta = self.gamma / self.gamma.sum(axis=1, keepdims=True)
        
        # Topic-word distributions
        self.phi_est = self.lambda_param / self.lambda_param.sum(axis=1, keepdims=True)
    
    def infer_topics(self, document: List[int], num_iterations: int = 20) -> np.ndarray:
        """
        Infer topic distribution for a new document.
        
        Args:
            document: List of word IDs
            num_iterations: Number of coordinate ascent iterations
            
        Returns:
            Topic distribution θ for the new document
        """
        # Initialize variational parameters for new document
        gamma_new = np.random.gamma(100, 1.0/100, size=self.K) + self.alpha
        phi_new = np.random.dirichlet([1.0/self.K] * self.K, size=len(document))
        
        # Coordinate ascent
        for iteration in range(num_iterations):
            # Update φ
            gamma_sum = gamma_new.sum()
            E_log_theta = self._digamma(gamma_new) - self._digamma(gamma_sum)
            
            for n, word_id in enumerate(document):
                if word_id >= self.V:
                    continue
                
                E_log_phi = self._digamma(self.lambda_param[:, word_id]) - \
                           self._digamma(self.lambda_param.sum(axis=1))
                
                log_phi_n = E_log_theta + E_log_phi
                log_phi_n = log_phi_n - log_phi_n.max()
                phi_n = np.exp(log_phi_n)
                phi_n = phi_n / (phi_n.sum() + 1e-10)
                phi_new[n, :] = phi_n
            
            # Update γ
            gamma_new = self.alpha + phi_new.sum(axis=0)
        
        # Return normalized topic distribution
        theta_new = gamma_new / gamma_new.sum()
        return theta_new


if __name__ == "__main__":
    print("LDA Variational Inference initialized")
    print("Mathematical formulation:")
    print("  Variational: q(θ) = Dir(γ), q(φ) = Dir(λ), q(z) = Mult(φ)")
    print("  Updates: γ_{dk} = α_k + Σ_n φ_{dnk}")
    print("           λ_{kv} = β_v + Σ_d Σ_n φ_{dnk} * [w_{dn} = v]")

