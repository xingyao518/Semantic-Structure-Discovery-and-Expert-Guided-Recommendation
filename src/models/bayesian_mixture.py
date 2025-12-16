"""
Bayesian Mixture-of-Experts Model for Running Advice Generation.

This module implements a Mixture Model (GMM-like) on LDA topic distributions to
cluster queries into different 'Expert' domains.

Mathematical formulation (EM Algorithm):
- Model: P(x) = sum_k pi_k P(x|k)
- Observation: x_n is the topic mixture vector for document n.
- Likelihood: P(x|k) ~ N(mu_k, sigma^2 I) (Gaussian on simplex)
- E-Step: r_{nk} propto pi_k * P(x_n|k)
- M-Step: Update pi_k, mu_k based on r_{nk}
- Log-Likelihood: sum_n log(sum_k pi_k P(x_n|k))
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from scipy.special import logsumexp

class RunningAdviceExpert:
    """
    Individual expert that generates structured running advice.
    """
    
    def __init__(self, expert_id: int, template: Dict):
        self.expert_id = expert_id
        self.template = template
        self.advice_count = 0
    
    def generate_advice(self, context: Dict) -> Dict:
        """Generate advice based on context."""
        advice = {
            'expert_id': self.expert_id,
            'template': self.template.copy(),
            'context': context
        }
        
        # Simple logic to populate advice
        if 'injury_risk' in context and context['injury_risk']:
            advice['recommendations'] = ['Rest', 'Ice', 'Consult professional']
        else:
            advice['recommendations'] = ['Continue training', 'Monitor progress']
            
        self.advice_count += 1
        return advice


class BayesianMixtureOfExperts:
    """
    Mixture Model for clustering queries into experts using EM.
    """
    
    def __init__(self, 
                 num_experts: int = 5,
                 feature_dim: int = 10,
                 random_seed: int = 42):
        """
        Initialize mixture model.
        
        Args:
            num_experts: Number of clusters (experts)
            feature_dim: Dimension of input vector (number of LDA topics)
            random_seed: Random seed
        """
        self.E = num_experts
        self.D = feature_dim
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Experts objects
        self.experts = []
        for e in range(num_experts):
            template = {
                'expert_type': f'expert_{e}',
                'focus_areas': ['general', 'injury', 'nutrition', 'training'][e % 4]
            }
            self.experts.append(RunningAdviceExpert(e, template))
        
        # Parameters
        # Mixing weights pi_k
        self.pi = np.ones(self.E) / self.E
        
        # Means mu_k (E x D)
        # Initialize randomly near center of simplex
        self.mu = np.random.dirichlet(np.ones(self.D), size=self.E)
        
        # Variance (scalar for simplicity, fixed or learned)
        self.sigma_sq = 0.05
        
        # History
        self.loglik_history = []
        self.responsibilities = None # N x E
    
    def _log_gaussian_pdf(self, X: np.ndarray, mean: np.ndarray, var: float) -> np.ndarray:
        """
        Compute log N(x | mean, var*I).
        Output: (N,)
        """
        # const = -D/2 * log(2*pi*var)
        # exp = -1/(2*var) * ||x - mean||^2
        D = X.shape[1]
        const = -0.5 * D * np.log(2 * np.pi * var)
        diff = X - mean
        # Sum squares along dimension 1
        sq_diff = np.sum(diff ** 2, axis=1)
        exp_term = -0.5 / var * sq_diff
        return const + exp_term
        
    def fit(self, 
            topic_mixtures: List[np.ndarray],
            num_iterations: int = 50,
            tolerance: float = 1e-4):
        """
        Train using EM algorithm.
        """
        X = np.array(topic_mixtures)
        N, D = X.shape
        if D != self.D:
            print(f"Warning: Feature dimension mismatch. Expected {self.D}, got {D}. Resizing.")
            self.D = D
            self.mu = np.random.dirichlet(np.ones(self.D), size=self.E)
            
        self.loglik_history = []
        prev_ll = -np.inf
        
        print(f"Training MoE (EM) on {N} samples with {self.E} experts...")
        
        for iteration in range(num_iterations):
            # --- E-Step ---
            # Calculate log probabilities: log(pi_k) + log P(x|k)
            log_res = np.zeros((N, self.E))
            
            for k in range(self.E):
                log_pi = np.log(self.pi[k] + 1e-10)
                log_p_x_k = self._log_gaussian_pdf(X, self.mu[k], self.sigma_sq)
                log_res[:, k] = log_pi + log_p_x_k
            
            # Normalize responsibilities using logsumexp for stability
            # log r_{nk} = log_res_{nk} - logsumexp_k(log_res_{nk})
            log_norm = logsumexp(log_res, axis=1, keepdims=True)
            self.responsibilities = np.exp(log_res - log_norm)
            
            # Compute Log-Likelihood
            # LL = sum_n log sum_k exp(log_res_{nk})
            current_ll = np.sum(log_norm)
            self.loglik_history.append(current_ll)
            
            # --- M-Step ---
            # Update pi_k
            # Nk = sum_n r_{nk}
            Nk = np.sum(self.responsibilities, axis=0)
            self.pi = Nk / N
            
            # Update mu_k
            # mu_k = (1/Nk) sum_n r_{nk} x_n
            # Use small epsilon to avoid division by zero
            for k in range(self.E):
                if Nk[k] > 1e-10:
                    self.mu[k] = np.sum(self.responsibilities[:, k:k+1] * X, axis=0) / Nk[k]
            
            # (Optional) Update sigma_sq
            # We keep it fixed for stability in this simple version or update it
            # sigma_sq_k = (1/ (D * Nk)) sum_n r_{nk} ||x_n - mu_k||^2
            
            # Check convergence
            if abs(current_ll - prev_ll) < tolerance:
                print(f"Converged at iteration {iteration+1}")
                break
            
            prev_ll = current_ll
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}, LL: {current_ll:.2f}")
                
        print(f"Final LL: {self.loglik_history[-1]:.2f}")
        return self.loglik_history

    def generate_advice(self, 
                       topic_mixture: np.ndarray,
                       risk_factors: Dict,
                       user_context: Dict = None) -> Dict:
        """
        Generate advice. Selects expert based on posterior probability.
        """
        # Compute P(k|x)
        x = topic_mixture.reshape(1, -1)
        log_res = np.zeros(self.E)
        for k in range(self.E):
            log_pi = np.log(self.pi[k] + 1e-10)
            log_p_x_k = self._log_gaussian_pdf(x, self.mu[k], self.sigma_sq)
            log_res[k] = log_pi + log_p_x_k
            
        # Normalize
        probs = np.exp(log_res - logsumexp(log_res))
        
        # Sample expert
        expert_idx = np.random.choice(self.E, p=probs)
        expert = self.experts[expert_idx]
        
        context = {
            'topic_mixture': topic_mixture,
            'expert_probs': probs,
            **risk_factors
        }
        if user_context:
            context.update(user_context)
            
        advice = expert.generate_advice(context)
        advice['mixture_weights'] = probs # Return posterior probs as weights
        advice['selected_expert'] = expert_idx
        
        return advice

if __name__ == "__main__":
    print("MoE EM Model initialized")
