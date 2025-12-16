"""
Topic Coherence Evaluation Module.

Implements UMass and PMI-based coherence metrics for evaluating topic quality.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import math


def compute_word_cooccurrence(corpus: List[List[str]], window_size: int = 10) -> Tuple[Dict, Dict, int]:
    """
    Compute word co-occurrence statistics from corpus.
    
    Args:
        corpus: List of tokenized documents
        window_size: Window size for co-occurrence (for PMI)
        
    Returns:
        Tuple of (word_doc_counts, word_pair_doc_counts, num_docs)
    """
    word_doc_counts = Counter()  # D(w): number of docs containing word w
    word_pair_doc_counts = Counter()  # D(w1, w2): docs containing both words
    num_docs = len(corpus)
    
    for doc in corpus:
        doc_words = set(doc)
        for word in doc_words:
            word_doc_counts[word] += 1
        
        # Count co-occurrences within document
        doc_words_list = list(doc_words)
        for i in range(len(doc_words_list)):
            for j in range(i + 1, len(doc_words_list)):
                pair = tuple(sorted([doc_words_list[i], doc_words_list[j]]))
                word_pair_doc_counts[pair] += 1
    
    return word_doc_counts, word_pair_doc_counts, num_docs


def compute_umass_coherence(top_words: List[str], 
                            word_doc_counts: Dict,
                            word_pair_doc_counts: Dict,
                            epsilon: float = 1e-12) -> float:
    """
    Compute UMass coherence for a single topic.
    
    UMass coherence: C_UMass = (2 / (M*(M-1))) * Σ_{m=2}^M Σ_{l=1}^{m-1} log((D(w_m, w_l) + ε) / D(w_l))
    
    Args:
        top_words: List of top words for the topic
        word_doc_counts: Word document frequency counts
        word_pair_doc_counts: Word pair document frequency counts
        epsilon: Smoothing parameter
        
    Returns:
        UMass coherence score
    """
    M = len(top_words)
    if M < 2:
        return 0.0
    
    coherence = 0.0
    count = 0
    
    for m in range(1, M):
        for l in range(m):
            w_m = top_words[m]
            w_l = top_words[l]
            
            D_wl = word_doc_counts.get(w_l, 0)
            if D_wl == 0:
                continue
            
            pair = tuple(sorted([w_m, w_l]))
            D_wm_wl = word_pair_doc_counts.get(pair, 0)
            
            coherence += math.log((D_wm_wl + epsilon) / D_wl)
            count += 1
    
    if count == 0:
        return 0.0
    
    return coherence / count


def compute_pmi_coherence(top_words: List[str],
                          word_doc_counts: Dict,
                          word_pair_doc_counts: Dict,
                          num_docs: int,
                          epsilon: float = 1e-12) -> float:
    """
    Compute PMI-based coherence for a single topic.
    
    PMI coherence: C_PMI = (2 / (M*(M-1))) * Σ_{m=2}^M Σ_{l=1}^{m-1} log((D(w_m, w_l) * N + ε) / (D(w_m) * D(w_l)))
    
    Args:
        top_words: List of top words for the topic
        word_doc_counts: Word document frequency counts
        word_pair_doc_counts: Word pair document frequency counts
        num_docs: Total number of documents
        epsilon: Smoothing parameter
        
    Returns:
        PMI coherence score
    """
    M = len(top_words)
    if M < 2:
        return 0.0
    
    coherence = 0.0
    count = 0
    
    for m in range(1, M):
        for l in range(m):
            w_m = top_words[m]
            w_l = top_words[l]
            
            D_wm = word_doc_counts.get(w_m, 0)
            D_wl = word_doc_counts.get(w_l, 0)
            
            if D_wm == 0 or D_wl == 0:
                continue
            
            pair = tuple(sorted([w_m, w_l]))
            D_wm_wl = word_pair_doc_counts.get(pair, 0)
            
            pmi = math.log((D_wm_wl * num_docs + epsilon) / (D_wm * D_wl + epsilon))
            coherence += pmi
            count += 1
    
    if count == 0:
        return 0.0
    
    return coherence / count


def compute_topic_coherence(phi: np.ndarray,
                           vocab: Dict,
                           corpus: List[List[str]],
                           top_n: int = 10,
                           method: str = 'umass') -> Tuple[float, List[float]]:
    """
    Compute topic coherence for all topics.
    
    Args:
        phi: Topic-word distribution matrix (K x V)
        vocab: Vocabulary dictionary (word -> id or id -> word)
        corpus: List of tokenized documents
        top_n: Number of top words per topic
        method: Coherence method ('umass' or 'pmi')
        
    Returns:
        Tuple of (average_coherence, per_topic_coherence)
    """
    # Convert vocab to id->word if needed
    if vocab and isinstance(vocab, dict):
        first_key = next(iter(vocab.keys()))
        if isinstance(first_key, str):
            id_to_word = {v: k for k, v in vocab.items()}
        else:
            id_to_word = vocab
    else:
        id_to_word = {}
    
    # Compute co-occurrence statistics
    word_doc_counts, word_pair_doc_counts, num_docs = compute_word_cooccurrence(corpus)
    
    K = phi.shape[0]
    per_topic_coherence = []
    
    for k in range(K):
        # Get top words for topic k
        top_indices = np.argsort(phi[k, :])[::-1][:top_n]
        top_words = [id_to_word.get(idx, f"word_{idx}") for idx in top_indices]
        
        if method == 'umass':
            coherence = compute_umass_coherence(
                top_words, word_doc_counts, word_pair_doc_counts
            )
        elif method == 'pmi':
            coherence = compute_pmi_coherence(
                top_words, word_doc_counts, word_pair_doc_counts, num_docs
            )
        else:
            raise ValueError(f"Unknown coherence method: {method}")
        
        per_topic_coherence.append(coherence)
    
    avg_coherence = np.mean(per_topic_coherence)
    return avg_coherence, per_topic_coherence


def compute_held_out_log_likelihood(phi: np.ndarray,
                                    theta: np.ndarray,
                                    held_out_docs: List[List[int]]) -> float:
    """
    Compute held-out log-likelihood for LDA model.
    
    Args:
        phi: Topic-word distributions (K x V)
        theta: Document-topic distributions (D x K)
        held_out_docs: List of held-out documents (word ID lists)
        
    Returns:
        Average held-out log-likelihood per word
    """
    total_ll = 0.0
    total_words = 0
    
    K, V = phi.shape
    
    for d, doc in enumerate(held_out_docs):
        if d >= len(theta):
            break
            
        for word_id in doc:
            if word_id >= V:
                continue
            
            # P(w) = sum_k theta[d,k] * phi[k, word_id]
            prob = np.sum(theta[d, :] * phi[:, word_id])
            if prob > 0:
                total_ll += np.log(prob)
            total_words += 1
    
    if total_words == 0:
        return 0.0
    
    return total_ll / total_words


if __name__ == "__main__":
    print("Topic Coherence Evaluation Module")
    print("Functions:")
    print("  - compute_umass_coherence()")
    print("  - compute_pmi_coherence()")
    print("  - compute_topic_coherence()")
    print("  - compute_held_out_log_likelihood()")

