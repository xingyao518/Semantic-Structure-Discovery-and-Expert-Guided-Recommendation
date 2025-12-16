import os
import numpy as np
import matplotlib.pyplot as plt


def ensure(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# -----------------------------
# LDA Perplexity Curve
# -----------------------------
def plot_perplexity(K_list, perp_list, path):
    ensure(path)
    plt.figure(figsize=(7,4))
    plt.plot(K_list, perp_list, marker="o")
    plt.xlabel("Number of Topics (K)")
    plt.ylabel("Perplexity")
    plt.title("LDA Held-out Perplexity vs K")
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# NMF Topic Words
# -----------------------------
def plot_nmf_topic(H, vocab, topic_id, path, top_n=12):
    ensure(path)
    idx = np.argsort(H[topic_id])[::-1][:top_n]
    words = [vocab[i] for i in idx]
    vals = H[topic_id][idx]

    plt.figure(figsize=(8,5))
    plt.barh(words[::-1], vals[::-1])
    plt.title(f"NMF Topic {topic_id}: Top Words")
    plt.xlabel("Weight")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# SBERT Similarity Bar Chart
# -----------------------------
def plot_sbert_scores(results, docs, path):
    ensure(path)
    labels = [f"Doc {i}" for i,_ in results]
    scores = [score for _,score in results]

    plt.figure(figsize=(7,4))
    plt.bar(labels, scores)
    plt.title("SBERT Retrieval: Similarity Scores")
    plt.ylabel("Cosine Similarity")
    plt.xticks(rotation=30)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


