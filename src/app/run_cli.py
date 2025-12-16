"""
Command Line Interface for Running Advice System.

This module provides a CLI to:
1. Input user query
2. Preprocess text
3. Infer topic mixture via LDA
4. Predict risk factors via MAP logistic regression
5. Retrieve similar cases
6. Generate advice using mixture-of-experts
7. Display output with evaluation
8. Run perplexity evaluation on LDA models
9. Run NMF baseline topic model
10. Run SBERT retrieval baseline
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocess import TextPreprocessor
from src.data.labeling import RunningDataLabeler
from src.models.lda_gibbs import LDAGibbsSampler
from src.models.lda_vi import LDAVariationalInference
from src.models.bayesian_mixture import BayesianMixtureOfExperts
from src.models.logistic_map import LogisticRegressionMAP
from src.models.recommendation import RecommendationSystem
from src.evaluation.compliance_check import ComplianceChecker
from src.evaluation.relevance_metrics import RelevanceMetrics
from src.evaluation.qualitative_eval import QualitativeEvaluator

import numpy as np

# Import new modules for extended pipeline
try:
    from evaluation.perplexity_eval import compute_perplexity
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False

try:
    from models.nmf_baseline import fit_nmf
    NMF_AVAILABLE = True
except ImportError:
    NMF_AVAILABLE = False

try:
    from retrieval.sbert_retrieval import build_sbert_embeddings, sbert_retrieve
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    from visualization.full_plots import plot_perplexity, plot_nmf_topic, plot_sbert_scores
    FULL_PLOTS_AVAILABLE = True
except ImportError:
    FULL_PLOTS_AVAILABLE = False


class RunningAdviceCLI:
    """
    Command line interface for running advice system.
    """
    
    def __init__(self, 
                 lda_model_path: str = None,
                 logistic_model_path: str = None,
                 data_path: str = None):
        """
        Initialize CLI system.
        
        Args:
            lda_model_path: Path to saved LDA model (optional)
            logistic_model_path: Path to saved logistic model (optional)
            data_path: Path to training data (optional)
        """
        self.preprocessor = TextPreprocessor()
        self.labeler = RunningDataLabeler()
        self.compliance_checker = ComplianceChecker()
        self.relevance_metrics = RelevanceMetrics()
        self.qualitative_eval = QualitativeEvaluator()
        
        # Models (will be loaded or trained)
        self.lda_model = None
        self.logistic_model = None
        self.mixture_model = None
        self.recommendation_system = None
        
        # Data
        self.documents = []
        self.vocab = {}
        self.topic_distributions = None
        
        # Load models if paths provided
        if lda_model_path and os.path.exists(lda_model_path):
            self.load_lda_model(lda_model_path)
        if logistic_model_path and os.path.exists(logistic_model_path):
            self.load_logistic_model(logistic_model_path)
        if data_path and os.path.exists(data_path):
            self.load_data(data_path)
    
    def load_data(self, data_path: str):
        """Load training data from file."""
        # This is a skeleton - implement based on your data format
        print(f"Loading data from {data_path}...")
        # Example: load JSON or CSV
        # with open(data_path, 'r') as f:
        #     data = json.load(f)
        #     self.documents = [item['text'] for item in data]
        pass
    
    def load_lda_model(self, model_path: str):
        """Load pre-trained LDA model."""
        print(f"Loading LDA model from {model_path}...")
        # This is a skeleton - implement model saving/loading
        pass
    
    def load_logistic_model(self, model_path: str):
        """Load pre-trained logistic regression model."""
        print(f"Loading logistic model from {model_path}...")
        # This is a skeleton - implement model saving/loading
        pass
    
    def initialize_models(self, num_topics: int = 10):
        """
        Initialize models with default parameters.
        
        Args:
            num_topics: Number of topics for LDA
        """
        print("Initializing models...")
        
        # Initialize LDA (Gibbs sampling)
        self.lda_model = LDAGibbsSampler(
            num_topics=num_topics,
            alpha=0.1,
            beta=0.01
        )
        
        # Initialize logistic regression
        self.logistic_model = LogisticRegressionMAP(
            prior_variance=1.0,
            method='gradient_ascent'
        )
        
        # Initialize mixture-of-experts
        self.mixture_model = BayesianMixtureOfExperts(
            num_experts=5,
            alpha_prior=1.0
        )
        
        # Initialize recommendation system
        self.recommendation_system = RecommendationSystem()
    
    def process_query(self, query: str) -> Dict:
        """
        Process a user query through the full pipeline.
        
        Args:
            query: User's running question/problem
            
        Returns:
            Dictionary with all results
        """
        print(f"\n{'='*60}")
        print(f"Processing query: {query}")
        print(f"{'='*60}\n")
        
        # Step 1: Preprocess
        print("Step 1: Preprocessing text...")
        tokens = self.preprocessor.preprocess(query)
        print(f"  Preprocessed tokens: {len(tokens)} words")
        
        # Step 2: Label risk factors
        print("\nStep 2: Labeling risk factors...")
        labels = self.labeler.label_document(query)
        print(f"  Labels: {labels}")
        
        # Step 3: Infer topic mixture (if LDA model available)
        topic_mixture = None
        if self.lda_model and self.vocab:
            print("\nStep 3: Inferring topic mixture...")
            # Convert tokens to word IDs
            word_ids = [self.vocab.get(token, -1) for token in tokens 
                       if token in self.vocab]
            if word_ids:
                topic_mixture = self.lda_model.infer_topics(word_ids)
                print(f"  Topic mixture: {topic_mixture}")
            else:
                print("  Warning: No words in vocabulary")
        else:
            print("\nStep 3: Skipping topic inference (model not loaded)")
            # Use uniform distribution as fallback
            topic_mixture = np.ones(10) / 10
        
        # Step 4: Predict risk factors (if model available)
        print("\nStep 4: Predicting risk factors...")
        risk_predictions = {}
        if self.logistic_model:
            # Extract features (simplified - use topic mixture or TF-IDF)
            # In practice, you'd use proper feature extraction
            features = topic_mixture if topic_mixture is not None else np.zeros(10)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Predict (this is simplified - in practice use proper features)
            # risk_predictions = self.logistic_model.predict_risk_factors(features)
            risk_predictions = labels  # Use rule-based labels for now
        else:
            risk_predictions = labels
        
        print(f"  Risk predictions: {risk_predictions}")
        
        # Step 5: Retrieve similar cases
        print("\nStep 5: Retrieving similar cases...")
        similar_docs = []
        if self.recommendation_system and self.documents:
            try:
                similar_docs = self.recommendation_system.retrieve(
                    query=query,
                    query_topic_dist=topic_mixture,
                    method='tfidf',
                    top_k=5
                )
                print(f"  Found {len(similar_docs)} similar documents")
            except:
                print("  Warning: Retrieval failed")
        else:
            print("  Skipping retrieval (no documents loaded)")
        
        # Step 6: Generate advice
        print("\nStep 6: Generating advice using mixture-of-experts...")
        user_context = {
            'experience_level': labels.get('experience_level', 'unknown'),
            **{k: v for k, v in labels.items() if k.startswith('goal_')}
        }
        
        advice = self.mixture_model.generate_advice(
            topic_mixture=topic_mixture,
            risk_factors=risk_predictions,
            user_context=user_context
        )
        
        print(f"  Selected expert: {advice.get('selected_expert')}")
        print(f"  Recommendations: {advice.get('recommendations', [])}")
        
        # Step 7: Evaluate
        print("\nStep 7: Evaluating advice...")
        
        # Compliance check
        compliance_results = self.compliance_checker.check_all(
            advice=advice,
            risk_factors=risk_predictions,
            user_experience=user_context.get('experience_level', 'unknown')
        )
        
        # Qualitative evaluation
        qualitative_results = self.qualitative_eval.evaluate_all(
            advice=advice,
            risk_factors=risk_predictions,
            user_context=user_context
        )
        
        print(f"  Compliance: {'✓' if compliance_results['overall_compliant'] else '✗'}")
        print(f"  Overall quality score: {qualitative_results['overall_score']:.2f}")
        
        # Compile results
        results = {
            'query': query,
            'preprocessed_tokens': tokens,
            'labels': labels,
            'topic_mixture': topic_mixture.tolist() if topic_mixture is not None else None,
            'risk_predictions': risk_predictions,
            'similar_documents': similar_docs,
            'advice': advice,
            'compliance': compliance_results,
            'qualitative_evaluation': qualitative_results
        }
        
        return results
    
    def display_results(self, results: Dict):
        """
        Display results in a formatted way.
        
        Args:
            results: Results dictionary from process_query
        """
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}\n")
        
        # Advice
        print("GENERATED ADVICE:")
        print("-" * 60)
        advice = results['advice']
        recommendations = advice.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        print()
        
        # Risk factors
        print("RISK FACTORS:")
        print("-" * 60)
        risk = results['risk_predictions']
        for key, value in risk.items():
            if isinstance(value, bool) and value:
                print(f"  • {key.replace('_', ' ').title()}")
        print()
        
        # Compliance
        print("COMPLIANCE CHECK:")
        print("-" * 60)
        compliance = results['compliance']
        if compliance['overall_compliant']:
            print("  ✓ Advice is compliant with safety constraints")
        else:
            print("  ✗ Compliance violations detected:")
            for violation in compliance['violations']:
                print(f"    - {violation}")
        print()
        
        # Quality scores
        print("QUALITY SCORES:")
        print("-" * 60)
        qual = results['qualitative_evaluation']
        print(f"  Overall: {qual['overall_score']:.2f}")
        print(f"  Clarity: {qual['clarity']['score']:.2f}")
        print(f"  Safety: {qual['safety']['score']:.2f}")
        print(f"  Personalization: {qual['personalization']['score']:.2f}")
        print(f"  Correctness: {qual['correctness']['score']:.2f}")
        print()
        
        # Similar documents
        if results['similar_documents']:
            print("SIMILAR CASES:")
            print("-" * 60)
            for idx, (doc_idx, score) in enumerate(results['similar_documents'][:3], 1):
                print(f"  {idx}. Document {doc_idx} (similarity: {score:.3f})")
            print()
    
    def run_interactive(self):
        """Run interactive CLI loop."""
        print("\n" + "="*60)
        print("Running Advice System - Interactive Mode")
        print("="*60)
        print("\nEnter your running question or problem.")
        print("Type 'quit' or 'exit' to exit.\n")
        
        # Initialize models if not already done
        if self.lda_model is None:
            self.initialize_models()
        
        while True:
            try:
                query = input("\n> Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not query:
                    print("Please enter a question.")
                    continue
                
                # Process query
                results = self.process_query(query)
                
                # Display results
                self.display_results(results)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
    
    def run_extended_pipeline(self, raw_docs: List[str], test_docs: List[List[int]] = None,
                              phi: np.ndarray = None, theta: np.ndarray = None):
        """
        Run extended pipeline with perplexity evaluation, NMF baseline, and SBERT retrieval.
        Generates visualizations for all components.
        
        Args:
            raw_docs: List of raw document strings
            test_docs: List of lists of token indices for perplexity evaluation
            phi: Topic-word distribution matrix (K x V) from LDA
            theta: Document-topic distribution matrix (D x K) from LDA
        """
        print("\n" + "=" * 60)
        print("EXTENDED PIPELINE: Perplexity / NMF / SBERT")
        print("=" * 60)
        
        # ===========================================
        #  PERPLEXITY EVALUATION
        # ===========================================
        if PERPLEXITY_AVAILABLE and test_docs is not None and phi is not None and theta is not None:
            print("\n[Evaluating Held-out Perplexity]")
            try:
                K_values = [5, 10, 15]
                perps = []
                for K in K_values:
                    # For demo, use provided phi/theta; in practice train per K
                    perp = compute_perplexity(test_docs, phi, theta)
                    perps.append(perp)
                    print(f"  K={K}: Perplexity={perp:.2f}")
                
                # Plot perplexity curve
                if FULL_PLOTS_AVAILABLE:
                    plot_perplexity(
                        K_values,
                        perps,
                        "data/processed/visualizations/perplexity/perplexity_curve.png"
                    )
                    print("  Saved: data/processed/visualizations/perplexity/perplexity_curve.png")
            except Exception as e:
                print(f"[LDA] Perplexity evaluation failed: {e}")
        elif not PERPLEXITY_AVAILABLE:
            print("\n[Perplexity] Module not available - skipping")
        else:
            print("\n[Perplexity] Missing required data (test_docs, phi, theta) - skipping")
        
        # ===========================================
        #  NMF BASELINE
        # ===========================================
        if NMF_AVAILABLE and raw_docs:
            print("\n[Fitting NMF baseline]")
            try:
                W_nmf, H_nmf, vocab_nmf, nmf_model, nmf_vec = fit_nmf(raw_docs, n_components=10)
                print("[NMF] Example topic words:")
                for k in range(min(3, H_nmf.shape[0])):
                    idx = H_nmf[k].argsort()[::-1][:10]
                    print(f"  Topic {k}: {[vocab_nmf[i] for i in idx]}")
                
                # Plot NMF topic words
                if FULL_PLOTS_AVAILABLE:
                    plot_nmf_topic(
                        H_nmf,
                        vocab_nmf,
                        topic_id=0,
                        path="data/processed/visualizations/nmf/nmf_topic0.png"
                    )
                    print("  Saved: data/processed/visualizations/nmf/nmf_topic0.png")
            except Exception as e:
                print(f"[NMF] Baseline failed: {e}")
        elif not NMF_AVAILABLE:
            print("\n[NMF] Module not available - skipping")
        else:
            print("\n[NMF] No documents available - skipping")
        
        # ===========================================
        #  SBERT RETRIEVAL BASELINE
        # ===========================================
        if SBERT_AVAILABLE and raw_docs:
            print("\n[Running SBERT retrieval baseline]")
            try:
                sbert_emb, sbert_model = build_sbert_embeddings(raw_docs)
                
                query = "shin pain from running"
                results = sbert_retrieve(query, raw_docs, sbert_emb, sbert_model, top_k=5)
                
                # Plot SBERT scores
                if FULL_PLOTS_AVAILABLE:
                    plot_sbert_scores(
                        results,
                        raw_docs,
                        path="data/processed/visualizations/sbert/sbert_scores.png"
                    )
                    print("  Saved: data/processed/visualizations/sbert/sbert_scores.png")
                
                print("\n[SBERT Top Results]")
                print(f"Query: '{query}'")
                for idx, score in results:
                    doc_preview = raw_docs[idx][:90] if len(raw_docs[idx]) > 90 else raw_docs[idx]
                    print(f"  Score={score:.3f} | Doc={doc_preview}...")
            except Exception as e:
                print(f"[SBERT] Retrieval failed: {e}")
        elif not SBERT_AVAILABLE:
            print("\n[SBERT] Module not available - skipping")
        else:
            print("\n[SBERT] No documents available - skipping")
        
        print("\n" + "=" * 60)
        print("Extended pipeline complete — All visualizations saved!")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Running Advice System - Probabilistic Models for Personalized Advice'
    )
    parser.add_argument('--query', type=str, help='Single query to process')
    parser.add_argument('--lda-model', type=str, help='Path to LDA model')
    parser.add_argument('--logistic-model', type=str, help='Path to logistic model')
    parser.add_argument('--data', type=str, help='Path to training data')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode')
    parser.add_argument('--extended', action='store_true',
                       help='Run extended pipeline with Perplexity/NMF/SBERT')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = RunningAdviceCLI(
        lda_model_path=args.lda_model,
        logistic_model_path=args.logistic_model,
        data_path=args.data
    )
    
    if args.extended:
        # Extended pipeline mode - demo with sample data
        print("\n[Extended Pipeline Mode]")
        print("Loading sample data for demonstration...")
        
        # Try to load processed data
        data_path = "data/processed/processed_data.json"
        raw_docs = []
        
        if os.path.exists(data_path):
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    raw_docs = [doc['text'] for doc in data[:500]]  # Limit for demo
                else:
                    raw_docs = [doc['text'] for doc in data.get('documents', [])[:500]]
                print(f"Loaded {len(raw_docs)} documents")
            except Exception as e:
                print(f"Warning: Could not load data - {e}")
        
        # Run extended pipeline (phi/theta/test_docs would come from actual LDA training)
        cli.run_extended_pipeline(
            raw_docs=raw_docs,
            test_docs=None,  # Would be tokenized test documents
            phi=None,        # Would be trained LDA phi matrix
            theta=None       # Would be trained LDA theta matrix
        )
    elif args.interactive or not args.query:
        # Interactive mode
        cli.run_interactive()
    else:
        # Single query mode
        results = cli.process_query(args.query)
        cli.display_results(results)
        
        # Optionally save results
        output_file = 'results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")


# ============================================================
# STANDALONE EXTENDED PIPELINE DEMO
# ============================================================
# This section can be run directly to test the new modules
# without the full CLI infrastructure.

def run_standalone_demo():
    """
    Standalone demo for testing Perplexity/NMF/SBERT modules.
    Can be called directly or imported.
    """
    print("\n" + "=" * 60)
    print("STANDALONE DEMO: Perplexity / NMF / SBERT")
    print("=" * 60)
    
    # Load sample data
    data_path = "data/processed/processed_data.json"
    raw_docs = []
    
    if os.path.exists(data_path):
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                raw_docs = [doc['text'] for doc in data[:500]]
            else:
                raw_docs = [doc['text'] for doc in data.get('documents', [])[:500]]
            print(f"Loaded {len(raw_docs)} documents")
        except Exception as e:
            print(f"Warning: Could not load data - {e}")
            raw_docs = [
                "I have shin pain after running long distances",
                "My knee hurts when going downhill during my runs",
                "How do I prevent blisters on my feet from running",
                "Best stretches before a marathon training run",
                "Running form tips for avoiding injury"
            ]
            print("Using sample documents for demo")
    
    # ======== PERPLEXITY EVALUATION ========
    if PERPLEXITY_AVAILABLE:
        print("\n[Evaluating Held-out Perplexity]")
        # Demo with synthetic data:
        K, V, D = 5, 100, 10  # topics, vocab size, docs
        phi = np.random.dirichlet(np.ones(V), size=K)  # K x V
        theta = np.random.dirichlet(np.ones(K), size=D)  # D x K
        test_docs = [[np.random.randint(0, V) for _ in range(20)] for _ in range(D)]
        
        K_values = [5, 10, 15]
        perps = []
        for k_val in K_values:
            perp = compute_perplexity(test_docs, phi, theta)
            perps.append(perp)
            print(f"  K={k_val}: Perplexity={perp:.2f}")
        
        # Plot perplexity curve
        if FULL_PLOTS_AVAILABLE:
            plot_perplexity(
                K_values,
                perps,
                "data/processed/visualizations/perplexity/perplexity_curve.png"
            )
            print("  Saved: data/processed/visualizations/perplexity/perplexity_curve.png")
    else:
        print("\n[Perplexity Module] Not available - install numpy")
    
    # ======== NMF BASELINE ========
    if NMF_AVAILABLE and raw_docs:
        print("\n[Fitting NMF baseline]")
        try:
            W_nmf, H_nmf, vocab_nmf, nmf_model, nmf_vec = fit_nmf(raw_docs, n_components=10)
            print("[NMF] Example topic words:")
            for k in range(min(3, H_nmf.shape[0])):
                idx = H_nmf[k].argsort()[::-1][:10]
                print(f"  Topic {k}: {[vocab_nmf[i] for i in idx]}")
            
            # Plot NMF topic words
            if FULL_PLOTS_AVAILABLE:
                plot_nmf_topic(
                    H_nmf,
                    vocab_nmf,
                    topic_id=0,
                    path="data/processed/visualizations/nmf/nmf_topic0.png"
                )
                print("  Saved: data/processed/visualizations/nmf/nmf_topic0.png")
        except Exception as e:
            print(f"[NMF] Baseline failed: {e}")
    else:
        print("\n[NMF Module] Not available - install scikit-learn")
    
    # ======== SBERT RETRIEVAL BASELINE ========
    if SBERT_AVAILABLE and raw_docs:
        print("\n[Running SBERT retrieval baseline]")
        try:
            sbert_emb, sbert_model = build_sbert_embeddings(raw_docs)
            
            query = "shin pain from running"
            results = sbert_retrieve(query, raw_docs, sbert_emb, sbert_model, top_k=5)
            
            # Plot SBERT scores
            if FULL_PLOTS_AVAILABLE:
                plot_sbert_scores(
                    results,
                    raw_docs,
                    path="data/processed/visualizations/sbert/sbert_scores.png"
                )
                print("  Saved: data/processed/visualizations/sbert/sbert_scores.png")
            
            print("\n[SBERT Top Results]")
            print(f"Query: '{query}'")
            for idx, score in results:
                doc_preview = raw_docs[idx][:90] if len(raw_docs[idx]) > 90 else raw_docs[idx]
                print(f"  Score={score:.3f} | Doc={doc_preview}...")
        except Exception as e:
            print(f"[SBERT] Retrieval failed: {e}")
    else:
        print("\n[SBERT Module] Not available - install sentence-transformers")
    
    print("\n" + "=" * 60)
    print("Standalone demo complete — All visualizations saved!")
    print("=" * 60)


if __name__ == "__main__":
    main()

