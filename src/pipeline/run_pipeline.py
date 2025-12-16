"""
Unified Pipeline Runner for Probabilistic Topic Modeling

This is the main entry point for the entire pipeline.
Run with: python -m src.pipeline.run_pipeline

Pipeline Steps:
  1. Load data
  2. Preprocess (tokenize, build vocab)
  3. Train LDA-Gibbs (or load existing)
  4. Train LDA-VI (or load existing)
  5. Train Mixture Model
  6. Run Recommendation
  7. Generate Visualizations
"""

import sys
import os
import time
import traceback
import argparse
import json
import pickle

# Add project root to path - MUST be done before any other imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import numpy as np


def print_step_complete(step_num, step_name: str):
    """Print step completion message."""
    print("\n" + "=" * 50)
    print(f" COMPLETED: STEP {step_num} - {step_name}")
    print("=" * 50 + "\n")
    sys.stdout.flush()


def print_step_start(step_num, step_name: str):
    """Print step start message."""
    print("\n" + "#" * 50)
    print(f" STARTING: STEP {step_num} - {step_name}")
    print("#" * 50 + "\n")
    sys.stdout.flush()


def print_error(step_name: str, error: Exception):
    """Print error message."""
    print("\n" + "!" * 50)
    print(f" ERROR IN: {step_name}")
    print("!" * 50)
    print(f"Error: {error}")
    traceback.print_exc()
    sys.stdout.flush()


def run_viz_force_mode():
    """
    Visualization-force mode: Load ALL existing artifacts and generate visualizations.
    No preprocessing, no training. Guaranteed to run visualization if files exist.
    """
    start_time = time.time()
    
    print("\n")
    print("=" * 60)
    print("  VISUALIZATION-FORCE MODE")
    print("=" * 60)
    print("\nLoading all existing artifacts...")
    print("(This mode skips ALL preprocessing and training)")
    sys.stdout.flush()
    
    results = {}
    loaded_files = []
    errors = []
    
    # =========================================================================
    # Ensure sys.path includes src for pickle loading
    # =========================================================================
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    src_path = os.path.join(PROJECT_ROOT, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # =========================================================================
    # Load vocab
    # =========================================================================
    print("\n[VIZ-FORCE] Loading vocab...")
    sys.stdout.flush()
    
    try:
        with open("data/processed/vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
        results['vocab'] = vocab
        results['vocab_size'] = len(vocab)
        loaded_files.append("vocab.json")
        print(f"  ✓ Loaded vocab: {len(vocab)} words")
        sys.stdout.flush()
    except Exception as e:
        errors.append(f"vocab.json: {e}")
        print(f"  ✗ Could not load vocab.json: {e}")
        sys.stdout.flush()
    
    # =========================================================================
    # Load LDA-Gibbs model
    # =========================================================================
    print("\n[VIZ-FORCE] Loading LDA-Gibbs model...")
    sys.stdout.flush()
    
    try:
        with open("data/processed/lda_gibbs_model.json", "r", encoding="utf-8") as f:
            gibbs_data = json.load(f)
        results['theta_gibbs'] = np.array(gibbs_data['theta'])
        results['phi_gibbs'] = np.array(gibbs_data['phi'])
        results['gibbs_history'] = gibbs_data.get('loglik_history', [])
        
        # Use vocab from model if not loaded yet
        if 'vocab' not in results and 'vocab' in gibbs_data:
            results['vocab'] = gibbs_data['vocab']
            results['vocab_size'] = len(gibbs_data['vocab'])
        
        loaded_files.append("lda_gibbs_model.json")
        print(f"  ✓ Loaded LDA-Gibbs: theta={results['theta_gibbs'].shape}, phi={results['phi_gibbs'].shape}")
        print(f"    gibbs_history: {len(results['gibbs_history'])} iterations")
        sys.stdout.flush()
    except Exception as e:
        errors.append(f"lda_gibbs_model.json: {e}")
        print(f"  ✗ Could not load LDA-Gibbs model: {e}")
        traceback.print_exc()
        sys.stdout.flush()
    
    # =========================================================================
    # Load LDA-VI model (optional)
    # =========================================================================
    print("\n[VIZ-FORCE] Loading LDA-VI model...")
    sys.stdout.flush()
    
    if os.path.exists("data/processed/lda_vi_model.json"):
        try:
            with open("data/processed/lda_vi_model.json", "r", encoding="utf-8") as f:
                vi_data = json.load(f)
            results['theta_vi'] = np.array(vi_data['theta'])
            results['phi_vi'] = np.array(vi_data['phi'])
            results['vi_history'] = vi_data.get('elbo_history', [])
            loaded_files.append("lda_vi_model.json")
            print(f"  ✓ Loaded LDA-VI: theta={results['theta_vi'].shape}, phi={results['phi_vi'].shape}")
            print(f"    vi_history: {len(results['vi_history'])} iterations")
            sys.stdout.flush()
        except Exception as e:
            errors.append(f"lda_vi_model.json: {e}")
            print(f"  ✗ Could not load LDA-VI model: {e}")
            sys.stdout.flush()
    else:
        print("  - LDA-VI model not found (optional), skipping")
        sys.stdout.flush()
    
    # =========================================================================
    # Load Gibbs metrics (fallback for convergence history)
    # =========================================================================
    if not results.get('gibbs_history'):
        print("\n[VIZ-FORCE] Loading Gibbs metrics (fallback)...")
        sys.stdout.flush()
        
        if os.path.exists("data/processed/metrics/lda_gibbs_metrics.json"):
            try:
                with open("data/processed/metrics/lda_gibbs_metrics.json", "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                results['gibbs_history'] = metrics.get('log_likelihoods', [])
                loaded_files.append("lda_gibbs_metrics.json")
                print(f"  ✓ Loaded Gibbs history: {len(results['gibbs_history'])} iterations")
                sys.stdout.flush()
            except Exception as e:
                print(f"  ✗ Could not load Gibbs metrics: {e}")
                sys.stdout.flush()
    
    # =========================================================================
    # Load Mixture model - CRITICAL FIX: ensure sys.path is set
    # =========================================================================
    print("\n[VIZ-FORCE] Loading Mixture model...")
    sys.stdout.flush()
    
    if os.path.exists("data/processed/mixture_model.pkl"):
        try:
            # Ensure models module is importable for pickle
            with open("data/processed/mixture_model.pkl", "rb") as f:
                mixture_model = pickle.load(f)
            
            # Extract responsibilities and weights from model object
            if hasattr(mixture_model, 'responsibilities') and mixture_model.responsibilities is not None:
                results['responsibilities'] = mixture_model.responsibilities
                print(f"  ✓ responsibilities: {results['responsibilities'].shape}")
            else:
                print(f"  ✗ responsibilities not found in model")
                
            if hasattr(mixture_model, 'pi') and mixture_model.pi is not None:
                results['mixture_weights'] = mixture_model.pi
                print(f"  ✓ mixture_weights (pi): {results['mixture_weights'].shape}")
            else:
                print(f"  ✗ mixture_weights (pi) not found in model")
                
            if hasattr(mixture_model, 'loglik_history') and mixture_model.loglik_history:
                results['mixture_history'] = mixture_model.loglik_history
                print(f"  ✓ mixture_history: {len(results['mixture_history'])} iterations")
            else:
                print(f"  - mixture_history not found in model")
            
            loaded_files.append("mixture_model.pkl")
            sys.stdout.flush()
        except Exception as e:
            errors.append(f"mixture_model.pkl: {e}")
            print(f"  ✗ Could not load Mixture model: {e}")
            traceback.print_exc()
            sys.stdout.flush()
    else:
        print("  - Mixture model not found, skipping")
        sys.stdout.flush()
    
    # =========================================================================
    # Load mixture metrics (fallback)
    # =========================================================================
    if not results.get('mixture_history') or results.get('mixture_weights') is None:
        if os.path.exists("data/processed/metrics/mixture_metrics.json"):
            try:
                with open("data/processed/metrics/mixture_metrics.json", "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                if not results.get('mixture_history'):
                    results['mixture_history'] = metrics.get('loglik_history', [])
                if results.get('mixture_weights') is None and 'mixture_weights' in metrics:
                    results['mixture_weights'] = np.array(metrics['mixture_weights'])
                loaded_files.append("mixture_metrics.json")
                print(f"  ✓ Loaded Mixture metrics from JSON (fallback)")
                sys.stdout.flush()
            except Exception as e:
                print(f"  ✗ Could not load Mixture metrics: {e}")
                sys.stdout.flush()
    
    # =========================================================================
    # Load Retrieval results
    # =========================================================================
    print("\n[VIZ-FORCE] Loading Retrieval results...")
    sys.stdout.flush()
    
    if os.path.exists("data/processed/retrieval_results.json"):
        try:
            with open("data/processed/retrieval_results.json", "r", encoding="utf-8") as f:
                results['retrieval_results'] = json.load(f)
            loaded_files.append("retrieval_results.json")
            rr = results['retrieval_results']
            print(f"  ✓ Loaded retrieval results:")
            print(f"    queries: {len(rr.get('queries', []))}")
            print(f"    tfidf_results: {len(rr.get('tfidf_results', []))}")
            print(f"    lda_results: {len(rr.get('lda_results', []))}")
            sys.stdout.flush()
        except Exception as e:
            errors.append(f"retrieval_results.json: {e}")
            print(f"  ✗ Could not load retrieval results: {e}")
            sys.stdout.flush()
    else:
        print("  - Retrieval results not found, skipping")
        sys.stdout.flush()
    
    # =========================================================================
    # Summary of loaded files
    # =========================================================================
    print("\n" + "-" * 50)
    print("LOADED FILES:")
    for f in loaded_files:
        print(f"  ✓ {f}")
    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
    print("-" * 50)
    sys.stdout.flush()
    
    # =========================================================================
    # Run Visualizations - UNCONDITIONALLY
    # =========================================================================
    print_step_start("VIZ", "GENERATE ALL VISUALIZATIONS")
    
    # Create output directories
    viz_dirs = [
        "data/processed/visualizations/lda",
        "data/processed/visualizations/lda_vi",
        "data/processed/visualizations/mixture",
        "data/processed/visualizations/retrieval"
    ]
    for d in viz_dirs:
        os.makedirs(d, exist_ok=True)
    
    generated_files = []
    viz_errors = []
    
    try:
        from src.pipeline.visualization_runner import (
            run_lda_visualizations,
            run_mixture_visualizations,
            run_retrieval_visualizations
        )
        
        # LDA Gibbs visualizations
        if results.get('theta_gibbs') is not None and results.get('phi_gibbs') is not None:
            print("\n[VIZ-FORCE] Running LDA Gibbs visualizations...")
            sys.stdout.flush()
            try:
                run_lda_visualizations(
                    results['theta_gibbs'],
                    results['phi_gibbs'],
                    results.get('vocab', {}),
                    results.get('gibbs_history', []),
                    output_dir="data/processed/visualizations/lda",
                    model_type="gibbs"
                )
                generated_files.extend([
                    "lda/topic_words.png",
                    "lda/doc_topic_heatmap.png",
                    "lda/gibbs_convergence.png"
                ])
            except Exception as e:
                viz_errors.append(f"LDA Gibbs: {e}")
                print(f"  ✗ LDA Gibbs visualization error: {e}")
                traceback.print_exc()
                sys.stdout.flush()
        else:
            print("\n[VIZ-FORCE] Skipping LDA Gibbs (theta/phi not available)")
            sys.stdout.flush()
        
        # LDA VI visualizations
        if results.get('theta_vi') is not None and results.get('phi_vi') is not None:
            print("\n[VIZ-FORCE] Running LDA VI visualizations...")
            sys.stdout.flush()
            try:
                run_lda_visualizations(
                    results['theta_vi'],
                    results['phi_vi'],
                    results.get('vocab', {}),
                    results.get('vi_history', []),
                    output_dir="data/processed/visualizations/lda_vi",
                    model_type="vi"
                )
                generated_files.extend([
                    "lda_vi/topic_words.png",
                    "lda_vi/doc_topic_heatmap.png",
                    "lda_vi/vi_convergence.png"
                ])
            except Exception as e:
                viz_errors.append(f"LDA VI: {e}")
                print(f"  ✗ LDA VI visualization error: {e}")
                traceback.print_exc()
                sys.stdout.flush()
        else:
            print("\n[VIZ-FORCE] Skipping LDA VI (theta_vi/phi_vi not available)")
            sys.stdout.flush()
        
        # Mixture visualizations - CRITICAL
        print("\n[VIZ-FORCE] Running Mixture visualizations...")
        sys.stdout.flush()
        
        if results.get('responsibilities') is not None:
            try:
                mixture_weights = results.get('mixture_weights')
                if mixture_weights is None:
                    mixture_weights = np.array([])
                    print("  [WARNING] mixture_weights is None, using empty array")
                
                run_mixture_visualizations(
                    results['responsibilities'],
                    results.get('mixture_history', []),
                    mixture_weights,
                    output_dir="data/processed/visualizations/mixture"
                )
                generated_files.extend([
                    "mixture/responsibilities_heatmap.png",
                    "mixture/mixture_weights.png",
                    "mixture/em_convergence.png",
                    "mixture/expert_assignments.png"
                ])
            except Exception as e:
                viz_errors.append(f"Mixture: {e}")
                print(f"  ✗ Mixture visualization error: {e}")
                traceback.print_exc()
                sys.stdout.flush()
        else:
            print("  [WARNING] responsibilities is None - cannot generate mixture visualizations")
            viz_errors.append("Mixture: responsibilities is None")
            sys.stdout.flush()
        
        # Retrieval visualizations - CRITICAL
        print("\n[VIZ-FORCE] Running Retrieval visualizations...")
        sys.stdout.flush()
        
        if results.get('retrieval_results') is not None:
            try:
                run_retrieval_visualizations(
                    results['retrieval_results'],
                    output_dir="data/processed/visualizations/retrieval"
                )
                generated_files.extend([
                    "retrieval/retrieval_comparison.png",
                    "retrieval/score_distribution.png"
                ])
            except Exception as e:
                viz_errors.append(f"Retrieval: {e}")
                print(f"  ✗ Retrieval visualization error: {e}")
                traceback.print_exc()
                sys.stdout.flush()
        else:
            print("  [WARNING] retrieval_results is None - cannot generate retrieval visualizations")
            viz_errors.append("Retrieval: retrieval_results is None")
            sys.stdout.flush()
        
        print_step_complete("VIZ", "VISUALIZATION GENERATION")
        
    except Exception as e:
        print_error("VISUALIZATION", e)
        viz_errors.append(f"General: {e}")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    elapsed_time = time.time() - start_time
    
    print("\n")
    print("=" * 60)
    print("  VISUALIZATION-FORCE MODE COMPLETE!")
    print("=" * 60)
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    
    print(f"\n--- FILES LOADED ({len(loaded_files)}) ---")
    for f in loaded_files:
        print(f"  ✓ {f}")
    
    print(f"\n--- VISUALIZATIONS GENERATED ---")
    for f in generated_files:
        full_path = f"data/processed/visualizations/{f}"
        exists = "✓" if os.path.exists(full_path) else "✗"
        print(f"  {exists} {f}")
    
    if viz_errors:
        print(f"\n--- ERRORS ({len(viz_errors)}) ---")
        for e in viz_errors:
            print(f"  ✗ {e}")
    
    print("=" * 60 + "\n")
    sys.stdout.flush()
    
    return results


def run_viz_only_mode():
    """
    Visualization-only mode: Alias for run_viz_force_mode for backward compatibility.
    """
    return run_viz_force_mode()


def run_pipeline(
    data_path: str = "data/processed/processed_data.json",
    num_topics: int = 10,
    num_gibbs_iterations: int = 50,
    num_vi_iterations: int = 30,
    num_experts: int = 5,
    num_docs_limit: int = None,
    skip_vi: bool = False,
    skip_mixture: bool = False,
    skip_recommendation: bool = False,
    skip_visualization: bool = False,
    reuse_lda: bool = False,
    reuse_preprocessed: bool = False,
    viz_only: bool = False,
    viz_force: bool = False
):
    """
    Run the complete pipeline.
    
    Args:
        data_path: Path to processed data
        num_topics: Number of LDA topics
        num_gibbs_iterations: Number of Gibbs sampling iterations
        num_vi_iterations: Number of VI iterations
        num_experts: Number of mixture experts
        num_docs_limit: Limit number of documents (for faster testing)
        skip_vi: Skip LDA-VI step
        skip_mixture: Skip mixture model step
        skip_recommendation: Skip recommendation step
        skip_visualization: Skip visualization step
        reuse_lda: Reuse existing LDA models instead of retraining
        reuse_preprocessed: Reuse saved preprocessing artifacts
        viz_only: Run visualization only (no preprocessing, no training)
        viz_force: Force visualization mode (load all artifacts, generate all visualizations)
    """
    
    # =========================================================================
    # VISUALIZATION-FORCE MODE (highest priority)
    # =========================================================================
    if viz_force:
        return run_viz_force_mode()
    
    # =========================================================================
    # VISUALIZATION-ONLY MODE
    # =========================================================================
    if viz_only:
        return run_viz_only_mode()
    
    start_time = time.time()
    
    print("\n")
    print("=" * 60)
    print("  PROBABILISTIC TOPIC MODELING PIPELINE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Data path: {data_path}")
    print(f"  - Num topics: {num_topics}")
    print(f"  - Gibbs iterations: {num_gibbs_iterations}")
    print(f"  - VI iterations: {num_vi_iterations}")
    print(f"  - Num experts: {num_experts}")
    print(f"  - Doc limit: {num_docs_limit or 'None (use all)'}")
    print(f"  - Reuse LDA: {reuse_lda}")
    print(f"  - Reuse Preprocessed: {reuse_preprocessed}")
    print("=" * 60 + "\n")
    sys.stdout.flush()
    
    # Ensure base output directories exist
    os.makedirs("data/processed/visualizations", exist_ok=True)
    os.makedirs("data/processed/metrics", exist_ok=True)
    
    # Track results
    results = {}
    skip_lda_training = False
    skip_preprocess = False
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print_step_start(1, "LOAD DATA")
    
    try:
        from src.pipeline.data_loader import load_data, extract_texts
        
        data = load_data(data_path)
        texts = extract_texts(data)
        
        # Limit documents if specified
        if num_docs_limit:
            texts = texts[:num_docs_limit]
            print(f"[INFO] Limited to {num_docs_limit} documents")
            sys.stdout.flush()
        
        results['texts'] = texts
        results['num_docs'] = len(texts)
        
        print_step_complete(1, "LOAD DATA")
        
    except Exception as e:
        print_error("LOAD DATA", e)
        return None
    
    # =========================================================================
    # REUSE PREPROCESSED ARTIFACTS IF REQUESTED
    # =========================================================================
    if reuse_preprocessed:
        print_step_start("2A", "LOAD PREPROCESSED ARTIFACTS")
        
        vocab_path = "data/processed/vocab.json"
        word_id_docs_path = "data/processed/word_id_docs.npy"
        
        if os.path.exists(vocab_path) and os.path.exists(word_id_docs_path):
            try:
                word_id_docs = np.load(word_id_docs_path, allow_pickle=True).tolist()
                with open(vocab_path, "r", encoding="utf-8") as f:
                    vocab = json.load(f)
                id_to_word = {int(v): k for k, v in vocab.items()}
                
                num_docs_cached = len(word_id_docs)
                num_docs_texts = len(texts)
                
                print(f"[INFO] Cached preprocessed: vocab_size={len(vocab)}, docs={num_docs_cached}")
                print(f"[INFO] Current text count: {num_docs_texts}")
                sys.stdout.flush()
                
                # SANITY CHECK: Don't reuse preprocessing if doc counts differ significantly
                if num_docs_cached < num_docs_texts * 0.9 or num_docs_cached < 100:
                    print(f"[WARNING] Cached preprocessing has {num_docs_cached} docs, but current run has {num_docs_texts} docs.")
                    print(f"[WARNING] Ignoring --reuse-preprocessed to avoid stale artifacts.")
                    sys.stdout.flush()
                    skip_preprocess = False
                else:
                    results['word_id_docs'] = word_id_docs
                    results['vocab'] = vocab
                    results['id_to_word'] = id_to_word
                    results['vocab_size'] = len(vocab)
                    
                    print_step_complete("2A", "LOAD PREPROCESSED")
                    skip_preprocess = True
                
            except Exception as e:
                print_error("LOAD PREPROCESSED", e)
                print("[WARNING] Falling back to full preprocessing.")
                sys.stdout.flush()
                skip_preprocess = False
        else:
            print(f"[WARNING] Preprocessed files not found:")
            if not os.path.exists(vocab_path):
                print(f"  Missing: {vocab_path}")
            if not os.path.exists(word_id_docs_path):
                print(f"  Missing: {word_id_docs_path}")
            print("[WARNING] Falling back to full preprocessing.")
            sys.stdout.flush()
            skip_preprocess = False
    
    # =========================================================================
    # STEP 2: Preprocess
    # =========================================================================
    if skip_preprocess:
        print("\n[INFO] Skipping STEP 2: PREPROCESS (reuse mode)")
        sys.stdout.flush()
    else:
        print_step_start(2, "PREPROCESS")
        
        try:
            from src.pipeline.preprocessing import run_preprocessing
            
            word_id_docs, vocab, id_to_word = run_preprocessing(texts)
            
            results['word_id_docs'] = word_id_docs
            results['vocab'] = vocab
            results['id_to_word'] = id_to_word
            results['vocab_size'] = len(vocab)
            
            # Save preprocessed artifacts for future reuse
            print("[INFO] Saving preprocessed artifacts for future reuse...")
            sys.stdout.flush()
            np.save("data/processed/word_id_docs.npy", np.array(word_id_docs, dtype=object))
            with open("data/processed/vocab.json", "w", encoding="utf-8") as f:
                json.dump(vocab, f)
            print("[INFO] Saved: word_id_docs.npy, vocab.json")
            sys.stdout.flush()
            
            print_step_complete(2, "PREPROCESS")
            
        except Exception as e:
            print_error("PREPROCESS", e)
            return None
    
    # =========================================================================
    # REUSE LDA MODELS IF REQUESTED
    # =========================================================================
    if reuse_lda:
        print_step_start("3A", "LOAD EXISTING LDA MODELS")
        
        try:
            # Load Gibbs
            with open("data/processed/lda_gibbs_model.json", "r", encoding="utf-8") as f:
                gibbs_data = json.load(f)
            theta_gibbs_loaded = np.array(gibbs_data["theta"])
            phi_gibbs_loaded = np.array(gibbs_data["phi"])
            gibbs_history_loaded = gibbs_data.get("loglik_history", [])
            
            print(f"[INFO] Loaded LDA-Gibbs: theta={theta_gibbs_loaded.shape}, phi={phi_gibbs_loaded.shape}")
            sys.stdout.flush()
            
            # SANITY CHECK: theta should have one row per document
            num_docs_lda = theta_gibbs_loaded.shape[0]
            num_docs_current = len(results.get('word_id_docs', []))
            
            if num_docs_current > 0 and num_docs_lda < num_docs_current * 0.9:
                print(f"[WARNING] Cached LDA has {num_docs_lda} docs, but current preprocessing has {num_docs_current} docs.")
                print(f"[WARNING] Ignoring --reuse-lda to avoid stale artifacts.")
                sys.stdout.flush()
                skip_lda_training = False
            else:
                results["theta_gibbs"] = theta_gibbs_loaded
                results["phi_gibbs"] = phi_gibbs_loaded
                results["gibbs_history"] = gibbs_history_loaded
                
                # Also load vocab from the model if available
                if "vocab" in gibbs_data:
                    results["vocab"] = gibbs_data["vocab"]
                
                # Load VI if available and not skipped
                if not skip_vi and os.path.exists("data/processed/lda_vi_model.json"):
                    with open("data/processed/lda_vi_model.json", "r", encoding="utf-8") as f:
                        vi_data = json.load(f)
                    results["theta_vi"] = np.array(vi_data["theta"])
                    results["phi_vi"] = np.array(vi_data["phi"])
                    results["vi_history"] = vi_data.get("elbo_history", [])
                    print(f"[INFO] Loaded LDA-VI: theta={results['theta_vi'].shape}, phi={results['phi_vi'].shape}")
                    sys.stdout.flush()
                
                print_step_complete("3A", "LOAD EXISTING LDA MODELS")
                skip_lda_training = True
            
        except Exception as e:
            print_error("LOAD EXISTING LDA MODELS", e)
            print("[WARNING] Falling back to full LDA training.")
            sys.stdout.flush()
            skip_lda_training = False
    
    # =========================================================================
    # STEP 3: LDA-Gibbs
    # =========================================================================
    if skip_lda_training:
        print("\n[INFO] Skipping STEP 3: LDA-Gibbs (reuse-lda enabled)")
        sys.stdout.flush()
    else:
        print_step_start(3, "LDA-GIBBS")
        
        try:
            from src.pipeline.lda_gibbs_runner import run_lda_gibbs
            
            word_id_docs = results.get('word_id_docs')
            vocab = results.get('vocab')
            id_to_word = results.get('id_to_word')
            
            theta_gibbs, phi_gibbs, gibbs_history = run_lda_gibbs(
                word_id_docs=word_id_docs,
                vocab=vocab,
                id_to_word=id_to_word,
                num_topics=num_topics,
                num_iterations=num_gibbs_iterations,
                output_path="data/processed/lda_gibbs_model.json",
                metrics_path="data/processed/metrics/lda_gibbs_metrics.json"
            )
            
            results['theta_gibbs'] = theta_gibbs
            results['phi_gibbs'] = phi_gibbs
            results['gibbs_history'] = gibbs_history
            
            print_step_complete(3, "LDA-GIBBS")
            
        except Exception as e:
            print_error("LDA-GIBBS", e)
            return None
    
    # =========================================================================
    # STEP 4: LDA-VI
    # =========================================================================
    if skip_lda_training or skip_vi:
        print("\n[INFO] Skipping STEP 4: LDA-VI (reuse-lda or --skip-vi enabled)")
        sys.stdout.flush()
    else:
        print_step_start(4, "LDA-VI")
        
        try:
            from src.pipeline.lda_vi_runner import run_lda_vi
            
            word_id_docs = results.get('word_id_docs')
            vocab = results.get('vocab')
            id_to_word = results.get('id_to_word')
            
            theta_vi, phi_vi, vi_history = run_lda_vi(
                word_id_docs=word_id_docs,
                vocab=vocab,
                id_to_word=id_to_word,
                num_topics=num_topics,
                num_iterations=num_vi_iterations,
                output_path="data/processed/lda_vi_model.json",
                metrics_path="data/processed/metrics/lda_vi_metrics.json"
            )
            
            results['theta_vi'] = theta_vi
            results['phi_vi'] = phi_vi
            results['vi_history'] = vi_history
            
            print_step_complete(4, "LDA-VI")
            
        except Exception as e:
            print_error("LDA-VI", e)
            print("[WARNING] Continuing without LDA-VI...")
            sys.stdout.flush()
    
    # =========================================================================
    # STEP 5: Mixture Model
    # =========================================================================
    if not skip_mixture:
        print_step_start(5, "MIXTURE MODEL")
        
        try:
            from src.pipeline.mixture_runner import run_mixture_model
            
            # Use theta_gibbs from results (either loaded or trained)
            theta_gibbs = results.get('theta_gibbs')
            
            if theta_gibbs is None:
                raise ValueError("theta_gibbs not available - cannot train mixture model")
            
            responsibilities, mixture_history, mixture_weights = run_mixture_model(
                theta=theta_gibbs,
                num_experts=num_experts,
                num_iterations=50,
                output_path="data/processed/mixture_model.pkl",
                metrics_path="data/processed/metrics/mixture_metrics.json"
            )
            
            results['responsibilities'] = responsibilities
            results['mixture_history'] = mixture_history
            results['mixture_weights'] = mixture_weights
            
            print_step_complete(5, "MIXTURE MODEL")
            
        except Exception as e:
            print_error("MIXTURE MODEL", e)
            print("[WARNING] Continuing without mixture model...")
            sys.stdout.flush()
    else:
        print("\n[INFO] Skipping STEP 5: MIXTURE MODEL")
        sys.stdout.flush()
    
    # =========================================================================
    # STEP 6: Recommendation
    # =========================================================================
    if not skip_recommendation:
        print_step_start(6, "RECOMMENDATION")
        
        try:
            from src.pipeline.recommendation_runner import run_recommendation
            
            # Use theta_gibbs from results (either loaded or trained)
            theta_gibbs = results.get('theta_gibbs')
            texts = results.get('texts')
            
            if theta_gibbs is None:
                raise ValueError("theta_gibbs not available - cannot run recommendation")
            
            retrieval_results = run_recommendation(
                texts=texts,
                theta=theta_gibbs,
                output_path="data/processed/retrieval_results.json"
            )
            
            results['retrieval_results'] = retrieval_results
            
            print_step_complete(6, "RECOMMENDATION")
            
        except Exception as e:
            print_error("RECOMMENDATION", e)
            print("[WARNING] Continuing without recommendation...")
            sys.stdout.flush()
    else:
        print("\n[INFO] Skipping STEP 6: RECOMMENDATION")
        sys.stdout.flush()
    
    # =========================================================================
    # STEP 7: Visualization
    # =========================================================================
    if not skip_visualization:
        print_step_start(7, "VISUALIZATION")
        
        try:
            from src.pipeline.visualization_runner import run_all_visualizations
            
            run_all_visualizations(
                theta_gibbs=results.get('theta_gibbs'),
                phi_gibbs=results.get('phi_gibbs'),
                vocab=results.get('vocab'),
                gibbs_history=results.get('gibbs_history'),
                theta_vi=results.get('theta_vi'),
                phi_vi=results.get('phi_vi'),
                vi_history=results.get('vi_history'),
                responsibilities=results.get('responsibilities'),
                mixture_history=results.get('mixture_history'),
                mixture_weights=results.get('mixture_weights'),
                retrieval_results=results.get('retrieval_results')
            )
            
            print_step_complete(7, "VISUALIZATION")
            
        except Exception as e:
            print_error("VISUALIZATION", e)
            print("[WARNING] Visualization failed, but pipeline complete.")
            sys.stdout.flush()
    else:
        print("\n[INFO] Skipping STEP 7: VISUALIZATION")
        sys.stdout.flush()
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    elapsed_time = time.time() - start_time
    
    print("\n")
    print("=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    print(f"\nDocuments processed: {results.get('num_docs', 0)}")
    print(f"Vocabulary size: {results.get('vocab_size', 0)}")
    print(f"Topics: {num_topics}")
    print(f"\nSaved artifacts:")
    if reuse_preprocessed:
        print(f"  - (reused preprocessed artifacts)")
    else:
        print(f"  - data/processed/word_id_docs.npy")
        print(f"  - data/processed/vocab.json")
    if not reuse_lda:
        print(f"  - data/processed/lda_gibbs_model.json")
        if not skip_vi:
            print(f"  - data/processed/lda_vi_model.json")
    else:
        print(f"  - (reused existing LDA models)")
    if not skip_mixture:
        print(f"  - data/processed/mixture_model.pkl")
    if not skip_recommendation:
        print(f"  - data/processed/retrieval_results.json")
    if not skip_visualization:
        print(f"  - data/processed/visualizations/lda/*.png")
        print(f"  - data/processed/visualizations/mixture/*.png")
        print(f"  - data/processed/visualizations/retrieval/*.png")
    print("=" * 60 + "\n")
    sys.stdout.flush()
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run probabilistic topic modeling pipeline')
    
    parser.add_argument('--data', type=str, default='data/processed/processed_data.json',
                       help='Path to processed data JSON')
    parser.add_argument('--topics', type=int, default=10,
                       help='Number of topics')
    parser.add_argument('--gibbs-iter', type=int, default=50,
                       help='Number of Gibbs sampling iterations')
    parser.add_argument('--vi-iter', type=int, default=30,
                       help='Number of VI iterations')
    parser.add_argument('--experts', type=int, default=5,
                       help='Number of mixture experts')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of documents (for testing)')
    parser.add_argument('--skip-vi', action='store_true',
                       help='Skip LDA-VI step')
    parser.add_argument('--skip-mixture', action='store_true',
                       help='Skip mixture model step')
    parser.add_argument('--skip-recommendation', action='store_true',
                       help='Skip recommendation step')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='Skip visualization step')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: limit to 500 docs, 20 Gibbs iter, skip VI')
    parser.add_argument('--reuse-lda', action='store_true',
                       help='Reuse existing LDA models instead of retraining')
    parser.add_argument('--reuse-preprocessed', action='store_true',
                       help='Reuse saved preprocessing artifacts (vocab + word_id_docs)')
    parser.add_argument('--viz-only', action='store_true',
                       help='Run visualization only (no preprocessing, no training)')
    parser.add_argument('--viz-force', action='store_true',
                       help='Force visualization mode (load all artifacts, generate all visualizations)')
    
    args = parser.parse_args()
    
    # Fast mode overrides
    if args.fast:
        args.limit = 500
        args.gibbs_iter = 20
        args.vi_iter = 15
        args.skip_vi = True
    
    run_pipeline(
        data_path=args.data,
        num_topics=args.topics,
        num_gibbs_iterations=args.gibbs_iter,
        num_vi_iterations=args.vi_iter,
        num_experts=args.experts,
        num_docs_limit=args.limit,
        skip_vi=args.skip_vi,
        skip_mixture=args.skip_mixture,
        skip_recommendation=args.skip_recommendation,
        skip_visualization=args.skip_visualization,
        reuse_lda=args.reuse_lda,
        reuse_preprocessed=args.reuse_preprocessed,
        viz_only=args.viz_only,
        viz_force=args.viz_force
    )


if __name__ == "__main__":
    main()
