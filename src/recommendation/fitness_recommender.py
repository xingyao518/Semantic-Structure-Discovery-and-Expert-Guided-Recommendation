"""
Fitness Recommendation System - Lightweight Layer

This module builds on top of existing LDA, Mixture, and Retrieval outputs
to provide intent-based fitness recommendations WITHOUT modifying existing code.

Usage:
    python -m src.recommendation.fitness_recommender --intent injury_recovery
    python -m src.recommendation.fitness_recommender --intent endurance_training
    python -m src.recommendation.fitness_recommender --list-intents
"""

import sys
import os
import json
import pickle
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


class FitnessRecommender:
    """
    Lightweight fitness recommender built on LDA topics and Mixture-of-Experts.
    
    Maps user intents to relevant topics/experts and provides ranked recommendations.
    """
    
    # Define user intents and their characteristics
    USER_INTENTS = {
        "injury_recovery": {
            "keywords": ["injury", "pain", "recovery", "rehab", "rest", "physical", "therapy"],
            "description": "Recommendations for injury prevention and recovery"
        },
        "endurance_training": {
            "keywords": ["marathon", "distance", "endurance", "long", "pace", "training", "run"],
            "description": "Long-distance running and endurance building"
        },
        "nutrition": {
            "keywords": ["nutrition", "diet", "food", "carb", "protein", "hydration", "supplement"],
            "description": "Nutrition and dietary advice for runners"
        },
        "running_form": {
            "keywords": ["form", "technique", "stride", "cadence", "posture", "biomechanics"],
            "description": "Running technique and form improvement"
        },
        "beginner_guidance": {
            "keywords": ["beginner", "start", "couch", "first", "new", "runner", "advice"],
            "description": "Getting started with running"
        },
        "race_preparation": {
            "keywords": ["race", "competition", "5k", "10k", "half", "taper", "strategy"],
            "description": "Race preparation and competition strategies"
        }
    }
    
    def __init__(self):
        """Initialize the fitness recommender."""
        self.lda_model = None
        self.mixture_model = None
        self.retrieval_results = None
        self.processed_data = None
        
        self.theta = None  # Document-topic distributions
        self.phi = None    # Topic-word distributions
        self.vocab = None
        self.id_to_word = None
        self.responsibilities = None  # Document-expert responsibilities
        self.mixture_weights = None   # Expert mixture weights
        
        self.intent_to_topic = {}  # Mapping: intent -> list of topic IDs
        self.intent_to_expert = {}  # Mapping: intent -> list of expert IDs
        
    def load_models(self):
        """Load all necessary models and data."""
        print("\n[FitnessRecommender] Loading models and data...")
        sys.stdout.flush()
        
        # Load LDA VI model
        lda_path = "data/processed/lda_vi_model.json"
        if os.path.exists(lda_path):
            print(f"  Loading LDA VI model from {lda_path}...")
            with open(lda_path, 'r', encoding='utf-8') as f:
                self.lda_model = json.load(f)
            
            self.theta = np.array(self.lda_model['theta'])
            self.phi = np.array(self.lda_model['phi'])
            self.vocab = self.lda_model.get('vocab', {})
            
            # Create id_to_word mapping
            if self.vocab:
                first_key = next(iter(self.vocab.keys()))
                if isinstance(first_key, str):
                    # vocab is word -> id, convert to id -> word
                    self.id_to_word = {v: k for k, v in self.vocab.items()}
                else:
                    # vocab is already id -> word
                    self.id_to_word = self.vocab
            
            print(f"    [OK] theta: {self.theta.shape}, phi: {self.phi.shape}")
            print(f"    [OK] vocab size: {len(self.vocab)}")
        else:
            print(f"  [X] LDA VI model not found at {lda_path}")
            print("    Using LDA Gibbs model as fallback...")
            
            lda_gibbs_path = "data/processed/lda_gibbs_model.json"
            if os.path.exists(lda_gibbs_path):
                with open(lda_gibbs_path, 'r', encoding='utf-8') as f:
                    self.lda_model = json.load(f)
                
                self.theta = np.array(self.lda_model['theta'])
                self.phi = np.array(self.lda_model['phi'])
                self.vocab = self.lda_model.get('vocab', {})
                
                if self.vocab:
                    first_key = next(iter(self.vocab.keys()))
                    if isinstance(first_key, str):
                        self.id_to_word = {v: k for k, v in self.vocab.items()}
                    else:
                        self.id_to_word = self.vocab
                
                print(f"    [OK] theta: {self.theta.shape}, phi: {self.phi.shape}")
            else:
                raise FileNotFoundError("No LDA model found")
        
        # Load Mixture model
        mixture_path = "data/processed/mixture_model.pkl"
        if os.path.exists(mixture_path):
            print(f"  Loading Mixture model from {mixture_path}...")
            with open(mixture_path, 'rb') as f:
                self.mixture_model = pickle.load(f)
            
            self.responsibilities = self.mixture_model.responsibilities
            self.mixture_weights = self.mixture_model.pi
            
            print(f"    [OK] responsibilities: {self.responsibilities.shape}")
            print(f"    [OK] mixture weights: {self.mixture_weights}")
        else:
            print(f"  [X] Mixture model not found at {mixture_path}")
        
        # Load retrieval results (optional)
        retrieval_path = "data/processed/retrieval_results.json"
        if os.path.exists(retrieval_path):
            print(f"  Loading retrieval results from {retrieval_path}...")
            with open(retrieval_path, 'r', encoding='utf-8') as f:
                self.retrieval_results = json.load(f)
            print(f"    [OK] retrieval queries: {len(self.retrieval_results.get('queries', []))}")
        else:
            print(f"  [X] Retrieval results not found at {retrieval_path}")
        
        # Load processed data for document texts
        data_path = "data/processed/processed_data.json"
        if os.path.exists(data_path):
            print(f"  Loading processed data from {data_path}...")
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Handle both list and dict formats
            if isinstance(raw_data, list):
                self.processed_data = raw_data
            elif isinstance(raw_data, dict):
                self.processed_data = raw_data.get('documents', [])
            else:
                self.processed_data = []
            
            num_docs_available = len(self.processed_data)
            print(f"    [OK] documents available: {num_docs_available}")
            
            # Trim to match theta size if necessary
            if num_docs_available > self.theta.shape[0]:
                print(f"    [INFO] Trimming documents to match theta size ({self.theta.shape[0]})")
                self.processed_data = self.processed_data[:self.theta.shape[0]]
        else:
            print(f"  [X] Processed data not found at {data_path}")
        
        print("\n[FitnessRecommender] All models loaded successfully!")
        sys.stdout.flush()
    
    def analyze_topics(self):
        """Analyze topics and print top words for each."""
        print("\n[FitnessRecommender] Analyzing LDA topics...")
        sys.stdout.flush()
        
        K = self.phi.shape[0]
        top_n = 10
        
        print(f"\nTop {top_n} words per topic:")
        print("=" * 60)
        
        for k in range(K):
            top_indices = np.argsort(self.phi[k, :])[::-1][:top_n]
            words = [self.id_to_word.get(idx, f"w{idx}") for idx in top_indices]
            probs = self.phi[k, top_indices]
            
            print(f"\nTopic {k}:")
            for word, prob in zip(words, probs):
                print(f"  {word:20s} {prob:.4f}")
        
        print("\n" + "=" * 60)
        sys.stdout.flush()
    
    def map_intents_to_topics(self):
        """
        Map user intents to LDA topics based on keyword overlap.
        
        Returns:
            Dict mapping intent -> list of (topic_id, score) tuples
        """
        print("\n[FitnessRecommender] Mapping intents to topics...")
        sys.stdout.flush()
        
        K = self.phi.shape[0]
        top_n = 20  # Consider top 20 words per topic
        
        for intent, intent_info in self.USER_INTENTS.items():
            intent_keywords = set(kw.lower() for kw in intent_info['keywords'])
            topic_scores = []
            
            for k in range(K):
                # Get top words for this topic
                top_indices = np.argsort(self.phi[k, :])[::-1][:top_n]
                topic_words = [self.id_to_word.get(idx, "").lower() for idx in top_indices]
                
                # Count keyword matches
                matches = sum(1 for word in topic_words if any(kw in word for kw in intent_keywords))
                
                if matches > 0:
                    # Weighted score: matches * average probability of top words
                    avg_prob = np.mean(self.phi[k, top_indices[:matches]])
                    score = matches * avg_prob
                    topic_scores.append((k, score))
            
            # Sort by score and keep top 2 topics
            topic_scores.sort(key=lambda x: x[1], reverse=True)
            self.intent_to_topic[intent] = [t[0] for t in topic_scores[:2]]
            
            print(f"  {intent:25s} -> Topics {self.intent_to_topic[intent]}")
        
        sys.stdout.flush()
    
    def map_intents_to_experts(self):
        """
        Map user intents to Mixture experts based on topic-expert alignment.
        
        Uses the dominant topics for each expert (based on responsibilities).
        """
        print("\n[FitnessRecommender] Mapping intents to experts...")
        sys.stdout.flush()
        
        if self.responsibilities is None:
            print("  [WARNING] No mixture responsibilities available, skipping expert mapping")
            return
        
        E = self.responsibilities.shape[1]
        
        # For each expert, find dominant topics
        expert_topics = {}
        for e in range(E):
            # Find documents with high responsibility for this expert
            high_resp_docs = np.where(self.responsibilities[:, e] > 0.5)[0]
            
            if len(high_resp_docs) == 0:
                high_resp_docs = np.argsort(self.responsibilities[:, e])[::-1][:100]
            
            # Average topic distribution for these documents
            avg_topic_dist = self.theta[high_resp_docs].mean(axis=0)
            top_topics = np.argsort(avg_topic_dist)[::-1][:3]
            expert_topics[e] = top_topics.tolist()
        
        print("\nExpert -> Dominant Topics:")
        for e, topics in expert_topics.items():
            print(f"  Expert {e}: Topics {topics}")
        
        # Map intents to experts based on topic overlap
        for intent in self.USER_INTENTS.keys():
            intent_topics = set(self.intent_to_topic.get(intent, []))
            expert_scores = []
            
            for e, e_topics in expert_topics.items():
                overlap = len(intent_topics & set(e_topics))
                if overlap > 0:
                    # Score by overlap and expert weight
                    score = overlap * self.mixture_weights[e]
                    expert_scores.append((e, score))
            
            # Sort by score and keep top 2 experts
            expert_scores.sort(key=lambda x: x[1], reverse=True)
            self.intent_to_expert[intent] = [e[0] for e in expert_scores[:2]]
            
            if expert_scores:
                print(f"  {intent:25s} -> Experts {self.intent_to_expert[intent]}")
            else:
                # Fallback: use most common expert
                self.intent_to_expert[intent] = [np.argmax(self.mixture_weights)]
                print(f"  {intent:25s} -> Experts {self.intent_to_expert[intent]} (fallback)")
        
        sys.stdout.flush()
    
    def recommend(self, intent: str, top_k: int = 10) -> List[Dict]:
        """
        Generate recommendations for a given user intent.
        
        Args:
            intent: User intent (e.g., "injury_recovery")
            top_k: Number of recommendations to return
            
        Returns:
            List of recommendation dicts with document info
        """
        if intent not in self.USER_INTENTS:
            raise ValueError(f"Unknown intent: {intent}. Use --list-intents to see available intents.")
        
        print(f"\n[FitnessRecommender] Generating recommendations for intent: {intent}")
        sys.stdout.flush()
        
        # Get mapped topics and experts
        intent_topics = self.intent_to_topic.get(intent, [])
        intent_experts = self.intent_to_expert.get(intent, [])
        
        print(f"  Mapped topics: {intent_topics}")
        print(f"  Mapped experts: {intent_experts}")
        sys.stdout.flush()
        
        # Filter candidate documents by expert (if available)
        if self.responsibilities is not None and intent_experts:
            # Select documents with high responsibility for intent experts
            candidate_mask = np.zeros(self.responsibilities.shape[0], dtype=bool)
            for expert_id in intent_experts:
                # Documents where this expert has > 15% responsibility (or top 30%)
                threshold = max(0.15, np.percentile(self.responsibilities[:, expert_id], 70))
                candidate_mask |= (self.responsibilities[:, expert_id] > threshold)
            
            candidate_docs = np.where(candidate_mask)[0]
            print(f"  Candidate documents (filtered by experts): {len(candidate_docs)}")
            
            # Fallback: if too few candidates, use top documents by expert responsibility
            if len(candidate_docs) < top_k * 3:
                print(f"  [INFO] Too few candidates, expanding to top documents by expert")
                for expert_id in intent_experts:
                    top_docs = np.argsort(self.responsibilities[:, expert_id])[::-1][:top_k * 5]
                    candidate_mask[top_docs] = True
                candidate_docs = np.where(candidate_mask)[0]
                print(f"  Expanded candidate documents: {len(candidate_docs)}")
        else:
            # Use all documents
            candidate_docs = np.arange(self.theta.shape[0])
            print(f"  Candidate documents (all): {len(candidate_docs)}")
        
        # Rank candidates by topic relevance
        if intent_topics:
            # Score = sum of probabilities for intent topics
            topic_scores = self.theta[candidate_docs][:, intent_topics].sum(axis=1)
        else:
            # Fallback: use uniform scores
            topic_scores = np.ones(len(candidate_docs))
        
        # Get top-k
        top_indices = np.argsort(topic_scores)[::-1][:top_k]
        top_doc_ids = candidate_docs[top_indices]
        top_scores = topic_scores[top_indices]
        
        # Build recommendations
        recommendations = []
        for i, (doc_id, score) in enumerate(zip(top_doc_ids, top_scores)):
            # Get dominant topic
            dominant_topic = int(np.argmax(self.theta[doc_id]))
            
            # Get dominant expert (if available)
            if self.responsibilities is not None:
                dominant_expert = int(np.argmax(self.responsibilities[doc_id]))
            else:
                dominant_expert = None
            
            # Get topic keywords for explanation
            if intent_topics:
                top_topic_words = []
                for t in intent_topics:
                    top_indices = np.argsort(self.phi[t, :])[::-1][:3]
                    words = [self.id_to_word.get(idx, f"w{idx}") for idx in top_indices]
                    top_topic_words.extend(words)
                explanation = f"Relevant topics: {', '.join(set(top_topic_words[:5]))}"
            else:
                explanation = "General fitness advice"
            
            # Get document text snippet (if available)
            doc_text = ""
            if self.processed_data and doc_id < len(self.processed_data):
                doc_obj = self.processed_data[doc_id]
                doc_text = doc_obj.get('text', '')[:200] + "..."
            
            rec = {
                "rank": i + 1,
                "document_id": int(doc_id),
                "score": float(score),
                "dominant_topic": dominant_topic,
                "dominant_expert": dominant_expert,
                "explanation": explanation,
                "text_snippet": doc_text
            }
            
            recommendations.append(rec)
        
        print(f"  Generated {len(recommendations)} recommendations")
        sys.stdout.flush()
        
        return recommendations
    
    def print_recommendations(self, intent: str, recommendations: List[Dict]):
        """Pretty print recommendations."""
        print("\n" + "=" * 80)
        print(f"FITNESS RECOMMENDATIONS FOR: {intent.upper()}")
        print("=" * 80)
        print(f"\nIntent description: {self.USER_INTENTS[intent]['description']}")
        print(f"Number of recommendations: {len(recommendations)}\n")
        
        for rec in recommendations:
            print(f"Rank {rec['rank']}:")
            print(f"  Document ID: {rec['document_id']}")
            print(f"  Relevance Score: {rec['score']:.4f}")
            print(f"  Dominant Topic: {rec['dominant_topic']}")
            if rec['dominant_expert'] is not None:
                print(f"  Dominant Expert: {rec['dominant_expert']}")
            print(f"  Explanation: {rec['explanation']}")
            if rec['text_snippet']:
                print(f"  Snippet: {rec['text_snippet'][:150]}...")
            print()
        
        print("=" * 80)
        sys.stdout.flush()
    
    def generate_all_examples(self):
        """Generate recommendation examples for all intents."""
        print("\n[FitnessRecommender] Generating examples for all intents...")
        sys.stdout.flush()
        
        all_examples = {}
        
        for intent in self.USER_INTENTS.keys():
            print(f"\n  Generating for: {intent}...")
            recommendations = self.recommend(intent, top_k=5)
            all_examples[intent] = {
                "description": self.USER_INTENTS[intent]['description'],
                "mapped_topics": [int(t) for t in self.intent_to_topic.get(intent, [])],
                "mapped_experts": [int(e) for e in self.intent_to_expert.get(intent, [])],
                "recommendations": recommendations
            }
        
        # Save to file
        output_path = "data/processed/recommendation_examples.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_examples, f, indent=2, ensure_ascii=False)
        
        print(f"\n[INFO] Saved examples to: {output_path}")
        sys.stdout.flush()
        
        # Print summary of recommendations for each intent
        print("\n" + "=" * 80)
        print("RECOMMENDATION SUMMARY - TOP 3 PER INTENT")
        print("=" * 80)
        
        for intent, data in all_examples.items():
            print(f"\n{'=' * 80}")
            print(f"Intent: {intent.upper()}")
            print(f"{'=' * 80}")
            print(f"Description: {data['description']}")
            print(f"Mapped Topics: {data['mapped_topics']}")
            print(f"Mapped Experts: {data['mapped_experts']}")
            print(f"\nTop 3 Recommendations:")
            print("-" * 80)
            
            for rec in data['recommendations'][:3]:
                print(f"\n  Rank {rec['rank']}:")
                print(f"    Document ID: {rec['document_id']}")
                print(f"    Relevance Score: {rec['score']:.4f}")
                print(f"    Dominant Topic: {rec['dominant_topic']}")
                if rec['dominant_expert'] is not None:
                    print(f"    Dominant Expert: {rec['dominant_expert']}")
                print(f"    Explanation: {rec['explanation']}")
                if rec.get('text_snippet'):
                    snippet = rec['text_snippet'][:120].replace('\n', ' ')
                    print(f"    Snippet: {snippet}...")
        
        print("\n" + "=" * 80)
        print(f"Full results saved to: {output_path}")
        print("=" * 80 + "\n")
        sys.stdout.flush()
        
        return all_examples


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fitness Recommendation System - Intent-based recommendations"
    )
    parser.add_argument(
        '--intent',
        type=str,
        help='User intent (e.g., injury_recovery, endurance_training)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of recommendations to generate (default: 10)'
    )
    parser.add_argument(
        '--list-intents',
        action='store_true',
        help='List all available user intents'
    )
    parser.add_argument(
        '--analyze-topics',
        action='store_true',
        help='Analyze and print LDA topics'
    )
    parser.add_argument(
        '--generate-all',
        action='store_true',
        help='Generate recommendation examples for all intents'
    )
    
    args = parser.parse_args()
    
    # List intents
    if args.list_intents:
        print("\n" + "=" * 60)
        print("AVAILABLE USER INTENTS")
        print("=" * 60)
        for intent, info in FitnessRecommender.USER_INTENTS.items():
            print(f"\n{intent}:")
            print(f"  {info['description']}")
            print(f"  Keywords: {', '.join(info['keywords'][:5])}...")
        print("\n" + "=" * 60)
        return
    
    # Initialize recommender
    recommender = FitnessRecommender()
    recommender.load_models()
    
    # Analyze topics
    if args.analyze_topics:
        recommender.analyze_topics()
        return
    
    # Map intents to topics and experts
    recommender.map_intents_to_topics()
    recommender.map_intents_to_experts()
    
    # Generate all examples
    if args.generate_all:
        recommender.generate_all_examples()
        return
    
    # Generate recommendations for specific intent
    if args.intent:
        recommendations = recommender.recommend(args.intent, top_k=args.top_k)
        recommender.print_recommendations(args.intent, recommendations)
    else:
        print("\nError: Please specify --intent or use --list-intents or --generate-all")
        parser.print_help()


if __name__ == "__main__":
    main()

