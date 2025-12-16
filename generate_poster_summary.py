"""
Generate Poster-Friendly Summary of Fitness Recommendations

This script loads recommendation examples and creates a clean summary
suitable for presentation posters.
"""

import json
import sys

def create_poster_summary():
    """Create a poster-friendly summary of recommendations."""
    
    # Load recommendation examples
    print("Loading recommendation examples...")
    with open("data/processed/recommendation_examples.json", "r", encoding="utf-8") as f:
        all_examples = json.load(f)
    
    # Select representative intents
    selected_intents = ["injury_recovery", "beginner_guidance", "nutrition"]
    
    # Build poster summary
    poster_data = {}
    
    print("\n" + "=" * 80)
    print("FITNESS RECOMMENDATION SYSTEM - POSTER SUMMARY")
    print("=" * 80)
    print("\nIntent-Based Recommendations Using LDA Topics + Mixture-of-Experts")
    print("=" * 80)
    
    for intent in selected_intents:
        if intent not in all_examples:
            continue
        
        data = all_examples[intent]
        
        # Extract info
        intent_display = intent.replace("_", " ").title()
        description = data['description']
        mapped_topics = data['mapped_topics']
        mapped_experts = data['mapped_experts']
        top_2_docs = data['recommendations'][:2]
        
        # Print formatted output
        print(f"\n{'=' * 80}")
        print(f"INTENT: {intent_display}")
        print(f"{'=' * 80}")
        print(f"Description: {description}")
        print(f"Mapped Topics: {mapped_topics if mapped_topics else 'General (fallback)'}")
        print(f"Mapped Experts: {mapped_experts}")
        print(f"\nTop 2 Recommendations:")
        print("-" * 80)
        
        recommendations_summary = []
        
        for i, rec in enumerate(top_2_docs, 1):
            doc_id = rec['document_id']
            score = rec['score']
            topic = rec['dominant_topic']
            expert = rec['dominant_expert']
            snippet = rec['text_snippet'][:100].replace('\n', ' ')
            
            print(f"\n  [{i}] Document {doc_id} (Score: {score:.3f})")
            print(f"      Topic: {topic} | Expert: {expert}")
            print(f"      \"{snippet}...\"")
            
            recommendations_summary.append({
                "rank": i,
                "document_id": doc_id,
                "relevance_score": round(score, 3),
                "dominant_topic": topic,
                "dominant_expert": expert,
                "snippet": snippet + "..."
            })
        
        # Store for JSON
        poster_data[intent] = {
            "intent_display": intent_display,
            "description": description,
            "mapped_topics": mapped_topics if mapped_topics else ["general"],
            "mapped_experts": mapped_experts,
            "top_2_recommendations": recommendations_summary,
            "explanation": f"Documents filtered by expert {mapped_experts[0] if mapped_experts else 0} and ranked by topic relevance"
        }
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("* System successfully maps user intents to LDA topics and expert clusters")
    print("* Intent-to-topic mapping uses keyword overlap with topic word distributions")
    print("* Candidate filtering leverages mixture-of-experts responsibilities")
    print("* Final ranking based on LDA topic similarity scores")
    print("* Tested on 5,000 fitness documents with 10 topics and 5 experts")
    print("=" * 80 + "\n")
    
    # Save poster summary
    output_path = "data/processed/poster_recommendation_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(poster_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Poster summary saved to: {output_path}")
    print("\nFull recommendation details available in:")
    print("  data/processed/recommendation_examples.json")
    print()
    
    return poster_data


if __name__ == "__main__":
    try:
        create_poster_summary()
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("Please ensure recommendation_examples.json exists.")
        print("Run: python -m src.recommendation.fitness_recommender --generate-all")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

