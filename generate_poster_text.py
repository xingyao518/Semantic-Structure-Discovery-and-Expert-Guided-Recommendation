"""
Generate Compact Poster Text from Recommendation Summary

Creates clean, concise text suitable for direct copy-paste into posters.
"""

import json

def generate_poster_text():
    """Generate compact poster-ready text from recommendation summary."""
    
    # Load poster summary
    with open("data/processed/poster_recommendation_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION SYSTEM DEMO - POSTER TEXT")
    print("=" * 70)
    print("\n[Copy the text below directly into your poster]\n")
    print("-" * 70)
    
    # Process each intent
    intents = ["injury_recovery", "beginner_guidance", "nutrition"]
    
    for idx, intent_key in enumerate(intents, 1):
        if intent_key not in summary:
            continue
        
        data = summary[intent_key]
        intent_name = data['intent_display']
        topics = data['mapped_topics']
        experts = data['mapped_experts']
        top_rec = data['top_2_recommendations'][0]  # Get the best one
        
        # Format topic and expert lists
        topic_str = f"Topic {topics[0]}" if len(topics) == 1 else f"Topics {', '.join(map(str, topics))}"
        expert_str = f"Expert {experts[0]}" if len(experts) == 1 else f"Experts {', '.join(map(str, experts))}"
        
        # Clean up snippet
        snippet = top_rec['snippet'].replace('...', '').strip()
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        
        # Print compact format
        print(f"\n{idx}. {intent_name.upper()}")
        print(f"   {topic_str} | {expert_str}")
        print(f"   Best match: Doc {top_rec['document_id']} (score: {top_rec['relevance_score']:.3f})")
        print(f'   "{snippet}"')
    
    print("\n" + "-" * 70)
    print("\nMethod: Intent keywords → LDA topic matching → Expert routing")
    print("        → Responsibility filtering → Topic similarity ranking")
    print("\nCorpus: 5,000 fitness documents | 10 topics | 5 experts")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        generate_poster_text()
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("Please run: python generate_poster_summary.py first")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

