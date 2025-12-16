"""
Generate Improved Recommendation Flow Visualization (v2)

Creates a poster-ready, left-to-right flow diagram with:
- Large, clear typography
- Credible examples
- Minimal notation
- Clean layout
- Subtle dataset stats in footer
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

def create_recommendation_flow_v2():
    """Create an improved, poster-ready recommendation flow diagram."""
    
    # Create figure with better aspect ratio for posters
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Define colors (pastel, professional)
    color_intent = '#5DADE2'      # Light Blue
    color_map = '#EC7063'          # Light Red
    color_route = '#58D68D'        # Light Green
    color_filter = '#F8C471'       # Light Yellow/Orange
    color_rank = '#AF7AC5'         # Light Purple
    color_result = '#85929E'       # Gray
    
    # Main title
    ax.text(8, 5.5, 'Expert-Guided Recommendation Pipeline', 
            fontsize=20, fontweight='bold', ha='center', va='top')
    
    # Subtitle
    ax.text(8, 5.1, 'Intent-based document retrieval using LDA topics and mixture-of-experts', 
            fontsize=11, ha='center', va='top', style='italic', color='#555')
    
    # Y-position for all boxes (aligned horizontally)
    box_y = 2.2
    box_height = 2.0
    box_width = 2.2
    
    # ===== Step 1: User Intent =====
    x1 = 0.5
    box1 = FancyBboxPatch((x1, box_y), box_width, box_height, 
                          boxstyle="round,pad=0.15", 
                          edgecolor=color_intent, 
                          facecolor=color_intent, 
                          alpha=0.25, 
                          linewidth=3)
    ax.add_patch(box1)
    ax.text(x1 + box_width/2, box_y + box_height - 0.35, 'User Intent', 
            fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(x1 + box_width/2, box_y + box_height/2, 'injury_recovery', 
            fontsize=11, ha='center', va='center', style='italic', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # ===== Arrow 1 =====
    arrow1 = FancyArrowPatch((x1 + box_width, box_y + box_height/2), 
                            (x1 + box_width + 0.5, box_y + box_height/2),
                            arrowstyle='->', 
                            mutation_scale=40, 
                            linewidth=3,
                            color='#2C3E50')
    ax.add_patch(arrow1)
    ax.text(x1 + box_width + 0.25, box_y + box_height/2 + 0.5, 'Keyword\nMatch', 
            fontsize=9, ha='center', va='bottom', color='#2C3E50')
    
    # ===== Step 2: Intent→Topic Mapping =====
    x2 = x1 + box_width + 0.5
    box2 = FancyBboxPatch((x2, box_y), box_width, box_height, 
                          boxstyle="round,pad=0.15", 
                          edgecolor=color_map, 
                          facecolor=color_map, 
                          alpha=0.25, 
                          linewidth=3)
    ax.add_patch(box2)
    ax.text(x2 + box_width/2, box_y + box_height - 0.35, 'Topic Mapping', 
            fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(x2 + box_width/2, box_y + box_height/2 + 0.3, 'Topic 2', 
            fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(x2 + box_width/2, box_y + box_height/2 - 0.2, 'injury, knee,\npain, recovery', 
            fontsize=9, ha='center', va='center', style='italic', color='#555')
    
    # ===== Arrow 2 =====
    arrow2 = FancyArrowPatch((x2 + box_width, box_y + box_height/2), 
                            (x2 + box_width + 0.5, box_y + box_height/2),
                            arrowstyle='->', 
                            mutation_scale=40, 
                            linewidth=3,
                            color='#2C3E50')
    ax.add_patch(arrow2)
    ax.text(x2 + box_width + 0.25, box_y + box_height/2 + 0.5, 'Topic\nOverlap', 
            fontsize=9, ha='center', va='bottom', color='#2C3E50')
    
    # ===== Step 3: Expert Routing =====
    x3 = x2 + box_width + 0.5
    box3 = FancyBboxPatch((x3, box_y), box_width, box_height, 
                          boxstyle="round,pad=0.15", 
                          edgecolor=color_route, 
                          facecolor=color_route, 
                          alpha=0.25, 
                          linewidth=3)
    ax.add_patch(box3)
    ax.text(x3 + box_width/2, box_y + box_height - 0.35, 'Expert Routing', 
            fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(x3 + box_width/2, box_y + box_height/2 + 0.3, 'Experts 0, 3', 
            fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(x3 + box_width/2, box_y + box_height/2 - 0.2, 'Dominant on\nTopics 2, 5, 6', 
            fontsize=9, ha='center', va='center', style='italic', color='#555')
    
    # ===== Arrow 3 =====
    arrow3 = FancyArrowPatch((x3 + box_width, box_y + box_height/2), 
                            (x3 + box_width + 0.5, box_y + box_height/2),
                            arrowstyle='->', 
                            mutation_scale=40, 
                            linewidth=3,
                            color='#2C3E50')
    ax.add_patch(arrow3)
    ax.text(x3 + box_width + 0.25, box_y + box_height/2 + 0.5, 'Filter', 
            fontsize=9, ha='center', va='bottom', color='#2C3E50')
    
    # ===== Step 4: Candidate Filtering =====
    x4 = x3 + box_width + 0.5
    box4 = FancyBboxPatch((x4, box_y), box_width, box_height, 
                          boxstyle="round,pad=0.15", 
                          edgecolor=color_filter, 
                          facecolor=color_filter, 
                          alpha=0.25, 
                          linewidth=3)
    ax.add_patch(box4)
    ax.text(x4 + box_width/2, box_y + box_height - 0.35, 'Candidate Filter', 
            fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(x4 + box_width/2, box_y + box_height/2 + 0.25, '1,501 docs', 
            fontsize=13, ha='center', va='center', fontweight='bold')
    ax.text(x4 + box_width/2, box_y + box_height/2 - 0.25, 'High expert\nresponsibility', 
            fontsize=9, ha='center', va='center', style='italic', color='#555')
    
    # ===== Arrow 4 =====
    arrow4 = FancyArrowPatch((x4 + box_width, box_y + box_height/2), 
                            (x4 + box_width + 0.5, box_y + box_height/2),
                            arrowstyle='->', 
                            mutation_scale=40, 
                            linewidth=3,
                            color='#2C3E50')
    ax.add_patch(arrow4)
    ax.text(x4 + box_width + 0.25, box_y + box_height/2 + 0.5, 'Rank', 
            fontsize=9, ha='center', va='bottom', color='#2C3E50')
    
    # ===== Step 5: Top-K Results =====
    x5 = x4 + box_width + 0.5
    box5 = FancyBboxPatch((x5, box_y), box_width, box_height, 
                          boxstyle="round,pad=0.15", 
                          edgecolor=color_result, 
                          facecolor=color_result, 
                          alpha=0.25, 
                          linewidth=3)
    ax.add_patch(box5)
    ax.text(x5 + box_width/2, box_y + box_height - 0.35, 'Top-K Results', 
            fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(x5 + box_width/2, box_y + box_height/2 + 0.35, 'Doc 3703', 
            fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(x5 + box_width/2, box_y + box_height/2 + 0.05, 'Score: 0.964', 
            fontsize=10, ha='center', va='center')
    ax.text(x5 + box_width/2, box_y + box_height/2 - 0.35, 'Doc 1274', 
            fontsize=10, ha='center', va='center')
    ax.text(x5 + box_width/2, box_y + box_height/2 - 0.6, 'Score: 0.959', 
            fontsize=9, ha='center', va='center', color='#555')
    
    # ===== Footer with dataset stats (subtle) =====
    footer_text = "Dataset: 5,000 docs | 10 topics | 5 experts | Vocab: 7,561 | 6 intents | Method: LDA Gibbs + VI + Mixture-of-Experts"
    ax.text(8, 0.4, footer_text, 
            fontsize=9, ha='center', va='center', 
            color='#7F8C8D', style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECF0F1', alpha=0.6, edgecolor='none'))
    
    # ===== Method annotation (top) =====
    method_text = "Pipeline: Keyword-based intent mapping → Topic-expert alignment → Responsibility-based filtering → Similarity ranking"
    ax.text(8, 1.3, method_text, 
            fontsize=9, ha='center', va='center', 
            color='#34495E', style='italic')
    
    # Save figure
    output_path = "data/processed/visualizations/recommendation_flow_v2.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n[SUCCESS] Improved recommendation flow diagram (v2) saved to:")
    print(f"  {output_path}")
    print(f"\nImprovements:")
    print(f"  * Left-to-right horizontal flow (single row)")
    print(f"  * Large, poster-friendly typography")
    print(f"  * Credible topic words (injury, knee, pain, recovery)")
    print(f"  * Dataset stats moved to subtle footer")
    print(f"  * Minimal notation, clean arrows")
    print(f"  * Professional color palette")
    print(f"\nOriginal diagram preserved at:")
    print(f"  data/processed/visualizations/recommendation_flow.png")
    print()


if __name__ == "__main__":
    try:
        create_recommendation_flow_v2()
    except Exception as e:
        print(f"\n[ERROR] Failed to generate v2 visualization: {e}")
        import traceback
        traceback.print_exc()

