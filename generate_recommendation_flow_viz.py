"""
Generate Recommendation Flow Visualization

Creates a simple block diagram showing the recommendation pipeline:
User Intent → LDA Topics → Mixture Experts → Candidate Documents → Top-K Recommendation
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

def create_recommendation_flow_diagram():
    """Create a block diagram showing the recommendation flow."""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    color_intent = '#3498db'      # Blue
    color_topic = '#e74c3c'       # Red
    color_expert = '#2ecc71'      # Green
    color_candidate = '#f39c12'   # Orange
    color_result = '#9b59b6'      # Purple
    
    # Title
    ax.text(5, 9.5, 'Expert-Guided Fitness Recommendation Flow', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Step 1: User Intent
    box1 = FancyBboxPatch((0.5, 7), 1.5, 1, 
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_intent, 
                          facecolor=color_intent, 
                          alpha=0.3, 
                          linewidth=2)
    ax.add_patch(box1)
    ax.text(1.25, 7.5, 'User Intent', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(1.25, 7.2, 'e.g., injury_recovery', fontsize=8, ha='center', va='center', style='italic')
    
    # Arrow 1
    arrow1 = FancyArrowPatch((2.0, 7.5), (2.8, 7.5),
                            arrowstyle='->', 
                            mutation_scale=30, 
                            linewidth=2.5,
                            color='black')
    ax.add_patch(arrow1)
    ax.text(2.4, 7.8, 'Keyword\nMatching', fontsize=8, ha='center', va='bottom', style='italic')
    
    # Step 2: LDA Topics
    box2 = FancyBboxPatch((2.8, 6.5), 1.8, 2, 
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_topic, 
                          facecolor=color_topic, 
                          alpha=0.3, 
                          linewidth=2)
    ax.add_patch(box2)
    ax.text(3.7, 7.8, 'LDA Topics', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(3.7, 7.4, 'Topic 2:\ntime, work, like', fontsize=8, ha='center', va='center')
    ax.text(3.7, 6.9, 'φ (topic-word)', fontsize=8, ha='center', va='center', style='italic')
    
    # Arrow 2
    arrow2 = FancyArrowPatch((4.6, 7.5), (5.4, 7.5),
                            arrowstyle='->', 
                            mutation_scale=30, 
                            linewidth=2.5,
                            color='black')
    ax.add_patch(arrow2)
    ax.text(5.0, 7.8, 'Topic\nOverlap', fontsize=8, ha='center', va='bottom', style='italic')
    
    # Step 3: Mixture Experts
    box3 = FancyBboxPatch((5.4, 6.5), 1.8, 2, 
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_expert, 
                          facecolor=color_expert, 
                          alpha=0.3, 
                          linewidth=2)
    ax.add_patch(box3)
    ax.text(6.3, 7.8, 'Mixture Experts', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(6.3, 7.4, 'Expert 0, 3', fontsize=8, ha='center', va='center')
    ax.text(6.3, 7.1, 'Dominant topics:', fontsize=7, ha='center', va='center')
    ax.text(6.3, 6.85, '[6, 2, 5]', fontsize=7, ha='center', va='center')
    
    # Arrow 3
    arrow3 = FancyArrowPatch((7.2, 7.5), (8.0, 6.5),
                            arrowstyle='->', 
                            mutation_scale=30, 
                            linewidth=2.5,
                            color='black')
    ax.add_patch(arrow3)
    ax.text(7.6, 7.3, 'Filter by\nResponsibility', fontsize=8, ha='center', va='bottom', style='italic')
    
    # Step 4: Candidate Documents
    box4 = FancyBboxPatch((7.5, 4.5), 2, 1.8, 
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_candidate, 
                          facecolor=color_candidate, 
                          alpha=0.3, 
                          linewidth=2)
    ax.add_patch(box4)
    ax.text(8.5, 5.7, 'Candidate Docs', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(8.5, 5.35, '1,501 documents', fontsize=8, ha='center', va='center')
    ax.text(8.5, 5.05, 'γ(expert|doc) > threshold', fontsize=7, ha='center', va='center', style='italic')
    
    # Arrow 4
    arrow4 = FancyArrowPatch((8.5, 4.5), (8.5, 3.5),
                            arrowstyle='->', 
                            mutation_scale=30, 
                            linewidth=2.5,
                            color='black')
    ax.add_patch(arrow4)
    ax.text(8.9, 4.0, 'Rank by\nTopic Score', fontsize=8, ha='left', va='center', style='italic')
    
    # Step 5: Top-K Results
    box5 = FancyBboxPatch((7.5, 2), 2, 1.3, 
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_result, 
                          facecolor=color_result, 
                          alpha=0.3, 
                          linewidth=2)
    ax.add_patch(box5)
    ax.text(8.5, 2.85, 'Top-K Results', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(8.5, 2.5, 'Doc 3703: 0.964', fontsize=8, ha='center', va='center')
    ax.text(8.5, 2.25, 'Doc 1274: 0.959', fontsize=8, ha='center', va='center')
    
    # Add side annotations
    # Left side: Input
    ax.text(0.2, 7.5, 'INPUT', fontsize=10, ha='center', va='center', 
            fontweight='bold', rotation=90, color='gray')
    
    # Right side: Output
    ax.text(9.8, 2.65, 'OUTPUT', fontsize=10, ha='center', va='center', 
            fontweight='bold', rotation=90, color='gray')
    
    # Bottom: Method description
    method_text = (
        "Method: Intent keywords → Topic word distributions (φ) → Expert-topic alignment (θ) → "
        "Responsibility filtering (γ) → Topic similarity ranking"
    )
    ax.text(5, 0.8, method_text, fontsize=8, ha='center', va='center', 
            style='italic', color='#555', wrap=True)
    
    # Add legend for symbols
    legend_elements = [
        mpatches.Patch(facecolor=color_intent, alpha=0.3, edgecolor=color_intent, label='User Input', linewidth=2),
        mpatches.Patch(facecolor=color_topic, alpha=0.3, edgecolor=color_topic, label='LDA Model', linewidth=2),
        mpatches.Patch(facecolor=color_expert, alpha=0.3, edgecolor=color_expert, label='Mixture Model', linewidth=2),
        mpatches.Patch(facecolor=color_candidate, alpha=0.3, edgecolor=color_candidate, label='Filtering', linewidth=2),
        mpatches.Patch(facecolor=color_result, alpha=0.3, edgecolor=color_result, label='Recommendations', linewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, frameon=True, 
             fancybox=True, shadow=True, ncol=5, bbox_to_anchor=(0.05, 0.02))
    
    # Add dataset info box
    info_box = FancyBboxPatch((0.5, 4.5), 2.5, 2.5, 
                              boxstyle="round,pad=0.15", 
                              edgecolor='#34495e', 
                              facecolor='#ecf0f1', 
                              alpha=0.8, 
                              linewidth=1.5)
    ax.add_patch(info_box)
    ax.text(1.75, 6.6, 'Dataset Info', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(1.75, 6.25, '• 5,000 fitness docs', fontsize=8, ha='center', va='center')
    ax.text(1.75, 5.95, '• 10 LDA topics', fontsize=8, ha='center', va='center')
    ax.text(1.75, 5.65, '• 5 Mixture experts', fontsize=8, ha='center', va='center')
    ax.text(1.75, 5.35, '• Vocab: 7,561 words', fontsize=8, ha='center', va='center')
    ax.text(1.75, 5.05, '• 6 user intents', fontsize=8, ha='center', va='center')
    ax.text(1.75, 4.75, '• Gibbs + VI training', fontsize=8, ha='center', va='center')
    
    # Save figure
    output_path = "data/processed/visualizations/recommendation_flow.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n[SUCCESS] Recommendation flow diagram saved to:")
    print(f"  {output_path}")
    print(f"\nVisualization shows:")
    print(f"  1. User Intent → LDA Topics (keyword matching)")
    print(f"  2. LDA Topics → Mixture Experts (topic overlap)")
    print(f"  3. Mixture Experts → Candidate Docs (responsibility filtering)")
    print(f"  4. Candidate Docs → Top-K Results (topic similarity ranking)")
    print()


if __name__ == "__main__":
    try:
        create_recommendation_flow_diagram()
    except Exception as e:
        print(f"\n[ERROR] Failed to generate visualization: {e}")
        import traceback
        traceback.print_exc()

