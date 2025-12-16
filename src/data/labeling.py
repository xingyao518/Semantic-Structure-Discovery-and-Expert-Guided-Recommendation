"""
Rule-based labeling for running Q&A data.

This module assigns labels to documents based on keyword matching:
- Risk factors (injury, heat, nutrition)
- Experience level (beginner/intermediate/advanced)
- Running goals (5k/10k/half-marathon/marathon)
"""

import re
from typing import List, Dict, Set
from collections import defaultdict


class RunningDataLabeler:
    """
    Rule-based labeler for running-related documents.
    
    Uses keyword matching to assign multiple labels per document.
    """
    
    def __init__(self):
        """Initialize labeler with keyword dictionaries."""
        self._init_keywords()
    
    def _init_keywords(self):
        """Initialize keyword dictionaries for different categories."""
        
        # Injury-related keywords
        self.injury_keywords = {
            'pain', 'hurt', 'injury', 'injured', 'ache', 'sore', 'strain',
            'sprain', 'tendonitis', 'shin', 'knee', 'ankle', 'hip', 'IT band',
            'plantar', 'fascia', 'stress fracture', 'muscle', 'ligament',
            'recovery', 'heal', 'rehab', 'physical therapy', 'PT'
        }
        
        # Heat/hydration keywords
        self.heat_keywords = {
            'heat', 'hot', 'temperature', 'humidity', 'dehydration', 'hydrate',
            'water', 'sweat', 'sweating', 'electrolyte', 'salt', 'cramp',
            'sun', 'sunburn', 'heat stroke', 'heat exhaustion', 'cooling',
            'ice', 'cold', 'freezing'
        }
        
        # Nutrition keywords
        self.nutrition_keywords = {
            'nutrition', 'diet', 'food', 'eat', 'eating', 'meal', 'carb',
            'carbohydrate', 'protein', 'iron', 'vitamin', 'supplement',
            'energy', 'fuel', 'gel', 'gu', 'pre-run', 'post-run', 'recovery meal',
            'calorie', 'calories', 'weight', 'lose weight', 'gain weight'
        }
        
        # Experience level indicators
        self.beginner_keywords = {
            'beginner', 'new', 'start', 'starting', 'first', 'never', 'just started',
            'week', 'weeks', 'month', 'months', 'couch', '5k', 'first run',
            'walk', 'walking', 'jog', 'jogging', 'slow', 'pace'
        }
        
        self.intermediate_keywords = {
            'intermediate', 'regular', 'consistent', 'training', 'workout',
            '10k', 'half marathon', 'half-marathon', 'HM', 'race', 'racing',
            'pace', 'speed', 'tempo', 'interval', 'long run'
        }
        
        self.advanced_keywords = {
            'advanced', 'elite', 'marathon', 'ultra', 'ultramarathon', 'trail',
            'competitive', 'PR', 'personal record', 'BQ', 'Boston', 'qualify',
            'training plan', 'periodization', 'taper', 'peak'
        }
        
        # Goal-related keywords
        self.goal_5k_keywords = {'5k', '5 km', '5 kilometer', 'parkrun'}
        self.goal_10k_keywords = {'10k', '10 km', '10 kilometer'}
        self.goal_hm_keywords = {
            'half marathon', 'half-marathon', 'HM', '13.1', '21k', '21 km'
        }
        self.goal_fm_keywords = {
            'marathon', '26.2', '42k', '42 km', 'full marathon'
        }
    
    def count_keyword_matches(self, text: str, keywords: Set[str]) -> int:
        """
        Count how many keywords appear in text (case-insensitive).
        
        Args:
            text: Input text
            keywords: Set of keywords to search for
            
        Returns:
            Number of unique keywords found
        """
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        return matches
    
    def label_risk_factors(self, text: str) -> Dict[str, bool]:
        """
        Label risk factor categories.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with risk factor labels (injury, heat, nutrition)
        """
        injury_count = self.count_keyword_matches(text, self.injury_keywords)
        heat_count = self.count_keyword_matches(text, self.heat_keywords)
        nutrition_count = self.count_keyword_matches(text, self.nutrition_keywords)
        
        return {
            'injury_risk': injury_count > 0,
            'heat_risk': heat_count > 0,
            'nutrition_risk': nutrition_count > 0
        }
    
    def label_experience_level(self, text: str) -> str:
        """
        Label experience level.
        
        Args:
            text: Input text
            
        Returns:
            Experience level: 'beginner', 'intermediate', 'advanced', or 'unknown'
        """
        beginner_count = self.count_keyword_matches(text, self.beginner_keywords)
        intermediate_count = self.count_keyword_matches(text, self.intermediate_keywords)
        advanced_count = self.count_keyword_matches(text, self.advanced_keywords)
        
        counts = {
            'beginner': beginner_count,
            'intermediate': intermediate_count,
            'advanced': advanced_count
        }
        
        max_level = max(counts, key=counts.get)
        
        if counts[max_level] > 0:
            return max_level
        else:
            return 'unknown'
    
    def label_goals(self, text: str) -> Dict[str, bool]:
        """
        Label running goals.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with goal labels (5k, 10k, half_marathon, marathon)
        """
        return {
            'goal_5k': self.count_keyword_matches(text, self.goal_5k_keywords) > 0,
            'goal_10k': self.count_keyword_matches(text, self.goal_10k_keywords) > 0,
            'goal_half_marathon': self.count_keyword_matches(text, self.goal_hm_keywords) > 0,
            'goal_marathon': self.count_keyword_matches(text, self.goal_fm_keywords) > 0
        }
    
    def label_document(self, text: str) -> Dict:
        """
        Label a single document with all categories.
        
        Args:
            text: Input text document
            
        Returns:
            Dictionary with all labels
        """
        labels = {}
        
        # Risk factors
        risk_labels = self.label_risk_factors(text)
        labels.update(risk_labels)
        
        # Experience level
        labels['experience_level'] = self.label_experience_level(text)
        
        # Goals
        goal_labels = self.label_goals(text)
        labels.update(goal_labels)
        
        return labels
    
    def label_documents(self, documents: List[str]) -> List[Dict]:
        """
        Label multiple documents.
        
        Args:
            documents: List of text documents
            
        Returns:
            List of label dictionaries
        """
        return [self.label_document(doc) for doc in documents]
    
    def get_label_statistics(self, labels: List[Dict]) -> Dict:
        """
        Compute statistics over a collection of labels.
        
        Args:
            labels: List of label dictionaries
            
        Returns:
            Dictionary with label counts and percentages
        """
        stats = defaultdict(int)
        total = len(labels)
        
        for label_dict in labels:
            for key, value in label_dict.items():
                if isinstance(value, bool) and value:
                    stats[key] += 1
                elif isinstance(value, str):
                    stats[f"{key}_{value}"] += 1
        
        # Convert to percentages
        stats_pct = {k: v / total for k, v in stats.items()}
        
        return {
            'counts': dict(stats),
            'percentages': stats_pct,
            'total_documents': total
        }


if __name__ == "__main__":
    labeler = RunningDataLabeler()
    
    # Example
    sample_text = "I'm a beginner runner training for my first 5k. My knees hurt after running. Should I rest?"
    labels = labeler.label_document(sample_text)
    print(f"Text: {sample_text}")
    print(f"Labels: {labels}")

