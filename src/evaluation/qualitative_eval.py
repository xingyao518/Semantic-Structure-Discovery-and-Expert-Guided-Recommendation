"""
Qualitative Evaluation Module for Running Advice.

This module provides rubric-based scoring for generated advice:
- Clarity
- Safety
- Personalization
- Correctness
"""

from typing import Dict, List, Tuple
import re


class QualitativeEvaluator:
    """
    Evaluates advice quality using qualitative rubrics.
    """
    
    def __init__(self):
        """Initialize qualitative evaluator."""
        pass
    
    def score_clarity(self, advice: Dict) -> Tuple[float, List[str]]:
        """
        Score advice clarity (0-1 scale).
        
        Criteria:
        - Clear language
        - Specific recommendations
        - Actionable steps
        - No jargon (unless explained)
        
        Args:
            advice: Generated advice dictionary
            
        Returns:
            Tuple of (score, feedback_comments)
        """
        advice_text = str(advice).lower()
        score = 0.0
        feedback = []
        
        # Check for specific numbers/quantities (more specific = clearer)
        has_numbers = bool(re.search(r'\d+', advice_text))
        if has_numbers:
            score += 0.25
        else:
            feedback.append("Lacks specific quantities/numbers")
        
        # Check for actionable verbs
        action_verbs = ['run', 'rest', 'drink', 'eat', 'stretch', 'consult', 
                       'increase', 'decrease', 'avoid', 'try']
        has_actions = any(verb in advice_text for verb in action_verbs)
        if has_actions:
            score += 0.25
        else:
            feedback.append("Lacks actionable recommendations")
        
        # Check for clear structure (lists, steps)
        has_structure = 'recommendation' in advice_text or 'step' in advice_text
        if has_structure:
            score += 0.25
        else:
            feedback.append("Could be more structured")
        
        # Check for jargon (negative)
        jargon_terms = ['vo2max', 'lactate threshold', 'periodization', 
                       'biomechanics', 'proprioception']
        has_jargon = any(term in advice_text for term in jargon_terms)
        if not has_jargon:
            score += 0.25
        else:
            feedback.append("Contains unexplained jargon")
        
        return score, feedback
    
    def score_safety(self, advice: Dict, risk_factors: Dict) -> Tuple[float, List[str]]:
        """
        Score advice safety (0-1 scale).
        
        Criteria:
        - No dangerous recommendations
        - Appropriate for risk level
        - Mentions safety precautions
        - No "push through pain" type advice
        
        Args:
            advice: Generated advice
            risk_factors: Risk factor labels
            
        Returns:
            Tuple of (score, feedback_comments)
        """
        advice_text = str(advice).lower()
        score = 1.0
        feedback = []
        
        # Check for dangerous phrases
        dangerous_phrases = [
            'push through pain', 'ignore pain', 'no pain no gain',
            'run through injury', 'tough it out', 'suck it up'
        ]
        
        for phrase in dangerous_phrases:
            if phrase in advice_text:
                score -= 0.3
                feedback.append(f"Contains dangerous phrase: '{phrase}'")
        
        # Check for safety warnings when needed
        if risk_factors.get('injury_risk', False):
            safety_keywords = ['rest', 'doctor', 'medical', 'stop', 'recover']
            has_safety = any(keyword in advice_text for keyword in safety_keywords)
            if has_safety:
                score += 0.1
            else:
                score -= 0.2
                feedback.append("Injury risk but no safety recommendations")
        
        if risk_factors.get('heat_risk', False):
            heat_safety = ['hydrate', 'water', 'cool', 'shade', 'avoid heat']
            has_heat_safety = any(keyword in advice_text for keyword in heat_safety)
            if has_heat_safety:
                score += 0.1
            else:
                score -= 0.2
                feedback.append("Heat risk but no heat safety recommendations")
        
        # Ensure score is in [0, 1]
        score = max(0.0, min(1.0, score))
        
        return score, feedback
    
    def score_personalization(self, 
                            advice: Dict,
                            user_context: Dict) -> Tuple[float, List[str]]:
        """
        Score advice personalization (0-1 scale).
        
        Criteria:
        - References user's specific situation
        - Appropriate for experience level
        - Considers user goals
        - Tailored recommendations
        
        Args:
            advice: Generated advice
            user_context: User context (experience, goals, etc.)
            
        Returns:
            Tuple of (score, feedback_comments)
        """
        advice_text = str(advice).lower()
        score = 0.0
        feedback = []
        
        # Check if advice considers experience level
        experience = user_context.get('experience_level', 'unknown')
        if experience != 'unknown':
            experience_terms = {
                'beginner': ['beginner', 'start', 'first', 'new'],
                'intermediate': ['intermediate', 'regular', 'consistent'],
                'advanced': ['advanced', 'elite', 'competitive']
            }
            
            if experience in experience_terms:
                has_experience_term = any(term in advice_text 
                                        for term in experience_terms[experience])
                if has_experience_term:
                    score += 0.25
                else:
                    feedback.append("Does not reference experience level")
        
        # Check if advice considers goals
        goals = [k for k, v in user_context.items() if k.startswith('goal_') and v]
        if goals:
            goal_terms = {
                'goal_5k': ['5k', '5 km'],
                'goal_10k': ['10k', '10 km'],
                'goal_half_marathon': ['half marathon', '13.1'],
                'goal_marathon': ['marathon', '26.2']
            }
            
            has_goal_reference = False
            for goal in goals:
                if goal in goal_terms:
                    if any(term in advice_text for term in goal_terms[goal]):
                        has_goal_reference = True
                        break
            
            if has_goal_reference:
                score += 0.25
            else:
                feedback.append("Does not reference user goals")
        
        # Check for personalized recommendations
        personalized_indicators = ['your', 'you', 'based on', 'considering', 
                                  'given that', 'for your']
        has_personalization = any(indicator in advice_text 
                                for indicator in personalized_indicators)
        if has_personalization:
            score += 0.25
        else:
            feedback.append("Lacks personalized language")
        
        # Check for context-specific advice
        if 'context' in advice or 'topic_mixture' in str(advice):
            score += 0.25
        else:
            feedback.append("Does not leverage user context")
        
        return score, feedback
    
    def score_correctness(self, advice: Dict, domain_knowledge: Dict = None) -> Tuple[float, List[str]]:
        """
        Score advice correctness (0-1 scale).
        
        Criteria:
        - Factually correct recommendations
        - Follows running best practices
        - No contradictory advice
        - Scientifically sound
        
        Args:
            advice: Generated advice
            domain_knowledge: Optional domain knowledge dictionary
            
        Returns:
            Tuple of (score, feedback_comments)
        """
        advice_text = str(advice).lower()
        score = 1.0
        feedback = []
        
        # Check for contradictions
        contradictions = [
            ('rest', 'run'),
            ('increase', 'decrease'),
            ('fast', 'slow'),
            ('long', 'short')
        ]
        
        for term1, term2 in contradictions:
            if term1 in advice_text and term2 in advice_text:
                # Check if they're in same context (simplified)
                # In practice, would use more sophisticated NLP
                score -= 0.1
                feedback.append(f"Potential contradiction: {term1} and {term2}")
        
        # Check for best practices
        best_practices = ['gradual', 'progression', 'listen to body', 
                         'warm up', 'cool down', 'stretch']
        has_best_practices = any(practice in advice_text for practice in best_practices)
        if has_best_practices:
            score += 0.1
        else:
            feedback.append("Could include more best practices")
        
        # Check for unrealistic recommendations
        # Extract mileage
        mileage_pattern = r'(\d+)\s*(?:mile|miles)'
        mileage_matches = re.findall(mileage_pattern, advice_text)
        
        if mileage_matches:
            max_mileage = max([int(m) for m in mileage_matches])
            if max_mileage > 100:  # Unrealistic for most runners
                score -= 0.2
                feedback.append(f"Unrealistic mileage recommendation: {max_mileage} miles")
        
        # Ensure score is in [0, 1]
        score = max(0.0, min(1.0, score))
        
        return score, feedback
    
    def evaluate_all(self,
                    advice: Dict,
                    risk_factors: Dict,
                    user_context: Dict,
                    domain_knowledge: Dict = None) -> Dict:
        """
        Run all qualitative evaluations.
        
        Args:
            advice: Generated advice
            risk_factors: Risk factor labels
            user_context: User context
            domain_knowledge: Optional domain knowledge
            
        Returns:
            Dictionary with all scores and feedback
        """
        clarity_score, clarity_feedback = self.score_clarity(advice)
        safety_score, safety_feedback = self.score_safety(advice, risk_factors)
        personalization_score, personalization_feedback = self.score_personalization(
            advice, user_context
        )
        correctness_score, correctness_feedback = self.score_correctness(
            advice, domain_knowledge
        )
        
        # Overall score (weighted average)
        overall_score = (
            0.25 * clarity_score +
            0.30 * safety_score +
            0.25 * personalization_score +
            0.20 * correctness_score
        )
        
        return {
            'overall_score': overall_score,
            'clarity': {
                'score': clarity_score,
                'feedback': clarity_feedback
            },
            'safety': {
                'score': safety_score,
                'feedback': safety_feedback
            },
            'personalization': {
                'score': personalization_score,
                'feedback': personalization_feedback
            },
            'correctness': {
                'score': correctness_score,
                'feedback': correctness_feedback
            }
        }
    
    def rate_advice_clarity(self, advice: Dict) -> float:
        """
        Rate advice clarity on 0-1 scale.
        
        Args:
            advice: Generated advice dictionary
            
        Returns:
            Clarity score (0.0 to 1.0)
        """
        score, _ = self.score_clarity(advice)
        return score
    
    def rate_advice_safety(self, advice: Dict, risk_factors: Dict = None) -> float:
        """
        Rate advice safety on 0-1 scale.
        
        Args:
            advice: Generated advice dictionary
            risk_factors: Optional risk factor labels
            
        Returns:
            Safety score (0.0 to 1.0)
        """
        if risk_factors is None:
            risk_factors = {}
        score, _ = self.score_safety(advice, risk_factors)
        return score
    
    def rate_advice_personalization(self, advice: Dict, user_context: Dict = None) -> float:
        """
        Rate advice personalization on 0-1 scale.
        
        Args:
            advice: Generated advice dictionary
            user_context: Optional user context dictionary
            
        Returns:
            Personalization score (0.0 to 1.0)
        """
        if user_context is None:
            user_context = {}
        score, _ = self.score_personalization(advice, user_context)
        return score
    
    def rate_overall_quality(self, advice: Dict, risk_factors: Dict = None, 
                            user_context: Dict = None) -> float:
        """
        Rate overall advice quality on 0-1 scale.
        
        Args:
            advice: Generated advice dictionary
            risk_factors: Optional risk factor labels
            user_context: Optional user context dictionary
            
        Returns:
            Overall quality score (0.0 to 1.0)
        """
        if risk_factors is None:
            risk_factors = {}
        if user_context is None:
            user_context = {}
        
        results = self.evaluate_all(advice, risk_factors, user_context)
        return results['overall_score']


if __name__ == "__main__":
    evaluator = QualitativeEvaluator()
    
    # Example
    sample_advice = {
        'recommendations': ['Rest for 2-3 days', 'Ice the affected area', 
                          'Gradually return to running']
    }
    risk_factors = {'injury_risk': True}
    user_context = {'experience_level': 'beginner', 'goal_5k': True}
    
    results = evaluator.evaluate_all(sample_advice, risk_factors, user_context)
    print(f"Evaluation results: {results}")

