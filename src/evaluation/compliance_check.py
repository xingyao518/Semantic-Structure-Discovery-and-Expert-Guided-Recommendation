"""
Compliance Check Module for Running Advice.

This module checks whether generated advice complies with safety constraints:
- Injury constraints (rest recommendations for injuries)
- Weather constraints (heat/humidity warnings)
- Experience level constraints (appropriate mileage)
- Safe progression (10% rule)
"""

from typing import Dict, List, Tuple
import re


class ComplianceChecker:
    """
    Checks compliance of running advice with safety constraints.
    """
    
    def __init__(self):
        """Initialize compliance checker."""
        self.violations = []
    
    def check_injury_constraints(self, advice: Dict, risk_factors: Dict) -> Tuple[bool, List[str]]:
        """
        Check if advice respects injury constraints.
        
        If injury_risk is True, advice should recommend rest/recovery.
        
        Args:
            advice: Generated advice dictionary
            risk_factors: Risk factor labels
            
        Returns:
            Tuple of (is_compliant, list_of_violations)
        """
        violations = []
        is_compliant = True
        
        if risk_factors.get('injury_risk', False):
            # Check if advice mentions rest, recovery, or medical attention
            advice_text = str(advice).lower()
            
            rest_keywords = ['rest', 'recover', 'stop', 'cease', 'pause', 
                           'doctor', 'medical', 'physician', 'pt', 'physical therapy']
            
            has_rest_mention = any(keyword in advice_text for keyword in rest_keywords)
            
            if not has_rest_mention:
                violations.append("Injury risk detected but no rest/recovery recommendation")
                is_compliant = False
            
            # Check for dangerous recommendations
            dangerous_keywords = ['push through', 'ignore pain', 'run anyway', 
                                'no pain no gain']
            has_dangerous = any(keyword in advice_text for keyword in dangerous_keywords)
            
            if has_dangerous:
                violations.append("Dangerous advice detected for injury risk")
                is_compliant = False
        
        return is_compliant, violations
    
    def check_weather_constraints(self, advice: Dict, risk_factors: Dict, 
                                  weather_data: Dict = None) -> Tuple[bool, List[str]]:
        """
        Check if advice respects weather/heat constraints.
        
        Args:
            advice: Generated advice
            risk_factors: Risk factor labels
            weather_data: Optional weather data dictionary
            
        Returns:
            Tuple of (is_compliant, list_of_violations)
        """
        violations = []
        is_compliant = True
        
        if risk_factors.get('heat_risk', False):
            advice_text = str(advice).lower()
            
            # Check for heat-related recommendations
            heat_keywords = ['hydrate', 'water', 'cool', 'shade', 'early morning',
                           'evening', 'avoid heat', 'electrolyte']
            
            has_heat_mention = any(keyword in advice_text for keyword in heat_keywords)
            
            if not has_heat_mention:
                violations.append("Heat risk detected but no heat-related recommendations")
                is_compliant = False
        
        # Check weather data if provided
        if weather_data:
            temp = weather_data.get('temperature', None)
            humidity = weather_data.get('humidity', None)
            
            if temp and temp > 80:  # Fahrenheit
                advice_text = str(advice).lower()
                if 'heat' not in advice_text and 'hot' not in advice_text:
                    violations.append("High temperature but no heat warning")
                    is_compliant = False
        
        return is_compliant, violations
    
    def check_experience_level(self, advice: Dict, user_experience: str) -> Tuple[bool, List[str]]:
        """
        Check if advice is appropriate for user's experience level.
        
        Args:
            advice: Generated advice
            user_experience: Experience level ('beginner', 'intermediate', 'advanced')
            
        Returns:
            Tuple of (is_compliant, list_of_violations)
        """
        violations = []
        is_compliant = True
        
        advice_text = str(advice).lower()
        
        # Extract mileage recommendations (simplified)
        mileage_pattern = r'(\d+)\s*(?:mile|miles|km|kilometer)'
        mileage_matches = re.findall(mileage_pattern, advice_text)
        
        if mileage_matches:
            max_mileage = max([int(m) for m in mileage_matches])
            
            if user_experience == 'beginner' and max_mileage > 5:
                violations.append(f"Too high mileage ({max_mileage} miles) for beginner")
                is_compliant = False
            elif user_experience == 'intermediate' and max_mileage > 15:
                violations.append(f"Potentially too high mileage ({max_mileage} miles) for intermediate")
                # Warning, not necessarily violation
        
        # Check for advanced concepts for beginners
        if user_experience == 'beginner':
            advanced_concepts = ['tempo', 'interval', 'fartlek', 'periodization', 
                               'taper', 'vo2max', 'lactate threshold']
            has_advanced = any(concept in advice_text for concept in advanced_concepts)
            
            if has_advanced:
                violations.append("Advanced training concepts recommended for beginner")
                is_compliant = False
        
        return is_compliant, violations
    
    def check_progression_rule(self, advice: Dict, current_mileage: float = None) -> Tuple[bool, List[str]]:
        """
        Check if advice follows the 10% progression rule.
        
        The 10% rule: weekly mileage should not increase by more than 10% per week.
        
        Args:
            advice: Generated advice
            current_mileage: Current weekly mileage (if known)
            
        Returns:
            Tuple of (is_compliant, list_of_violations)
        """
        violations = []
        is_compliant = True
        
        if current_mileage is None:
            # Cannot check without baseline
            return True, []
        
        advice_text = str(advice).lower()
        
        # Extract mileage recommendations
        mileage_pattern = r'(\d+)\s*(?:mile|miles|km|kilometer)'
        mileage_matches = re.findall(mileage_pattern, advice_text)
        
        if mileage_matches:
            recommended_mileage = max([int(m) for m in mileage_matches])
            
            # Check 10% rule
            max_safe_mileage = current_mileage * 1.1
            
            if recommended_mileage > max_safe_mileage:
                violations.append(
                    f"Recommended mileage ({recommended_mileage}) exceeds 10% increase "
                    f"from current ({current_mileage:.1f})"
                )
                is_compliant = False
        
        return is_compliant, violations
    
    def check_all(self, 
                  advice: Dict,
                  risk_factors: Dict,
                  user_experience: str = 'unknown',
                  weather_data: Dict = None,
                  current_mileage: float = None) -> Dict:
        """
        Run all compliance checks.
        
        Args:
            advice: Generated advice
            risk_factors: Risk factor labels
            user_experience: User experience level
            weather_data: Optional weather data
            current_mileage: Optional current weekly mileage
            
        Returns:
            Dictionary with compliance results
        """
        results = {
            'overall_compliant': True,
            'violations': [],
            'warnings': []
        }
        
        # Injury constraints
        compliant, violations = self.check_injury_constraints(advice, risk_factors)
        if not compliant:
            results['overall_compliant'] = False
            results['violations'].extend(violations)
        
        # Weather constraints
        compliant, violations = self.check_weather_constraints(advice, risk_factors, weather_data)
        if not compliant:
            results['overall_compliant'] = False
            results['violations'].extend(violations)
        
        # Experience level
        if user_experience != 'unknown':
            compliant, violations = self.check_experience_level(advice, user_experience)
            if not compliant:
                results['overall_compliant'] = False
                results['violations'].extend(violations)
        
        # Progression rule
        if current_mileage is not None:
            compliant, violations = self.check_progression_rule(advice, current_mileage)
            if not compliant:
                results['overall_compliant'] = False
                results['violations'].extend(violations)
        
        return results
    
    def check_injury_constraints_profile(self, advice: Dict, user_profile: Dict) -> Tuple[bool, List[str]]:
        """
        Check injury constraints with user profile context.
        
        Wrapper method that extracts risk factors from user profile.
        
        Args:
            advice: Generated advice dictionary
            user_profile: User profile containing risk_factors, experience, etc.
            
        Returns:
            Tuple of (is_compliant, list_of_violations)
        """
        risk_factors = user_profile.get('risk_factors', {})
        return self.check_injury_constraints(advice, risk_factors)
    
    def check_weather_constraints_profile(self, advice: Dict, user_profile: Dict) -> Tuple[bool, List[str]]:
        """
        Check weather constraints with user profile context.
        
        Wrapper method that extracts risk factors and weather data from user profile.
        
        Args:
            advice: Generated advice dictionary
            user_profile: User profile containing risk_factors, weather_data, etc.
            
        Returns:
            Tuple of (is_compliant, list_of_violations)
        """
        risk_factors = user_profile.get('risk_factors', {})
        weather_data = user_profile.get('weather_data', None)
        return self.check_weather_constraints(advice, risk_factors, weather_data)
    
    def check_progression_safety(self, advice: Dict, user_profile: Dict) -> Tuple[bool, List[str]]:
        """
        Check if advice follows safe progression rules (10% rule, etc.).
        
        Wrapper method that extracts current mileage from user profile.
        
        Args:
            advice: Generated advice dictionary
            user_profile: User profile containing current_mileage, training_history, etc.
            
        Returns:
            Tuple of (is_compliant, list_of_violations)
        """
        current_mileage = user_profile.get('current_mileage', None)
        return self.check_progression_rule(advice, current_mileage)
    
    def check_experience_alignment(self, advice: Dict, user_profile: Dict) -> Tuple[bool, List[str]]:
        """
        Check if advice aligns with user's experience level.
        
        Wrapper method that extracts experience level from user profile.
        
        Args:
            advice: Generated advice dictionary
            user_profile: User profile containing experience_level, goals, etc.
            
        Returns:
            Tuple of (is_compliant, list_of_violations)
        """
        experience_level = user_profile.get('experience_level', 'unknown')
        return self.check_experience_level(advice, experience_level)


if __name__ == "__main__":
    checker = ComplianceChecker()
    
    # Example
    sample_advice = {
        'recommendations': ['Run 10 miles', 'Push through the pain']
    }
    risk_factors = {'injury_risk': True}
    
    results = checker.check_all(sample_advice, risk_factors, user_experience='beginner')
    print(f"Compliance: {results}")

