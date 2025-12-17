"""
Response evaluator for Multi-LLM System.
Provides readability scoring and response quality metrics.
"""
from typing import Dict, Any
from .providers.base import LLMResponse


class ResponseEvaluator:
    """Evaluator for LLM responses."""
    
    @staticmethod
    def calculate_readability(text: str) -> Dict[str, Any]:
        """
        Calculate readability metrics for text.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Dictionary with readability metrics
        """
        try:
            import textstat
            
            # Calculate various readability scores
            flesch_reading_ease = textstat.flesch_reading_ease(text)
            flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
            
            # Interpret Flesch Reading Ease score
            if flesch_reading_ease >= 90:
                interpretation = "Very Easy (5th grade)"
            elif flesch_reading_ease >= 80:
                interpretation = "Easy (6th grade)"
            elif flesch_reading_ease >= 70:
                interpretation = "Fairly Easy (7th grade)"
            elif flesch_reading_ease >= 60:
                interpretation = "Standard (8th-9th grade)"
            elif flesch_reading_ease >= 50:
                interpretation = "Fairly Difficult (10th-12th grade)"
            elif flesch_reading_ease >= 30:
                interpretation = "Difficult (College)"
            else:
                interpretation = "Very Difficult (College graduate)"
            
            return {
                "flesch_reading_ease": round(flesch_reading_ease, 2),
                "flesch_kincaid_grade": round(flesch_kincaid_grade, 2),
                "interpretation": interpretation,
                "word_count": len(text.split()),
                "sentence_count": textstat.sentence_count(text)
            }
        except ImportError:
            # Fallback if textstat is not available
            words = text.split()
            sentences = text.count('.') + text.count('!') + text.count('?')
            sentences = max(1, sentences)  # Avoid division by zero
            
            return {
                "word_count": len(words),
                "sentence_count": sentences,
                "avg_word_length": sum(len(w) for w in words) / max(1, len(words)),
                "interpretation": "Basic metrics only (install textstat for full analysis)"
            }
    
    @staticmethod
    def evaluate_response(response: LLMResponse) -> Dict[str, Any]:
        """
        Evaluate an LLM response.
        
        Args:
            response: LLMResponse object
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not response.success:
            return {
                "success": False,
                "error": response.error,
                "model": response.model_name
            }
        
        readability = ResponseEvaluator.calculate_readability(response.content)
        
        # Speed rating
        if response.latency < 1.0:
            speed_rating = "Very Fast"
        elif response.latency < 3.0:
            speed_rating = "Fast"
        elif response.latency < 5.0:
            speed_rating = "Moderate"
        elif response.latency < 10.0:
            speed_rating = "Slow"
        else:
            speed_rating = "Very Slow"
        
        return {
            "success": True,
            "model": response.model_name,
            "latency": round(response.latency, 2),
            "speed_rating": speed_rating,
            "tokens_used": response.tokens_used,
            "cost": response.estimated_cost,
            "readability": readability,
            "response_length": len(response.content)
        }
    
    @staticmethod
    def compare_responses(responses: Dict[str, LLMResponse]) -> Dict[str, Any]:
        """
        Compare multiple responses.
        
        Args:
            responses: Dictionary mapping provider names to responses
            
        Returns:
            Comparison metrics
        """
        evaluations = {}
        for name, response in responses.items():
            evaluations[name] = ResponseEvaluator.evaluate_response(response)
        
        # Find fastest successful response
        successful = {
            name: eval_data 
            for name, eval_data in evaluations.items() 
            if eval_data.get("success", False)
        }
        
        if successful:
            fastest = min(
                successful.items(),
                key=lambda x: x[1].get("latency", float('inf'))
            )
            
            # Find most readable (highest Flesch Reading Ease if available)
            most_readable = None
            highest_readability = float('-inf')
            
            for name, eval_data in successful.items():
                readability_score = eval_data.get("readability", {}).get("flesch_reading_ease", float('-inf'))
                if readability_score > highest_readability and readability_score != float('-inf'):
                    highest_readability = readability_score
                    most_readable = name
            
            return {
                "evaluations": evaluations,
                "fastest_model": fastest[0],
                "fastest_latency": fastest[1]["latency"],
                "most_readable_model": most_readable,
                "total_models": len(responses),
                "successful_models": len(successful)
            }
        else:
            return {
                "evaluations": evaluations,
                "total_models": len(responses),
                "successful_models": 0,
                "error": "All models failed"
            }
