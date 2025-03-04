
import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

#--------------------------------------------------------------------------------------------#




import json
from src.evaluation.exactmatchscore import calculate_match_score
from src.evaluation.cosinesimilarityscore import compute_cosine_similarity
from src.evaluation.intelligentmatch import semantic_match_score, semantic_match_score_v2

import re


def calculate_exact_match(generated, original):
    """
    Calculate exact match score between generated output and groundtruth
    Returns a score between 0 and 1
    """

    try:
        score = calculate_match_score(generated, original)
    except Exception as e:
        print(f"Error in exact match calculation: {str(e)}")
        return 0.0
    return score


def calculate_similarity(generated, original):
    """Calculate similarity between generated and original outputs"""
    if original is None:
        print("Warning: No groundtruth provided for similarity calculation")
        return 0.0
        
    if generated is None:
        print("Warning: No generated output provided for similarity calculation")
        return 0.0
        
    try:
        # Convert to strings if they're dictionaries
        if isinstance(generated, dict):
            generated = json.dumps(generated)
        if isinstance(original, dict):
            original = json.dumps(original)
            
        # Ensure we have strings
        generated = str(generated)
        original = str(original)
        
        cosine_similarity_score = compute_cosine_similarity(generated, original)
        return cosine_similarity_score if cosine_similarity_score is not None else 0.0
        
    except Exception as e:
        print(f"Warning: Error calculating similarity: {str(e)}")
        return 0.0
    


def calculate_semantic_match_score(generated_output, groundtruth, llm_choice):
    """Calculate exact match score between generated output and groundtruth"""

    # Convert inputs to appropriate format
    if isinstance(generated_output, dict):
        generated_output = json.dumps(generated_output, indent=2)
    else:
        generated_output = str(generated_output)
        
    if isinstance(groundtruth, dict):
        groundtruth = json.dumps(groundtruth, indent=2)
    else:
        groundtruth = str(groundtruth)

    # Use LLM-based semantic matching
    return semantic_match_score(generated_output, groundtruth, llm_choice)




