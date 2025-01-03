
import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

#--------------------------------------------------------------------------------------------#




import json
from collections import Counter
from src.evaluation.exactmatchscore import calculate_match_score
from src.evaluation.cosinesimilarityscore import compute_cosine_similarity

import re


def calculate_exact_match(generated, original):
    """
    Calculate exact match score between generated output and groundtruth
    Returns a score between 0 and 1
    """

    try:
        score = calculate_match_score(generated, original)
        #print(f"Exact Match Score: {score:.2%}")
        # found_values = check_values_in_text(generated, original)
        # print("Values Found in Input Text:", found_values)

        # fuzzy_matches = fuzzy_match_in_text(generated, original)
        # print("Fuzzy Matches:", fuzzy_matches)

        # Implement exact matching logic
        # Example: Count matching fields / total fields
    except Exception as e:
        print(f"Error in exact match calculation: {str(e)}")
        return 0.0
    return score

def calculate_similarity(generated, original):
    """
    Calculate similarity score between generated output and groundtruth
    Returns a score between 0 and 1
    """
    cosine_similarity_score = compute_cosine_similarity(generated, original)
    #print("Cosine Similarity:", cosine_similarity_score)

    # Implement similarity calculation logic
    # Could use techniques like cosine similarity, Jaccard similarity, etc.
    return cosine_similarity_score

# if __name__ == "__main__":
#     # Example initial state and actions
#     invoice = f"""
# -----------------Invoice------------------
#                               Page 1 of 3

# Invoice Number: INV-12345
# Customer: XYZ Corp
# Invoice Date: 2024-06-01


# Item    Quantity    Price     Total
# item_1     5         $100      500
# item_2     10        $50       500
# item_3     6         $10       60

# 					Subtotal: 1060
# 					Total GST: 500
# 					Total Amount: $1560
# --------------------------------------------
# """
#     schema = schema = {
#     "invoice_number": "string",
#     "customer": "string",
#     "invoice_date": "yyyy-mm-dd",
#     "sub_total": "number",
#     "total_GST": "number",
#     "total_amount": "number",
#     "Line_Items": [
#         {
#             "item": "string",
#             "quantity": "number",
#             "price": "number",
#             "total": "number"
#         }
#     ] 
#     }
   
#     generated_output = {
#     "invoice_number": "INV-12345",
#     "invoice_date": "2024-06-01",
#     "sub_total": 1060,
#     "total_amount": 1560, 
#     "Line_Items": [
#         {
#         "item": "item_1",
#         "quantity": 5,
#         "price": "$100",
#         "total": 500
#         },
#         {
#         "item": "item_2",
#         "quantity": 10,
#         "price": "$50",
#         "total": 500
#         }
#     ]
#     } 
#     groundtruth = {
#     "invoice_number": "INV-12345",
#     "customer": "XYZ Corp",
#     "invoice_date": "2024-06-01",
#     "sub_total": 1060,
#     "total_GST":500,
#     "total_amount": 1560,
#     "Line_Items": [
#         {
#         "item": "item_1",
#         "quantity": 5,
#         "price": 100,
#         "total": 500
#         },
#         {
#         "item": "item_2",
#         "quantity": 10,
#         "price": 50,
#         "total": 500
#         },
#         {
#         "item": "item_3",
#         "quantity": 6,
#         "price": 10,
#         "total": 60
#         }
#     ]
#     }
#     score, found_values, fuzzy_matches = calculate_exact_match(generated_output, groundtruth)
#     print(score)
#     print(found_values)
#     print(fuzzy_matches)
#     calculate_similarity(generated_output, groundtruth)
#     print(calculate_similarity)