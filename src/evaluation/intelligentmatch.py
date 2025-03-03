
import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

#--------------------------------------------------------------------------------------------#



from src.utils.LLM_utils import get_completion_gpt4
from src.utils.prompt_template import PromptTemplate
import json

def semantic_match_score(generated_output, groundtruth):
    """Use LLM to semantically compare generated output with groundtruth"""
    
    semantic_match_prompt = PromptTemplate(
        template="""
You are an expert evaluator for document extraction systems. 
Compare the generated output with the ground truth and:
1. Identify semantically matching fields (even if they have different key names)
2. Count total fields in generated output
3. Count how many fields and respective values from generated output match with ground truth (considering semantic similarity)
4. Compute score = (matched_fields / total_fields)
5. ALL SCORES ARE BETWEEN 0 AND 1.

Generated Output:
{{generated_output}}

Ground Truth:
{{groundtruth}}

Respond in JSON format:
{
    "total_fields": number,
    "matched_fields": number,
    "score": number,
    "field_matches": [
        {
            "generated_field": "field name",
            "groundtruth_field": "matching field name",
            "reason": "explain why the field matches"
        }
    ]
}
    """,
    input_variables=["generated_output", "groundtruth"]
    )
    prompt=semantic_match_prompt.format(
        generated_output=generated_output,
        groundtruth=groundtruth
    )

    try:
        response = get_completion_gpt4(
            [{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        result = json.loads(response.choices[0].message.content)
        
        # Print detailed matching information
        print("\nSemantic Matching Details:")
        print(f"Total fields in generated output: {result['total_fields']}")
        print(f"Matched fields: {result['matched_fields']}")
        print("\nField Matches:")
        for match in result['field_matches']:
            print(f"- {match['generated_field']} → {match['groundtruth_field']}")
            print(f"  Reason: {match['reason']}")
        
        return result['score']
        
    except Exception as e:
        print(f"Error in semantic matching: {str(e)}")
        return 0.0
    

def semantic_match_score_v2(generated_output, groundtruth):
    """Use LLM to semantically compare generated output with groundtruth"""
    
    semantic_match_prompt = PromptTemplate(
        template="""
You are an expert evaluator for information extraction systems. Compare the extracted fields from the document against the ground truth provided. Evaluate the extraction based on:

1. Field Matching: Determine if the extracted fields semantically match the ground truth fields (even if they have different names).
2. Value Accuracy: Check if the values of the extracted fields align correctly with the ground truth.
3. Completeness: Identify any missing fields in the extracted output compared to the ground truth.
4. Relevance: Ensure that no irrelevant fields or extra data are included in the extracted output.
5. Confidence Score: Compute a confidence score = (matched_fields / total_fields)
5. ALL SCORES ARE BETWEEN 0 AND 1.

For tabular structures:
- Compare row by row alignment
- Check column header matches
- Verify cell value accuracy

For key-value pairs:
- Match semantically similar keys
- Compare corresponding values
- Consider formatting variations

Generated Output:
{{generated_output}}

Ground Truth:
{{groundtruth}}

Respond in JSON format:
{
    "total_fields": number,
    "matched_fields": number,
    "confidence_score": number,
    "field_analysis": [
        {
            "ground_truth_field": "field name from ground truth",
            "extracted_field": "field name from extracted output",
            "reason": "explain why the score was assigned"
        }
    ],
    "table_analysis": {
        "row_alignment_score": number,
        "header_match_score": number,
        "cell_accuracy_score": number
    },
    "overall_score": number
}
""",
    input_variables=["generated_output", "groundtruth"]
    )
    prompt=semantic_match_prompt.format(
        generated_output=generated_output,
        groundtruth=groundtruth
    )

    try:
        response = get_completion_gpt4(
            [{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        result = json.loads(response.choices[0].message.content)
        
        # Print detailed matching information
        print("\nSemantic Matching Details:")
        print(f"Total fields: {result['total_fields']}")
        print(f"Matched fields: {result['matched_fields']}")
        print(f"Confidence score: {result['confidence_score']}")
        
        print("\nField Analysis:")
        for analysis in result['field_analysis']:
            print(f"- {analysis['ground_truth_field']} → {analysis['extracted_field']}")
            print(f"  Score: {analysis['score']}")
            print(f"  Reason: {analysis['reason']}")
            
        if 'table_analysis' in result:
            print("\nTable Analysis:")
            print(f"Row Alignment Score: {result['table_analysis']['row_alignment_score']}")
            print(f"Header Match Score: {result['table_analysis']['header_match_score']}")
            print(f"Cell Accuracy Score: {result['table_analysis']['cell_accuracy_score']}")
            
        print(f"\nOverall Score: {result['overall_score']}")
        
        return result['overall_score']
        
    except Exception as e:
        print(f"Error in semantic matching: {str(e)}")
        return 0.0
    
