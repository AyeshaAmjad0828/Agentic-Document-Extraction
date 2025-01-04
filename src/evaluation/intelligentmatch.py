
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
4. Provide a score between 0 and 1

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
    
