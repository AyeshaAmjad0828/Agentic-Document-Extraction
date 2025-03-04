import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.prompt_template import PromptTemplate
from src.utils.LLM_utils import get_llm_completion
import numpy as np


def schema_building_with_llm(baseprompt, document_text, llm_choice):
    """
    Build schema using either GPT or Llama based on user choice
    Args:
        baseprompt: Base prompt for schema building
        document_text: Text content of the document
        llm_choice: Either "gpt" or "llama" (default: "gpt")
    """
    schema_builder_prompt = PromptTemplate(
    template="""
    
{{baseprompt}}

{{document_text}}

    """,
    input_variables=["baseprompt","document_text"]
    )
    prompt = schema_builder_prompt.format(
        baseprompt=baseprompt,
        document_text=document_text,)
    try:
        response = get_llm_completion(
            messages=[{"role": "user", "content": prompt}],
            llm_choice=llm_choice,
            response_format={ "type": "json_object" },
            logprobs=True,
            temperature=0.1,
        )
        
        # Get logprobs based on model type
        if llm_choice == "gpt":
            logprobs = [token.logprob for token in response.choices[0].logprobs.content]
        else:  # llama
            logprobs = response.choices[0].logprobs.token_logprobs
            
        response_text = response.choices[0].message.content
        perplexity_score = np.exp(-np.mean(logprobs))
        
        max_starter_length = max(len(s) for s in ["Schema \n:", "Perplexity:"])
        #print("Response:".ljust(max_starter_length), response_text, "\n")
        #print("Perplexity:".ljust(max_starter_length), perplexity_score, "\n")

    except Exception as e:
        print(f"Error during Schema Building: {e}")
        return "Error", None, None
        
    return prompt, response_text, perplexity_score 

# if __name__ == "__main__":
#     # Example initial state and actions
#     baseprompt = '''
# You are a JSON schema builder that contains metadata of key-value pairs and tables. You will receive an unstructured form-like document and you will perform following tasks:

# 1. Create a JSON schema by extracting only metadata of key-value pairs and table headers from the document. 

# 2. Always arrange all keys phrases with data types in the parent region of the json

# 3. Always arrange all table headers with data types in the child region.

# 4. Ensure the order of key names and table headers aligns with the original text.

# 5. ONLY include unique column names as they appear in the document, without duplicates or associated data. 

# 6. If the document contains multiple tables, ensure that each table is represented in the JSON schema with its own parent region.

# 7. If a value does not hvae any defined key or header, then generate an appropriate key or header for the value.

# 8. ONLY Respond with the JSON schema with all key phrases and table column headers.

# Here is the content of the form-like document:

# '''
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
#     a, b, c = schema_building_with_llm(baseprompt, invoice, llm_choice="llama")
#     print(a)
#     print(b)
#     print(c)







 
