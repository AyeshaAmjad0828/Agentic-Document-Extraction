import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.prompt_template import PromptTemplate
from src.utils.LLM_utils import get_completion_gpt4
import numpy as np


def schema_building_with_llm(baseprompt, document_text):
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
        response = get_completion_gpt4([{"role": "user", "content": prompt}],
                                       response_format={ "type": "json_object" },
                                       logprobs=True,
                                       )
        logprobs = [token.logprob for token in response.choices[0].logprobs.content]
        response_text = response.choices[0].message.content
        max_starter_length = max(len(s) for s in ["Schema \n:", "Perplexity:"])


        perplexity_score = np.exp(-np.mean(logprobs))
        #print("Response:".ljust(max_starter_length), response_text, "\n")
        #print("Perplexity:".ljust(max_starter_length), perplexity_score, "\n")

    except Exception as e:
        print(f"Error during Schema Building: {e}")
        return "Error"
    return prompt, response_text, perplexity_score










 
