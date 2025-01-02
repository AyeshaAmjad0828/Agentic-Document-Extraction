import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.prompt_template import PromptTemplate


def document_extractor_agent(actor_prompt, document_type, document_text, output_json_schema):
    document_extractor_prompt = PromptTemplate(
	template="""
{{base_prompt}}
	
	
### INPUT {{DocumentType}}:
	
{{DocumentText}}
	
### OUTPUT JSON SCHEMA:
	
{{JsonSchema}}

	""",
    input_variables=["base_prompt", "DocumentType", "DocumentText","JsonSchema"]
    )
    return document_extractor_prompt.format(base_prompt=actor_prompt,DocumentType=document_type,
                                            DocumentText=document_text,
                                            JsonSchema=output_json_schema,)

def baseline_extractor_agent(prompt, document_type, document_text):
    baseline_extractor_prompt = PromptTemplate(
        template="""
{{base_prompt}}
	
### Perform the task as per the above instructions on the following {{DocumentType}} document: 
{{DocumentText}}

If you cannot find a value then denote it as "null" in the output. 
Only get me the JSON output object without any irrelevant text or explanation.
Do not leave out any information, extract all line items and key-value pairs. 

""",
	input_variables=["base_prompt","DocumentType","DocumentText"]
    )
    return baseline_extractor_prompt.format(base_prompt=prompt,
                                            DocumentType=document_type, 
                                            DocumentText=document_text,)


