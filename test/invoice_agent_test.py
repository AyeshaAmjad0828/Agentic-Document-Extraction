import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import openai
import Invoice_prompts
from Invoice_prompts import singlepageinvoice, firstpage, middlepages, lastpage

MODEL = "gpt-4o-mini"

load_dotenv('.env')

openai_api_key = os.getenv("OPENAI_API_KEY")


def query_llm( page_text,json_schema):
    """
    Placeholder function for querying the LLM with a prompt.
    Args:
        prompt (str): The combined prompt for the LLM.
        page_text (str): The content of the document page to be processed.
        
    Returns:
        dict: Simulated response JSON (replace this with actual LLM call).
    """
    schema_text = json.dumps(json_schema, indent=2)
    prompt=f"### Perform the task as per above instructions on the following invoice document text:\n\n{page_text}\n\n ###	User-defined Output JSON schema:\n{schema_text}\nIf you cannot find a value then leave it as empty like this "" in the output. \nYou must only get me the complete JSON output without any irrelevant text or explanation."
    response = openai.Completion.create(
        model="gpt-4o-mini",  # Change to the appropriate LLM model
        prompt=prompt,
        max_tokens=1500,
        temperature=0.2
    )
    return json.loads(response.choices[0].text.strip())

def process_single_page_document(document_text, json_schema):
    """
    Processes a single-page document by combining prompt instructions and querying LLM.
    
    Args:
        document_text (str): Text content of the single-page document.
        json_schema (dict): The JSON schema dict for data extraction.
        
    Returns:
        dict: Extracted data as a JSON response.
    """
    response_json = query_llm(document_text,json_schema)
    return response_json

def process_multi_page_document(pages, system_instructions, json_schema):
    """
    Processes a multi-page document by splitting tasks for first, middle, and last pages.
    
    Args:
        pages (list of str): List of text contents for each page.
        system_instructions (str): The system prompt with custom instructions.
        json_schema (dict): The JSON schema dict for data extraction.
        
    Returns:
        dict: Extracted data as a JSON response.
    """
    combined_prompt = combine_instructions_with_schema(system_instructions, json_schema)
    extracted_data = {}
    
    # Process the first page with specific instructions
    first_page_prompt = combined_prompt + "\n\n[Process: First Page]"
    extracted_data["first_page"] = query_llm(first_page_prompt, pages[0])
    
    # Process the middle pages with general instructions (if there are multiple pages)
    if len(pages) > 2:
        middle_page_prompt = combined_prompt + "\n\n[Process: Middle Pages]"
        extracted_data["middle_pages"] = [query_llm(middle_page_prompt, page) for page in pages[1:-1]]
    
    # Process the last page with specific instructions
    last_page_prompt = combined_prompt + "\n\n[Process: Last Page]"
    extracted_data["last_page"] = query_llm(last_page_prompt, pages[-1])
    
    return extracted_data

# Example usage
system_instructions = """
Extract the relevant fields according to the classification and follow the JSON schema precisely.
Ensure the fields are complete and accurate based on the document content.
"""

# Sample JSON schema for output format
json_schema = {
    "Invoice Number": "string",
    "Date": "string",
    "Amount": "float",
    "Customer Details": {
        "Name": "string",
        "Address": "string"
    },
    "Items": [
        {
            "Description": "string",
            "Quantity": "int",
            "Price": "float"
        }
    ]
}

# Simulate document processing
# For a single-page document
single_page_text = "Invoice #123, Date: 2024-10-31, Total: $1500, Customer: John Doe, Items: Widget A - Qty: 2, Price: $750"
single_page_result = process_single_page_document(single_page_text, system_instructions, json_schema)
print("Single Page Document Extraction Result:", single_page_result)

# For a multi-page document
multi_page_texts = [
    "Invoice #456, Date: 2024-10-31, Total: $3000, Customer: Jane Smith",
    "Items: Widget A - Qty: 2, Price: $750, Widget B - Qty: 1, Price: $1500",
    "Shipping Details: Standard Delivery"
]
multi_page_result = process_multi_page_document(multi_page_texts, system_instructions, json_schema)
print("Multi-Page Document Extraction Result:", multi_page_result)

