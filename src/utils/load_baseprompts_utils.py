import os

def load_prompt_from_file(filename: str = None, document_type: str = None) -> str:
    """
    Load prompt template from a text file based on document type or filename
    
    Args:
        filename: Optional specific prompt file to load
        document_type: Type of document to load prompt for
        
    Returns:
        str: Prompt template text
    """
    # Get the absolute path to the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    # If specific filename is provided, use it
    if filename:
        prompt_file = filename
    # Otherwise select prompt based on document type
    elif document_type:
        prompt_mapping = {
            "Invoice": "invoice_prompt.txt",
            "Purchase Order": "purchaseorders_prompt.txt",
            "Utility Bill": "utilitybill_prompt.txt",
            "Financial Document": "financial_prompt.txt",
            "Receipt": "invoice_prompt.txt",  # Using invoice prompt for receipts
            "Salary Slip": "salaryslip_prompt.txt",  # Using financial prompt for salary slips
            "Unknown": "financial_prompt.txt"  # Default to financial prompt for unknown types
        }
        prompt_file = prompt_mapping.get(document_type, "financial_prompt.txt")
    else:
        raise ValueError("Either filename or document_type must be provided")

    # Try different possible paths
    possible_paths = [
        os.path.join(project_root, 'Unstructured-Data-Extraction', 'src', 'actor_agents', 'Prompts', prompt_file),
        os.path.join(project_root, 'src', 'actor_agents', 'Prompts', prompt_file)
    ]
    
    for prompt_path in possible_paths:
        try:
            with open(prompt_path, 'r', encoding='utf-8') as file:
                print(f"Loading prompt from: {prompt_path}")
                return file.read()
        except FileNotFoundError:
            continue
            
    raise FileNotFoundError(f"Prompt file not found. Tried paths: {possible_paths}")