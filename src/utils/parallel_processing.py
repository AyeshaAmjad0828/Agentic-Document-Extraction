from typing import Dict, List, Any, Tuple
from langchain_openai import ChatOpenAI
from src.environments.data_extraction_env import DataExtractionEnvIterative
from src.rl_agents.gymnasium_extraction_agent import GymnasiumAgent as ExtractionAgent

def process_single_page(args: Tuple[str, str, str, dict, dict, int, int]) -> Dict[str, Any]:
    """
    Process a single page for data extraction in parallel.
    
    Args:
        args: Tuple containing:
            - page_text (str): The text content of the page
            - doc_type (str): Type of document
            - extraction_prompt (str): The prompt template for extraction
            - schema (dict): The schema to use for extraction
            - groundtruth (dict): Optional groundtruth for validation
            - page_num (int): Current page number
            - total_pages (int): Total number of pages
    
    Returns:
        Dict containing:
            - page_num: The processed page number
            - results: Extraction results for the page
            - steps: Number of steps taken
    """
    page_text, doc_type, extraction_prompt, schema, groundtruth, page_num, total_pages = args
    
    print(f"\nProcessing page {page_num + 1}/{total_pages}")
    
    # Initialize new chat model for this process
    chat_model = ChatOpenAI(model="gpt-4o-mini")
    
    # Create extraction environment
    extraction_env = DataExtractionEnvIterative(
        baseprompt=extraction_prompt,
        document_type=doc_type,
        document=page_text,
        schema=schema,
        groundtruth=groundtruth
    )
    
    # Run extraction
    extraction_agent = ExtractionAgent(chat_model, extraction_env)
    extraction_agent.interact()
    
    # Get results
    page_results = extraction_env.get_best_results()
    
    return {
        'page_num': page_num,
        'results': page_results,
        'steps': extraction_env.current_step
    }