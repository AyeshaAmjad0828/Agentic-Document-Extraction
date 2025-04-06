import os
import sys
from typing import Dict, List
import json

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.rl_agents.langchain_learned_prompt_optimization_openai import LearnedPromptOptimizer

def validate_json_output(text: str) -> bool:
    """Validate if the text is a valid JSON string"""
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False

def validate_selection_metadata(metadata: Dict) -> bool:
    """Validate that selection metadata has all required fields"""
    if not metadata:
        return False
        
    required_fields = {
        'context': ['doc_type', 'current_output', 'groundtruth'],
        'to_select_from': ['prompt_strategy'],
        'selected': ['index', 'value', 'score']
    }
    
    try:
        for section, fields in required_fields.items():
            if section not in metadata:
                print(f"Missing section: {section}")
                return False
            for field in fields:
                if field not in metadata[section]:
                    print(f"Missing field in {section}: {field}")
                    return False
        return True
    except Exception as e:
        print(f"Error validating metadata: {e}")
        return False

def test_prompt_optimization():
    # Initialize the optimizer with default OpenAI LLM
    optimizer = LearnedPromptOptimizer(
        model_save_dir="models/prompt_optimizer"  # Directory to save the learned policy
    )
    
    # Test cases for different document types
    test_cases = [
        {
            "doc_type": "invoice",
            "base_prompt": """
            Extract the following information from the invoice:
            - Invoice number
            - Date
            - Total amount
            - Vendor name
            - Line items
            
            Return the information in JSON format.
            """,
            "current_output": """
            {
                "invoice_number": "INV-2024-001",
                "date": "2024-03-20",
                "total_amount": "$1,500.00",
                "vendor_name": "ABC Company",
                "line_items": [
                    {"item": "Service A", "amount": "$1,000.00"},
                    {"item": "Service B", "amount": "$500.00"}
                ]
            }
            """,
            "groundtruth": """
            {
                "invoice_number": "INV-2024-001",
                "date": "2024-03-20",
                "total_amount": "$1,500.00",
                "vendor_name": "ABC Company Inc.",
                "line_items": [
                    {"item": "Service A", "amount": "$1,000.00", "quantity": 1},
                    {"item": "Service B", "amount": "$500.00", "quantity": 2}
                ]
            }
            """
        },
        {
            "doc_type": "receipt",
            "base_prompt": """
            Extract the following information from the receipt:
            - Store name
            - Date
            - Total amount
            - Items purchased
            
            Return the information in JSON format.
            """,
            "current_output": """
            {
                "store_name": "Grocery Store",
                "date": "2024-03-21",
                "total_amount": "$45.50",
                "items": [
                    {"name": "Milk", "price": "$4.50"},
                    {"name": "Bread", "price": "$3.00"}
                ]
            }
            """,
            "groundtruth": """
            {
                "store_name": "Fresh Grocery Store",
                "date": "2024-03-21",
                "total_amount": "$45.50",
                "items": [
                    {"name": "Milk", "price": "$4.50", "quantity": 2},
                    {"name": "Bread", "price": "$3.00", "quantity": 1}
                ]
            }
            """
        }
    ]
    
    # Test multiple iterations to see if learning improves
    num_iterations = 3
    results_history: List[Dict] = []
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        print("=" * 80)
        
        for test_case in test_cases:
            print(f"\nTesting document type: {test_case['doc_type']}")
            print("-" * 80)
            
            try:
                # Run prompt optimization
                optimization_result = optimizer.optimize_prompt(
                    base_prompt=test_case['base_prompt'],
                    doc_type=test_case['doc_type'],
                    current_output=test_case['current_output'],
                    groundtruth=test_case['groundtruth']
                )
                
                # Validate selection metadata
                print("\nValidating Selection Metadata:")
                print("-" * 40)
                metadata = optimization_result.get('selection_metadata')
                if metadata:
                    is_valid = validate_selection_metadata(metadata)
                    print(f"Selection Metadata Valid: {is_valid}")
                    if is_valid:
                        print(f"Selected Strategy Index: {metadata['selected']['index']}")
                        print(f"Number of Variations: {len(metadata['to_select_from']['prompt_strategy'])}")
                else:
                    print("No selection metadata available")
                
                # Validate the optimized prompt
                print("\nOriginal Prompt:")
                print("-" * 40)
                print(test_case['base_prompt'])
                
                print("\nOptimized Prompt:")
                print("-" * 40)
                print(optimization_result['optimized_prompt'])
                
                print("\nPrompt Variations Generated:")
                print("-" * 40)
                for i, variation in enumerate(optimization_result['base_variations']):
                    print(f"\nStrategy {i}:")
                    print(variation[:200] + "..." if len(variation) > 200 else variation)
                
                # Validate that the optimized prompt maintains JSON output requirement
                if "JSON format" in test_case['base_prompt'] and "JSON" not in optimization_result['optimized_prompt']:
                    print("\nWarning: Optimized prompt may have lost JSON format requirement")
                
                # Simulate extraction success (in real usage, this would be based on actual extraction results)
                # Here we're simulating improvement over iterations
                extraction_success = min(0.5 + (iteration * 0.2), 0.9)  # Success rate improves with each iteration
                
                # Update the optimizer with feedback
                optimizer.update_with_results(optimization_result, extraction_success)
                
                # Store results for analysis
                results_history.append({
                    'iteration': iteration + 1,
                    'doc_type': test_case['doc_type'],
                    'success_rate': extraction_success,
                    'has_selection_metadata': metadata is not None,
                    'metadata_valid': validate_selection_metadata(metadata) if metadata else False
                })
                
                print(f"\nExtraction success rate: {extraction_success:.2f}")
                
            except Exception as e:
                print(f"Error during optimization: {e}")
                continue
    
    # Print learning progress summary
    print("\nLearning Progress Summary:")
    print("=" * 80)
    for doc_type in set(r['doc_type'] for r in results_history):
        doc_results = [r for r in results_history if r['doc_type'] == doc_type]
        print(f"\nDocument Type: {doc_type}")
        for result in doc_results:
            print(f"Iteration {result['iteration']}: "
                  f"Success Rate = {result['success_rate']:.2f}, "
                  f"Has Metadata = {result['has_selection_metadata']}, "
                  f"Valid Metadata = {result['metadata_valid']}")

if __name__ == "__main__":
    test_prompt_optimization() 