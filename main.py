import sys
import os
import json
import gymnasium as gym
from langchain.output_parsers import RegexParser
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
import numpy as np
from langchain_openai import ChatOpenAI
from src.utils.read_data_utils import DocumentReader
from src.utils.LLM_utils import get_completion_gpt4
from src.utils.load_baseprompts_utils import load_prompt_from_file
from src.utils.jsonparser_utils import clean_llm_output, json_to_dataframe
from src.actor_agents.document_classifier import classify_document_with_llm
from src.actor_agents.schema_builder import schema_building_with_llm
from src.environments.schema_builder_env import SchemaBuilderEnv
from src.environments.data_extraction_env import DataExtractionEnvIterative
from src.rl_agents.gymnasium_schemabuilder_agent import GymnasiumAgent as SchemaAgent
from src.rl_agents.gymnasium_extraction_agent import GymnasiumAgent as ExtractionAgent


def process_document(file_path: str, output_dir: str = None) -> dict:
    """
    Process a document through the complete pipeline
    """
    # Initialize components
    reader = DocumentReader()
    chat_model = ChatOpenAI(model="gpt-4o-mini")
    
    print(f"Processing document: {file_path}")
    
    # 1. Read document based on file type
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.txt':
            # Handle text files
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
                pages_text = [document_text]  # Single page for text files
                num_pages = 1
        elif reader.is_image_file(file_path):
            # Handle image files
            result = reader.read_image(file_path)
            document_text = result['text']
            pages_text = [document_text]  # Single page for images
            num_pages = 1
            print(f"Image processed with confidence: {sum(result['confidence_scores'])/len(result['confidence_scores']):.2f}")
        else:
            # Handle PDFs and other document types
            result = reader.read_document(file_path)
            document_text = result['text']
            pages_text = result['pages']
            num_pages = result['num_pages']
        
        print(f"Successfully read document with {num_pages} pages")
    except Exception as e:
        raise Exception(f"Error reading document: {e}")

    # 2. Classify document using first page
    try:
        doc_type, confidence = classify_document_with_llm(pages_text[0])
        print(f"Document classified as: {doc_type} (confidence: {confidence}%)")
        
        # Handle unknown document types
        if doc_type == "Unknown":
            unknown_dir = os.path.join(output_dir, "Unknown_Docs")
            os.makedirs(unknown_dir, exist_ok=True)
            
            # Copy file to Unknown_Docs folder
            import shutil
            unknown_file_path = os.path.join(unknown_dir, os.path.basename(file_path))
            shutil.copy2(file_path, unknown_file_path)
            
            print(f"Unrecognized document type. File copied to: {unknown_file_path}")
            return {
                'document_type': "Unknown",
                'confidence': confidence,
                'schema': None,
                'extracted_data': None,
                'num_pages': num_pages
            }
            
    except Exception as e:
        raise Exception(f"Error classifying document: {e}")

    # 3. Build and optimize schema
    try:
        schema_prompt = load_prompt_from_file("schema_building_prompt.txt")
        schema_env = SchemaBuilderEnv(
            baseprompt=schema_prompt,
            document_text=pages_text[0],  # Use first page for schema building
            groundtruth=None  # No ground truth for schema building
        )
        schema_agent = SchemaAgent(chat_model, schema_env)
        schema_agent.interact()
        best_schema = schema_env.get_best_results()['best_schema']
        best_perplexity_score = schema_env.get_best_results()['best_perplexity']
        best_schema_prompt = schema_env.get_best_results()['best_prompt']
        print("Schema created and optimized")
    except Exception as e:
        raise Exception(f"Error building schema: {e}")

    # 4. Extract data from each page and combine results
    try:
        extraction_prompt = load_prompt_from_file("data_extraction_prompt.txt")
        combined_results = []
        
        for page_num, page_text in enumerate(pages_text):
            print(f"\nProcessing page {page_num + 1}/{num_pages}")
            
            # Create extraction environment for current page
            extraction_env = DataExtractionEnvIterative(
                baseprompt=extraction_prompt,
                document_type=doc_type,
                document=page_text,
                schema=best_schema,
                groundtruth=None  # No ground truth for actual extraction
            )
            
            # Run extraction optimization
            extraction_agent = ExtractionAgent(chat_model, extraction_env)
            extraction_agent.interact()

            # Get the best metrics
            best_exact_match = extraction_env.get_best_results()['best_exact_match']
            best_similarity = extraction_env.get_best_results()['best_similarity']
            
            # Get best results from this page
            page_results = extraction_env.get_best_results()['best_output']
            combined_results.append(page_results)
            
        # Merge results from all pages
        final_results = {}
        for page_result in combined_results:
            for key, value in page_result.items():
                if key not in final_results:
                    final_results[key] = []
                if isinstance(value, list):
                    final_results[key].extend(value)
                else:
                    final_results[key].append(value)
        
        print("\nData extraction completed")
        
        # Save results if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_extracted.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2)
            print(f"Results saved to: {output_file}")
        
        return {
            'document_type': doc_type,
            'confidence': confidence,
            'schema': best_schema,
            'extracted_data': final_results,
            'num_pages': num_pages
        }
        
    except Exception as e:
        raise Exception(f"Error in data extraction: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process documents for data extraction')
    parser.add_argument('input_path', help='Path to input document or directory')
    parser.add_argument('--output-dir', help='Directory to save results', default='output')
    args = parser.parse_args()
    
    if os.path.isfile(args.input_path):
        # Process single file
        result = process_document(args.input_path, args.output_dir)
        print("\nProcessing complete!")
        print(f"Document Type: {result['document_type']}")
        print(f"Number of Pages: {result['num_pages']}")
        print("Extracted Data:", json.dumps(result['extracted_data'], indent=2))
    
    elif os.path.isdir(args.input_path):
        # Process all files in directory
        for file in os.listdir(args.input_path):
            file_path = os.path.join(args.input_path, file)
            if os.path.isfile(file_path):
                try:
                    print(f"\nProcessing: {file}")
                    result = process_document(file_path, args.output_dir)
                    print(f"Successfully processed {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    else:
        print("Invalid input path")