import sys
import os
import json
import shutil
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

import pandas as pd
from datetime import datetime

def update_metrics_excel(metrics_dict: dict, excel_path: str = "extraction_metrics.xlsx"):
    """
    Update or create Excel file with document processing metrics
    """
    try:
        # Try to read existing Excel file
        df_existing = pd.read_excel(excel_path)
        df_new = pd.DataFrame([metrics_dict])
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        # Create new DataFrame if file doesn't exist
        df_updated = pd.DataFrame([metrics_dict])
    
    # Save updated DataFrame to Excel
    df_updated.to_excel(excel_path, index=False)
    print(f"Metrics updated in: {excel_path}")


def process_document(file_path: str, output_dir: str = None, schema_groundtruth: dict = None, extraction_groundtruth: dict = None) -> dict:
    """
    Process a document through the complete pipeline

    Args:
        file_path: Path to the document to process
        output_dir: Directory to save results
        schema_groundtruth: Optional groundtruth for schema building (dict)
        extraction_groundtruth: Optional groundtruth for data extraction (dict)
    """


    metrics = {
        'File_Path': file_path,
        'File_Name': os.path.basename(file_path),
        'Processing_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Number_of_Pages': 0,
        'Document_Type': None,
        'Classification_Confidence': 0,
        'Best_Perplexity_Score': None,
        'Schema_Steps': 0,
        'Best_Exact_Match': None,
        'Best_Similarity': None,
        'Extraction_Steps': 0,
        'Schema_Groundtruth_Used': bool(schema_groundtruth),
        'Extraction_Groundtruth_Used': bool(extraction_groundtruth)
    }
    
    # Initialize components
    reader = DocumentReader()
    chat_model = ChatOpenAI(model="gpt-4")
    
    print(f"Processing document: {file_path}")
    
    # 1. Read document based on file type
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.txt':
            # Handle text files
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
                pages_text = [document_text]  # Single page for text files
                metrics['Number_of_Pages'] = 1
        elif reader.is_image_file(file_path):
            # Handle image files
            result = reader.read_image(file_path)
            document_text = result['text']
            pages_text = [document_text]  # Single page for images
            metrics['Number_of_Pages'] = 1
            print(f"Image processed with confidence: {sum(result['confidence_scores'])/len(result['confidence_scores']):.2f}")
        else:
            # Handle PDFs and other document types
            result = reader.read_document(file_path)
            document_text = result['text']
            pages_text = result['pages']
            metrics['Number_of_Pages'] = result['num_pages']
        
        print(f"Successfully read document with {metrics['Number_of_Pages']} pages")

    except Exception as e:
        raise Exception(f"Error reading document: {e}")

    # 2. Classify document using first page
    try:
        doc_type, confidence = classify_document_with_llm(pages_text[0])  # Unpack only two values
        metrics['Document_Type'] = doc_type
        metrics['Classification_Confidence'] = confidence
        print(f"Document classified as: {doc_type} (confidence: {confidence}%)")
        
        if doc_type == "Unknown":
            unknown_dir = os.path.join(output_dir, "Unknown_Docs")
            os.makedirs(unknown_dir, exist_ok=True)
            
            # Copy file to Unknown_Docs folder
            unknown_file_path = os.path.join(unknown_dir, os.path.basename(file_path))
            shutil.copy2(file_path, unknown_file_path)
            
            print(f"Unrecognized document type. File copied to: {unknown_file_path}")
            
            # Update metrics file even for unknown documents
            update_metrics_excel(metrics)
            
            return {
                'document_type': "Unknown",
                'confidence': confidence,
                'schema': None,
                'extracted_data': None,
                'num_pages': metrics['Number_of_Pages']
            }
            
    except Exception as e:
        raise Exception(f"Error classifying document: {e}")

    # 3. Build and optimize schema
    try:
        schema_prompt = load_prompt_from_file(filename="schema_builder_prompt.txt")
        schema_env = SchemaBuilderEnv(
            baseprompt=schema_prompt,
            document_text=pages_text[0],  # Use first page for schema building
            groundtruth=schema_groundtruth  # Pass optional groundtruth
        )
        schema_agent = SchemaAgent(chat_model, schema_env)
        schema_agent.interact()

        schema_results = schema_env.get_best_results()
        best_schema = schema_results['best_schema']
        metrics['Best_Perplexity_Score'] = schema_results['best_perplexity']
        metrics['Schema_Steps'] = schema_env.current_step

        if schema_groundtruth:
            metrics['Schema_Match_Score'] = schema_results.get('groundtruth_match_score', None)
        
        print("Schema created and optimized")

    except Exception as e:
        raise Exception(f"Error building schema: {e}")

    # 4. Extract data from each page and combine results
    try:
        extraction_prompt = load_prompt_from_file(document_type=doc_type)
        combined_results = []
        max_extraction_steps = 0
        best_exact_match = 0
        best_similarity = 0
        
        for page_num, page_text in enumerate(pages_text):
            print(f"\nProcessing page {page_num + 1}/{metrics['Number_of_Pages']}")

            # Get page-specific groundtruth if available
            page_groundtruth = None
            if extraction_groundtruth and str(page_num + 1) in extraction_groundtruth:
                page_groundtruth = extraction_groundtruth[str(page_num + 1)]
            
            # Create extraction environment for current page
            extraction_env = DataExtractionEnvIterative(
                baseprompt=extraction_prompt,
                document_type=doc_type,
                document=page_text,
                schema=best_schema,
                groundtruth=page_groundtruth  # Pass optional page-specific groundtruth
            )
            
            # Run extraction optimization
            extraction_agent = ExtractionAgent(chat_model, extraction_env)
            extraction_agent.interact()

            # Track best scores and steps across all pages
            page_results = extraction_env.get_best_results()
            best_exact_match = max(best_exact_match, page_results['best_exact_match'])
            best_similarity = max(best_similarity, page_results['best_similarity'])
            max_extraction_steps = max(max_extraction_steps, extraction_env.current_step)
            
            combined_results.append(page_results['best_output'])

        # Update metrics with best scores
        metrics['Best_Exact_Match'] = best_exact_match
        metrics['Best_Similarity'] = best_similarity
        metrics['Extraction_Steps'] = max_extraction_steps
            
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
        
        # Update metrics Excel file
        update_metrics_excel(metrics)
        
        return {
            'document_type': doc_type,
            'confidence': confidence,
            'schema': best_schema,
            'extracted_data': final_results,
            'num_pages': metrics['Number_of_Pages']
        }
        
    except Exception as e:
        raise Exception(f"Error in data extraction: {e}")

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Process documents for data extraction')
    parser.add_argument('input_path', help='Path to input document or directory')
    parser.add_argument('--output-dir', help='Directory to save results', default='output')
    parser.add_argument('--schema-groundtruth', help='Path to schema groundtruth JSON file')
    parser.add_argument('--extraction-groundtruth', help='Path to extraction groundtruth JSON file')
    args = parser.parse_args()
    
    # Load groundtruth files if provided
    schema_groundtruth = None
    extraction_groundtruth = None
    
    if args.schema_groundtruth:
        try:
            with open(args.schema_groundtruth, 'r') as f:
                schema_groundtruth = json.load(f)
            print("Loaded schema groundtruth")
        except Exception as e:
            print(f"Error loading schema groundtruth: {e}")
    
    if args.extraction_groundtruth:
        try:
            with open(args.extraction_groundtruth, 'r') as f:
                extraction_groundtruth = json.load(f)
            print("Loaded extraction groundtruth")
        except Exception as e:
            print(f"Error loading extraction groundtruth: {e}")
    
    if os.path.isfile(args.input_path):
        # Process single file
        result = process_document(
            args.input_path, 
            args.output_dir,
            schema_groundtruth,
            extraction_groundtruth
        )
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
                    result = process_document(
                        file_path, 
                        args.output_dir,
                        schema_groundtruth,
                        extraction_groundtruth
                    )
                    print(f"Successfully processed {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    else:
        print("Invalid input path")