import sys
import os
import json
import shutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from multiprocessing import Pool, cpu_count
from langchain_openai import ChatOpenAI

from src.utils.read_data_utils import DocumentReader
from src.utils.LLM_utils import get_completion_gpt4
from src.utils.load_baseprompts_utils import load_prompt_from_file
from src.utils.jsonparser_utils import clean_llm_output, json_to_dataframe
from src.actor_agents.document_classifier import classify_document_with_llm
from src.environments.schema_builder_env import SchemaBuilderEnv
from src.rl_agents.gymnasium_schemabuilder_agent import GymnasiumAgent as SchemaAgent
from src.utils.parallel_processing import process_single_page
from src.utils.logging_utils import setup_logging
from src.utils.cache_utils import cache_results



@cache_results
def update_metrics_excel(metrics_dict: dict, excel_path: str = "output\extraction_metrics.xlsx"):
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
    logging.info(f"Metrics updated in: {excel_path}")


def process_document(file_path: str, extraction_groundtruth: dict, output_dir: str = None, 
                    schema_groundtruth: dict = None, max_workers: int = None, max_steps: int = 5) -> dict:
    """
    Process a document through the complete pipeline with parallel page processing
    
    Args:
        file_path: Path to the document
        extraction_groundtruth: Groundtruth for data extraction
        output_dir: Directory to save results
        schema_groundtruth: Groundtruth for schema building
        max_workers: Maximum number of parallel workers (default: number of CPU cores)
    """

    # Start timing the entire process
    start_time = time.time()
    
    metrics = {
        'File_Path': file_path,
        'File_Name': os.path.basename(file_path),
        'Processing_Start_Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Number_of_Pages': 0,
        'Document_Type': None,
        'Classification_Confidence': 0,
        'Best_Perplexity_Score': None,
        'Schema_Steps': 0,
        'Best_Exact_Match': None,
        'Best_Semantic_Match': None,
        'Best_Similarity': None,
        'Extraction_Steps': 0,
        'Schema_Groundtruth_Used': bool(schema_groundtruth),
        'Extraction_Groundtruth_Used': bool(extraction_groundtruth),
        # Add timing metrics
        'Total_Processing_Time': None,
        'Reading_Time': None,
        'Classification_Time': None,
        'Schema_Building_Time': None,
        'Extraction_Time': None
    }    


    # Initialize components
    reader = DocumentReader()
    chat_model = ChatOpenAI(model="gpt-4o-mini")
    
    logging.info(f"Processing document: {file_path}")
    
    # 1. Read document based on file type
    reading_start = time.time()
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
            logging.info(f"Image processed with confidence: {sum(result['confidence_scores'])/len(result['confidence_scores']):.2f}")
        else:
            # Handle PDFs and other document types
            result = reader.read_document(file_path)
            document_text = result['text']
            pages_text = result['pages']
            metrics['Number_of_Pages'] = result['num_pages']
        

        metrics['Reading_Time'] = round(time.time() - reading_start, 2)
        logging.info(f"Successfully read document with {metrics['Number_of_Pages']} pages")

    except Exception as e:
        metrics['Reading_Time'] = round(time.time() - reading_start, 2)
        logging.error(f"Error reading document: {str(e)}")
        raise

    # 2. Classify document using first page
    classification_start = time.time()
    try:
        doc_type, confidence = classify_document_with_llm(pages_text[0])  # Unpack only two values
        metrics['Document_Type'] = doc_type
        metrics['Classification_Confidence'] = confidence
        logging.info(f"Document classified as: {doc_type} (confidence: {confidence}%)")
        
        if doc_type == "Unknown":
            unknown_dir = os.path.join(output_dir, "Unknown_Docs")
            os.makedirs(unknown_dir, exist_ok=True)
            
            # Copy file to Unknown_Docs folder
            unknown_file_path = os.path.join(unknown_dir, os.path.basename(file_path))
            shutil.copy2(file_path, unknown_file_path)
            
            logging.warning(f"Unrecognized document type. File copied to: {unknown_file_path}")
            
            # Update metrics file even for unknown documents
            update_metrics_excel(metrics)
            
            return {
                'document_type': "Unknown",
                'confidence': confidence,
                'schema': None,
                'extracted_data': None,
                'num_pages': metrics['Number_of_Pages']
            }
            
        metrics['Classification_Time'] = round(time.time() - classification_start, 2)
            
    except Exception as e:
        metrics['Classification_Time'] = round(time.time() - classification_start, 2)
        logging.error(f"Error classifying document: {str(e)}")
        raise

    # 3. Build and optimize schema
    schema_start = time.time()
    try:
        schema_prompt = load_prompt_from_file(filename="schema_builder_prompt.txt")
        schema_env = SchemaBuilderEnv(
            baseprompt=schema_prompt,
            document_text=pages_text[0],  # Use first page for schema building
            groundtruth=schema_groundtruth,  # Pass optional groundtruth
            max_steps=args.max_steps  # Pass max_steps to environment
        )
        schema_agent = SchemaAgent(chat_model, schema_env)
        schema_agent.interact()

        schema_results = schema_env.get_best_results()
        best_schema = schema_results['best_schema']
        metrics['Best_Perplexity_Score'] = schema_results['best_perplexity']
        metrics['Schema_Steps'] = schema_env.current_step

        if schema_groundtruth:
            metrics['Schema_Match_Score'] = schema_results.get('groundtruth_match_score', None)
        
        metrics['Schema_Building_Time'] = round(time.time() - schema_start, 2)
        logging.info("Schema created and optimized")

    except Exception as e:
        metrics['Schema_Building_Time'] = round(time.time() - schema_start, 2)
        logging.error(f"Error building schema: {str(e)}")
        raise

    # 4. Parallel Data Extraction
    extraction_start = time.time()
    try:
        extraction_prompt = load_prompt_from_file(document_type=doc_type)
        
        # Prepare arguments for parallel processing
        process_args = [
            (page_text, doc_type, extraction_prompt, best_schema, extraction_groundtruth, 
             idx, len(pages_text), max_steps) 
            for idx, page_text in enumerate(pages_text)
        ]
        
        # Determine number of workers with fallback
        try:
            available_cores = cpu_count()
        except:
            available_cores = 2  # fallback to 2 workers if cpu_count fails
            
        n_workers = min(max_workers or available_cores, len(pages_text))
        logging.info(f"\nStarting parallel processing with {n_workers} workers")
        
        # Process pages in parallel
        with Pool(processes=n_workers) as pool:
            page_results = pool.map(process_single_page, process_args)
        
        # Initialize result aggregation
        combined_results = []
        max_extraction_steps = 0
        best_exact_match = 0
        best_semantic_match = 0
        best_similarity = 0
        
        # Process results from all pages
        for result in sorted(page_results, key=lambda x: x['page_num']):
            page_data = result['results']
            
            # Update metrics
            best_exact_match = max(best_exact_match, page_data['best_exact_match'])
            best_semantic_match = max(best_semantic_match, page_data['best_semantic_match'])
            best_similarity = max(best_similarity, page_data['best_similarity'])
            max_extraction_steps = max(max_extraction_steps, result['steps'])
            
            # Parse and add page results
            if isinstance(page_data['best_output'], str):
                try:
                    parsed_data = json.loads(page_data['best_output'].strip())
                    combined_results.append(parsed_data)
                except json.JSONDecodeError:
                    logging.warning(f"Warning: Could not parse JSON from page {result['page_num'] + 1}")
                    continue
            else:
                combined_results.append(page_data['best_output'])


        # Update metrics
        metrics['Best_Exact_Match'] = best_exact_match
        metrics['Best_Semantic_Match'] = best_semantic_match
        metrics['Best_Similarity'] = best_similarity
        metrics['Extraction_Steps'] = max_extraction_steps

        # Merge results from all pages
        final_results = {}
        for page_result in combined_results:
            if not isinstance(page_result, dict):
                continue
                
            for key, value in page_result.items():
                # Initialize the key if it doesn't exist
                if key not in final_results:
                    # Check if the value is meant to be a list
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        final_results[key] = []  # Initialize as list for nested objects
                    else:
                        final_results[key] = None  # Initialize as None for simple values
                
                # Handle the value
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    # Keep arrays of objects as arrays (like 'lines')
                    final_results[key].extend(value)
                else:
                    # For simple values, take the first if it's a list
                    if isinstance(value, list):
                        value = value[0] if value else None
                    # Only update if current value is None
                    final_results[key] = value if final_results[key] is None else final_results[key]
        
        logging.info("\nData extraction completed")
        
        
        # Save results if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_extracted.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2)
            logging.info(f"Results saved to: {output_file}")
        
        metrics['Extraction_Time'] = round(time.time() - extraction_start, 2)
        
        # Calculate total processing time
        metrics['Total_Processing_Time'] = round(time.time() - start_time, 2)
        metrics['Processing_End_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add processing time per page
        if metrics['Number_of_Pages'] > 0:
            metrics['Average_Time_Per_Page'] = round(metrics['Extraction_Time'] / metrics['Number_of_Pages'], 2)
        
        # Update metrics Excel file
        update_metrics_excel(metrics)
        
        return {
            'document_type': doc_type,
            'confidence': confidence,
            'schema': best_schema,
            'extracted_data': final_results,
            'num_pages': metrics['Number_of_Pages'],
            'processing_times': {
                'total_time': metrics['Total_Processing_Time'],
                'reading_time': metrics['Reading_Time'],
                'classification_time': metrics['Classification_Time'],
                'schema_time': metrics['Schema_Building_Time'],
                'extraction_time': metrics['Extraction_Time'],
                'avg_time_per_page': metrics.get('Average_Time_Per_Page')
            }
        }
        
    except Exception as e:
        metrics['Extraction_Time'] = round(time.time() - extraction_start, 2)
        metrics['Total_Processing_Time'] = round(time.time() - start_time, 2)
        metrics['Processing_End_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_metrics_excel(metrics)  # Save metrics even if there's an error
        logging.error(f"Error in data extraction: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Process documents for data extraction')
    parser.add_argument('input_path', help='Path to input document or directory')
    parser.add_argument('--output-dir', help='Directory to save results', default='output')
    parser.add_argument('--schema-groundtruth', help='Path to schema groundtruth JSON file')
    parser.add_argument('--extraction-groundtruth', help='Path to extraction groundtruth JSON file')
    parser.add_argument('--max-steps', type=int, default=3,
                       help='Maximum number of steps before terminating (default: 3)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: number of CPU cores)')
    args = parser.parse_args()

    try:
        # Setup logging
        log_file = setup_logging(args.output_dir, args.input_path)
        logging.info("Starting document processing pipeline")
        logging.info(f"Input path: {args.input_path}")
        logging.info(f"Output directory: {args.output_dir}")
        
        # Initialize groundtruth variables
        schema_groundtruth = None
        extraction_groundtruth = None
        
        # Load groundtruth files if provided
        if args.schema_groundtruth:
            try:
                file_ext = os.path.splitext(args.schema_groundtruth)[1].lower()
                with open(args.schema_groundtruth, 'r') as f:
                    if file_ext == '.json':
                        schema_groundtruth = json.load(f)
                    else:  # .txt or other
                        schema_groundtruth = f.read().strip()
                        # Try to parse as JSON if it's in JSON format
                        try:
                            schema_groundtruth = json.loads(schema_groundtruth)
                        except json.JSONDecodeError:
                            pass  # Keep as string if not JSON
                logging.info("Loaded schema groundtruth")
            except Exception as e:
                logging.error(f"Error loading schema groundtruth: {str(e)}")
        
        if args.extraction_groundtruth:
            try:
                file_ext = os.path.splitext(args.extraction_groundtruth)[1].lower()
                with open(args.extraction_groundtruth, 'r') as f:
                    if file_ext == '.json':
                        extraction_groundtruth = json.load(f)
                    else:  # .txt or other
                        extraction_groundtruth = f.read().strip()
                        # Try to parse as JSON if it's in JSON format
                        try:
                            extraction_groundtruth = json.loads(extraction_groundtruth)
                        except json.JSONDecodeError:
                            pass  # Keep as string if not JSON
                logging.info("Loaded extraction groundtruth")
            except Exception as e:
                logging.error(f"Error loading extraction groundtruth: {e}")
        
        if os.path.isfile(args.input_path):
            # Process single file
            try: 
                logging.info(f"Processing single file: {args.input_path}")
                result = process_document(
                    args.input_path, 
                    extraction_groundtruth,  
                    args.output_dir,
                    schema_groundtruth,
                    args.max_workers,
                    args.max_steps
                )
                logging.info("\nProcessing complete!")
                logging.info(f"Document Type: {result['document_type']}")
                logging.info(f"Number of Pages: {result['num_pages']}")
                print("Extracted Data:", json.dumps(result['extracted_data'], indent=2))
            finally:
                # Clean up logging
                for handler in logging.root.handlers[:]:
                    handler.close()
                    logging.root.removeHandler(handler)

        
        elif os.path.isdir(args.input_path):
            logging.info(f"Processing directory: {args.input_path}")
            # Clean up initial logging before processing individual files
            for handler in logging.root.handlers[:]:
                handler.close()
                logging.root.removeHandler(handler)

            for file in os.listdir(args.input_path):
                file_path = os.path.join(args.input_path, file)
                if os.path.isfile(file_path):
                    try:
                        # Setup new log file for each input file
                        log_file = setup_logging(args.output_dir, file_path)
                        logging.info(f"Processing: {file}")
                        result = process_document(
                            file_path,                # input file path
                            extraction_groundtruth,   # extraction groundtruth
                            args.output_dir,         # output directory
                            schema_groundtruth,      # schema groundtruth
                            args.max_workers,         # max workers
                            args.max_steps           # max steps
                        )
                        logging.info(f"Successfully processed {file}")
                    except Exception as e:
                        logging.error(f"Error processing {file}: {str(e)}")
                    finally:
                        # Clean up logging for this file
                        for handler in logging.root.handlers[:]:
                            handler.close()
                            logging.root.removeHandler(handler)
        
            else:
                logging.error("Invalid input path")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise            