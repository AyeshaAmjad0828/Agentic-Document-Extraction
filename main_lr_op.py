import sys
import os
import json
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
import time
import logging
from multiprocessing import Pool, cpu_count
from langchain_openai import ChatOpenAI

from src.utils.read_data_utils import DocumentReader
from src.utils.LLM_utils import get_completion_gpt4
from src.utils.load_baseprompts_utils import load_prompt_from_file
from src.utils.jsonparser_utils import clean_llm_output, json_to_dataframe
from src.actor_agents.document_classifier import classify_document_with_llm
from src.rl_agents.langchain_learned_prompt_optimization_openai import LearnedPromptOptimizer
from src.utils.parallel_processing import process_single_page
from src.utils.logging_utils import setup_logging
from src.utils.cache_utils import cache_results

def update_metrics_excel(metrics_dict: dict, excel_path: str = "output/metrics/learned_prompt_metrics.xlsx"):
    """Update or create Excel file with document processing metrics"""
    try:
        df_existing = pd.read_excel(excel_path)
        df_new = pd.DataFrame([metrics_dict])
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_updated = pd.DataFrame([metrics_dict])
    
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    df_updated.to_excel(excel_path, index=False)
    logging.info(f"Metrics updated in: {excel_path}")

def process_page_with_learned_prompts(args):
    """Process a single page using learned prompt optimization"""
    page_text, doc_type, base_prompt, groundtruth, page_num, total_pages, max_steps = args
    
    logging.info(f"\nProcessing page {page_num + 1}/{total_pages}")
    
    # Initialize the prompt optimizer
    optimizer = LearnedPromptOptimizer(
        model_save_dir=f"models/prompt_optimizer/{doc_type}",
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
    )
    
    try:
        # Optimize the prompt using the optimizer
        optimization_result = optimizer.optimize_prompt(
            base_prompt=base_prompt,
            doc_type=doc_type,
            current_output=None,  # Will be updated in iterations
            groundtruth=groundtruth
        )
        
        # Initialize variables for tracking best results
        best_output = None
        best_exact_match = 0
        best_semantic_match = 0
        best_similarity = 0
        steps_taken = 0
        
        # Iteratively improve the prompt
        for step in range(max_steps):
            try:
                # Get completion using the optimized prompt
                response = get_completion_gpt4(
                    optimization_result['optimized_prompt'] + "\n\nDocument Text:\n" + page_text
                )
                
                # Clean and parse the response
                cleaned_output = clean_llm_output(response)
                
                # Calculate similarity scores if groundtruth is available
                if groundtruth:
                    exact_match = calculate_exact_match(cleaned_output, groundtruth)
                    semantic_match = calculate_semantic_match(cleaned_output, groundtruth)
                    similarity = calculate_similarity(cleaned_output, groundtruth)
                    
                    # Update best scores
                    if semantic_match > best_semantic_match:
                        best_output = cleaned_output
                        best_exact_match = exact_match
                        best_semantic_match = semantic_match
                        best_similarity = similarity
                else:
                    best_output = cleaned_output
                
                # Update the optimizer with results
                extraction_success = best_semantic_match if groundtruth else 0.5
                optimizer.update_with_results(optimization_result, extraction_success)
                
                # Get new optimized prompt for next iteration
                optimization_result = optimizer.optimize_prompt(
                    base_prompt=base_prompt,
                    doc_type=doc_type,
                    current_output=best_output,
                    groundtruth=groundtruth
                )
                
                steps_taken = step + 1
                
                # Early stopping if we achieve high accuracy
                if best_semantic_match > 0.95:
                    break
                    
            except Exception as e:
                logging.warning(f"Error in optimization step {step}: {str(e)}")
                continue
        
        return {
            'page_num': page_num,
            'steps': steps_taken,
            'results': {
                'best_output': best_output,
                'best_exact_match': best_exact_match,
                'best_semantic_match': best_semantic_match,
                'best_similarity': best_similarity,
                'final_prompt': optimization_result['optimized_prompt']
            }
        }
        
    except Exception as e:
        logging.error(f"Error processing page {page_num + 1}: {str(e)}")
        return {
            'page_num': page_num,
            'steps': 0,
            'results': {
                'best_output': None,
                'best_exact_match': 0,
                'best_semantic_match': 0,
                'best_similarity': 0,
                'final_prompt': base_prompt
            }
        }

@cache_results
def process_document_with_learned_prompts(file_path: str, extraction_groundtruth: dict = None, 
                                        output_dir: str = None, max_workers: int = None, 
                                        max_steps: int = 3, force: bool = False) -> dict:
    """Process a document using learned prompt optimization"""
    
    # Create output directories
    if output_dir:
        os.makedirs(os.path.join(output_dir, "extracted_data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "unknown_docs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "optimized_prompts"), exist_ok=True)

    # Initialize metrics
    metrics = {
        'File_Path': file_path,
        'File_Name': os.path.basename(file_path),
        'Processing_Start_Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Number_of_Pages': 0,
        'Document_Type': None,
        'Classification_Confidence': 0,
        'Best_Exact_Match': None,
        'Best_Semantic_Match': None,
        'Best_Similarity': None,
        'Extraction_Steps': 0,
        'Extraction_Groundtruth_Used': bool(extraction_groundtruth),
        'Total_Processing_Time': None,
        'Reading_Time': None,
        'Classification_Time': None,
        'Extraction_Time': None,
        'Prompt_Optimization_Time': None
    }

    start_time = time.time()
    
    # Initialize components
    reader = DocumentReader()
    
    # 1. Read document
    reading_start = time.time()
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
                pages_text = [document_text]
                metrics['Number_of_Pages'] = 1
        elif reader.is_image_file(file_path):
            result = reader.read_image(file_path)
            document_text = result['text']
            pages_text = [document_text]
            metrics['Number_of_Pages'] = 1
        else:
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

    # 2. Classify document
    classification_start = time.time()
    try:
        doc_type, confidence = classify_document_with_llm(pages_text[0])
        metrics['Document_Type'] = doc_type
        metrics['Classification_Confidence'] = confidence
        
        if doc_type == "Unknown":
            unknown_dir = os.path.join(output_dir, "unknown_docs")
            shutil.copy2(file_path, os.path.join(unknown_dir, os.path.basename(file_path)))
            metrics['Classification_Time'] = round(time.time() - classification_start, 2)
            update_metrics_excel(metrics)
            return {'document_type': "Unknown", 'confidence': confidence}
            
        metrics['Classification_Time'] = round(time.time() - classification_start, 2)
        
    except Exception as e:
        metrics['Classification_Time'] = round(time.time() - classification_start, 2)
        logging.error(f"Error classifying document: {str(e)}")
        raise

    # 3. Process pages with learned prompt optimization
    extraction_start = time.time()
    try:
        base_prompt = load_prompt_from_file(document_type=doc_type)
        
        # Prepare arguments for parallel processing
        process_args = [
            (page_text, doc_type, base_prompt, extraction_groundtruth, 
             idx, len(pages_text), max_steps)
            for idx, page_text in enumerate(pages_text)
        ]
        
        # Determine number of workers
        n_workers = min(max_workers or cpu_count(), len(pages_text))
        
        # Process pages in parallel
        with Pool(processes=n_workers) as pool:
            page_results = pool.map(process_page_with_learned_prompts, process_args)
        
        # Aggregate results
        combined_results = []
        max_extraction_steps = 0
        best_exact_match = 0
        best_semantic_match = 0
        best_similarity = 0
        optimized_prompts = []
        
        for result in sorted(page_results, key=lambda x: x['page_num']):
            page_data = result['results']
            
            # Update metrics
            best_exact_match = max(best_exact_match, page_data['best_exact_match'])
            best_semantic_match = max(best_semantic_match, page_data['best_semantic_match'])
            best_similarity = max(best_similarity, page_data['best_similarity'])
            max_extraction_steps = max(max_extraction_steps, result['steps'])
            
            # Store optimized prompts
            optimized_prompts.append({
                'page': result['page_num'] + 1,
                'prompt': page_data['final_prompt']
            })
            
            # Parse and add page results
            if isinstance(page_data['best_output'], str):
                try:
                    cleaned_output = clean_llm_output(page_data['best_output'])
                    parsed_data = json.loads(cleaned_output)
                    combined_results.append(parsed_data)
                except json.JSONDecodeError:
                    logging.warning(f"Could not parse JSON from page {result['page_num'] + 1}")
                    continue
            elif isinstance(page_data['best_output'], dict):
                combined_results.append(page_data['best_output'])

        # Update metrics
        metrics['Best_Exact_Match'] = best_exact_match
        metrics['Best_Semantic_Match'] = best_semantic_match
        metrics['Best_Similarity'] = best_similarity
        metrics['Extraction_Steps'] = max_extraction_steps
        
        # Merge results from all pages
        final_results = {}
        for page_result in combined_results:
            for key, value in page_result.items():
                if key not in final_results:
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        final_results[key] = []
                    else:
                        final_results[key] = None
                
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    final_results[key].extend(value)
                else:
                    if isinstance(value, list):
                        value = value[0] if value else None
                    final_results[key] = value if final_results[key] is None else final_results[key]
        
        # Save results
        if output_dir:
            # Save extracted data
            extracted_data_path = os.path.join(output_dir, "extracted_data", 
                                             f"{os.path.splitext(os.path.basename(file_path))[0]}_extracted.json")
            with open(extracted_data_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2)
            
            # Save optimized prompts
            prompts_path = os.path.join(output_dir, "optimized_prompts", 
                                      f"{os.path.splitext(os.path.basename(file_path))[0]}_prompts.json")
            with open(prompts_path, 'w', encoding='utf-8') as f:
                json.dump(optimized_prompts, f, indent=2)
        
        metrics['Extraction_Time'] = round(time.time() - extraction_start, 2)
        metrics['Total_Processing_Time'] = round(time.time() - start_time, 2)
        metrics['Processing_End_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if metrics['Number_of_Pages'] > 0:
            metrics['Average_Time_Per_Page'] = round(metrics['Extraction_Time'] / metrics['Number_of_Pages'], 2)
        
        # Update metrics
        update_metrics_excel(metrics)
        
        return {
            'document_type': doc_type,
            'confidence': confidence,
            'extracted_data': final_results,
            'optimized_prompts': optimized_prompts,
            'num_pages': metrics['Number_of_Pages'],
            'processing_times': {
                'total_time': metrics['Total_Processing_Time'],
                'reading_time': metrics['Reading_Time'],
                'classification_time': metrics['Classification_Time'],
                'extraction_time': metrics['Extraction_Time'],
                'avg_time_per_page': metrics.get('Average_Time_Per_Page')
            }
        }
        
    except Exception as e:
        metrics['Extraction_Time'] = round(time.time() - extraction_start, 2)
        metrics['Total_Processing_Time'] = round(time.time() - start_time, 2)
        metrics['Processing_End_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_metrics_excel(metrics)
        logging.error(f"Error in data extraction: {str(e)}")
        raise

def calculate_exact_match(output: str, groundtruth: dict) -> float:
    """Calculate exact match score between output and groundtruth"""
    try:
        output_dict = json.loads(output) if isinstance(output, str) else output
        matches = sum(1 for k, v in groundtruth.items() if k in output_dict and output_dict[k] == v)
        total = len(groundtruth)
        return matches / total if total > 0 else 0
    except:
        return 0

def calculate_semantic_match(output: str, groundtruth: dict) -> float:
    """Calculate semantic similarity between output and groundtruth"""
    try:
        output_dict = json.loads(output) if isinstance(output, str) else output
        # Implement semantic matching logic here
        # For now, return a simple match score
        return calculate_exact_match(output_dict, groundtruth)
    except:
        return 0

def calculate_similarity(output: str, groundtruth: dict) -> float:
    """Calculate overall similarity between output and groundtruth"""
    exact = calculate_exact_match(output, groundtruth)
    semantic = calculate_semantic_match(output, groundtruth)
    return (exact + semantic) / 2

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process documents using learned prompt optimization')
    parser.add_argument('input_path', help='Path to input document or directory')
    parser.add_argument('--output-dir', help='Directory to save results', default='output')
    parser.add_argument('--extraction-groundtruth', help='Path to extraction groundtruth directory')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: number of CPU cores)')
    parser.add_argument('--max-steps', type=int, default=3,
                       help='Maximum optimization steps per page (default: 3)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if cache exists')
    args = parser.parse_args()

    try:
        # Setup logging
        log_file = setup_logging(args.output_dir, args.input_path)
        logging.info("Starting document processing with learned prompt optimization")
        
        if os.path.isfile(args.input_path):
            # Process single file
            try:
                # Get matching groundtruth if available
                extraction_groundtruth = None
                if args.extraction_groundtruth:
                    groundtruth_path = os.path.join(
                        args.extraction_groundtruth,
                        f"{os.path.splitext(os.path.basename(args.input_path))[0]}.json"
                    )
                    if os.path.exists(groundtruth_path):
                        with open(groundtruth_path, 'r') as f:
                            extraction_groundtruth = json.load(f)
                
                result = process_document_with_learned_prompts(
                    args.input_path,
                    extraction_groundtruth,
                    args.output_dir,
                    args.max_workers,
                    args.max_steps,
                    args.force
                )
                
                logging.info("\nProcessing complete!")
                logging.info(f"Document Type: {result['document_type']}")
                logging.info(f"Number of Pages: {result['num_pages']}")
                print("Extracted Data:", json.dumps(result['extracted_data'], indent=2))
                
            except Exception as e:
                logging.error(f"Error processing file: {str(e)}")
            finally:
                for handler in logging.root.handlers[:]:
                    handler.close()
                    logging.root.removeHandler(handler)
                    
        elif os.path.isdir(args.input_path):
            # Process directory
            for file in os.listdir(args.input_path):
                file_path = os.path.join(args.input_path, file)
                if os.path.isfile(file_path):
                    try:
                        # Setup new log file for each input file
                        log_file = setup_logging(args.output_dir, file_path)
                        logging.info(f"Processing: {file}")
                        
                        # Get matching groundtruth if available
                        extraction_groundtruth = None
                        if args.extraction_groundtruth:
                            groundtruth_path = os.path.join(
                                args.extraction_groundtruth,
                                f"{os.path.splitext(file)[0]}.json"
                            )
                            if os.path.exists(groundtruth_path):
                                with open(groundtruth_path, 'r') as f:
                                    extraction_groundtruth = json.load(f)
                        
                        result = process_document_with_learned_prompts(
                            file_path,
                            extraction_groundtruth,
                            args.output_dir,
                            args.max_workers,
                            args.max_steps,
                            args.force
                        )
                        
                        logging.info(f"Successfully processed {file}")
                        
                    except Exception as e:
                        logging.error(f"Error processing {file}: {str(e)}")
                    finally:
                        for handler in logging.root.handlers[:]:
                            handler.close()
                            logging.root.removeHandler(handler)
                            
        else:
            logging.error("Invalid input path")
            
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise 