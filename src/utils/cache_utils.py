import functools
import hashlib
import json
import os
import logging

def cache_results(func):
    """Cache function results to avoid reprocessing the same documents"""
    cache_dir = os.path.join('cache', 'document_processing')
    os.makedirs(cache_dir, exist_ok=True)
    
    @functools.wraps(func)
    def wrapper(*args, force=False, **kwargs):
        # Extract file_path from args (it's the first argument in process_document)
        file_path = args[0] if args else None
        
        # Skip caching if file_path is not a string/path
        if not isinstance(file_path, (str, bytes, os.PathLike)):
            return func(*args, **kwargs)
        
        # Create cache key based on file content hash
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        cache_file = os.path.join(cache_dir, f'{file_hash}.json')
        
        # Check cache if not forcing reprocess
        if not force and os.path.exists(cache_file):
            logging.info(f"Loading cached results for {os.path.basename(file_path)}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Process and cache result
        logging.info(f"Processing new document: {os.path.basename(file_path)}")
        result = func(*args, **kwargs)  # Pass all arguments including force
        with open(cache_file, 'w') as f:
            json.dump(result, f)
        
        return result
    
    return wrapper