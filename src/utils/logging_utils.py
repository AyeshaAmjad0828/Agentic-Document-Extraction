import logging
import os
from datetime import datetime

def setup_logging(output_dir: str) -> str:
    """
    Setup logging configuration to write to both file and console
    
    Args:
        output_dir: Directory where log files will be stored
    
    Returns:
        str: Path to the created log file
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'processing_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will maintain console output
        ]
    )
    
    logging.info(f"Log file created at: {log_file}")
    return log_file