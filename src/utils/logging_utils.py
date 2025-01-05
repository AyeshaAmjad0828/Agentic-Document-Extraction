import logging
import os
import sys
from datetime import datetime


class TeeHandler(logging.StreamHandler):
    """
    A handler that writes to both a file and stdout, capturing all output
    """
    def __init__(self, filename):
        super().__init__()
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self  # Redirect stdout to this handler

    def emit(self, record):
        # Write to log file
        msg = self.format(record)
        self.file.write(msg + '\n')
        self.file.flush()
        
        # Write to console
        self.stdout.write(msg + '\n')
        self.stdout.flush()

    def write(self, msg):
        # Handle direct writes (print statements)
        if msg.strip():  # Only log non-empty messages
            self.file.write(msg)
            self.stdout.write(msg)
            self.file.flush()
            self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        sys.stdout = self.stdout  # Restore original stdout
        self.file.close()


def setup_logging(output_dir: str, input_file: str = None) -> str:
    """
    Setup logging configuration to write to both file and console
    
    Args:
        output_dir: Directory where log files will be stored
        input_file: Name of the input file being processed
    
    Returns:
        str: Path to the created log file
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log filename with input filename and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if input_file:
        # Get the filename without extension
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        log_filename = f'processing_{base_filename}_{timestamp}.log'
    else:
        log_filename = f'processing_{timestamp}.log'
    
    log_file = os.path.join(logs_dir, log_filename)
    
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        if isinstance(handler, TeeHandler):
            handler.close()
    
    # Create and configure the TeeHandler
    tee_handler = TeeHandler(log_file)
    tee_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[tee_handler],
        force=True
    )
    
    logging.info(f"Log file created at: {log_file}")
    return log_file