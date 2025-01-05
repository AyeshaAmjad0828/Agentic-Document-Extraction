# Intelligent Document Processing Pipeline

An advanced document processing system that leverages Large Language Models (LLMs) and Reinforcement Learning to extract structured data from various document formats. The system features intelligent document classification, dynamic schema generation, and adaptive data extraction capabilities.

## 🌟 Features

- **Multi-Format Support**: Process various document types including:
  - PDF (searchable and scanned)
  - Images (with OCR)
  - Microsoft Office documents (DOC/DOCX)
  - Text files
  - Common image formats (PNG, JPG, TIFF, BMP)

- **Intelligent Processing**:
  - Automatic document classification
  - Dynamic schema generation
  - Layout-preserving text extraction
  - Table structure preservation
  - Reinforcement learning for optimization

- **Performance Features**:
  - Results caching
  - Parallel processing
  - Comprehensive logging
  - Multiple evaluation metrics

## 📋 Requirements

- Python 3.10+
- OpenAI API key
- Dependencies listed in [requirements.txt](requirements.txt)
- For OCR support:
  - PaddleOCR
- For Image Reading support:
  - open-cv
  - PyMuPDF

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AyeshaAmjad0828/Unstructured-Data-Extraction.git
   cd Unstructured-Data-Extraction
   ```
2. **Set Up Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install Python Dependencies**
   ```bash
   # Upgrade pip
   python -m pip install --upgrade pip
   
   # Install requirements
   pip install -r requirements.txt
   ```
4. **Configure Environment Variables**
   ```bash
   # Edit the environment variables file and add your api keys
   OPENAI_API_KEY = "Add you Open AI api key here"
   TOGETHER_API_KEY = "Add your Together AI api key here"
   LANGCHAIN_API_KEY = "Add your langchain api key here"
   ```

## 🚀 Usage
**Run the Pipeline**

### Basic Usage

   ```bash
   python main.py --input_path "path/to/your/documents" --output_path "path/to/save/results"
   ```

### Advanced Usage
   ```bash
#Single Document Processing
python main.py "path/to/your/document.pdf" --output-dir "output" --schema-groundtruth "path/to/schema.json" --extraction-groundtruth "path/to/groundtruth.json" --max-workers 2 --max-steps 5

#Multiple Document Processing
python main.py "path/to/your/documents" --output-dir "output" --schema-groundtruth "path/to/schema.json" --extraction-groundtruth "path/to/groundtruth.json" --max-workers 4 --max-steps 5

   ```
## 🏗️ Architecture

### Core Components

1. **Document Reader** ([src/utils/read_data_utils.py](src/utils/read_data_utils.py))
   - Handles multiple document formats
   - Preserves document layout
   - Implements OCR for scanned documents and images

2. **Document Classifier** ([src/actor_agents/document_classifier.py](src/actor_agents/document_classifier.py))
   - LLM-based document type detection
   - Confidence scoring
   - Multi-class classification

3. **Schema Builder** ([src/environments/schema_builder_env.py](src/environments/schema_builder_env.py))
   - Schema building agent ([src/actor_agents/schema_builder.py](src/actor_agents/schema_builder.py))
   - Dynamic schema generation
   - Perplexity-based evaluation

4. **Data Extractor** ([src/environments/data_extraction_env.py](src/environments/data_extraction_env.py))
   - Data extraction agent ([src/actor_agents/data_extraction.py](src/actor_agents/data_extraction.py))
   - Iterative extraction process
   - Multi-metric evaluation

5. **Reinforcement learning Agent** ([src/rl_agents](src/rl_agents))
   - Gymnasium environment data extraction ([src/rl_agents/gymnasium_extraction_agent.py](src/rl_agents/gymnasium_extraction_agent.py))
   - Gymnasium environment schema generation ([src/rl_agents/gymnasium_schemabuilder_agent.py](src/rl_agents/gymnasium_schemabuilder_agent.py))
   - Action space with meta-prompting-agent ([src/action_space/meta_prompting_agent.py](src/action_space/meta_prompting_agent.py))
   - Custom reward function
   - Custom observation space

6. **Base Prompts** ([src/actor_agents/Prompts](src/actor_agents/Prompts))
   - Base prompts for document data extraction 
   - Base prompts for schema generation 


### Supporting Components

- **Parallel Processing** ([src/utils/parallel_processing.py](src/utils/parallel_processing.py))
- **Caching System** ([src/utils/cache_utils.py](src/utils/cache_utils.py))
- **Logging System** ([src/utils/logging_utils.py](src/utils/logging_utils.py))
- **Evaluation Metrics** ([src/evaluation/scoring.py](src/evaluation/scoring.py))

## 📊 Output Structure

output/
├── extracted_data/
│ ├── document1.json
│ └── document2.json
├── logs/
│ └── processing_YYYYMMDD_HHMMSS.log
├── metrics/
│ └── extraction_metrics.xlsx
└── unknown_docs/
└── unclassified_doc.pdf

## 🔍 Evaluation Metrics

The system uses multiple evaluation approaches ([src/evaluation/scoring.py](src/evaluation/scoring.py)):
- Exact Match Score
- Semantic Match Score
- Cosine Similarity
- Perplexity (for schema generation)

## 🔄 Processing Pipeline

1. **Document Ingestion**
   - Format detection
   - Text extraction
   - Layout preservation

2. **Classification**
   - Document type identification
   - Confidence scoring
   - Unknown type handling

3. **Schema Generation**
   - Dynamic schema creation
   - RL-based optimization
   - Schema validation

4. **Data Extraction**
   - Parallel processing
   - Iterative refinement
   - Multi-metric evaluation

## ⚙️ Configuration

Key configuration options:
- `max_steps`: Maximum optimization iterations
- `max_workers`: Parallel processing threads
- `cache_enabled`: Enable/disable result caching
- `log_level`: Logging detail level

## 📝 Logging

The system implements comprehensive logging ([src/utils/logging_utils.py](src/utils/logging_utils.py)):
- Processing steps
- Error tracking
- Performance metrics
- Extraction results

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

[Add your license information here]

## 🔗 References

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction.html)
- [OpenAI API Documentation](https://platform.openai.com/docs/introduction)
- [Gymnasium Documentation](https://gymnasium.farama.org/)


## Miscellaneous

### Common Issues

1. **PDF Processing Issues**
   - Error: `poppler not found`
   - Solution: Ensure poppler-utils is installed and added to PATH

2. **OCR Issues**
   - Error: `paddleocr not found`
   - Solution: Verify PaddleOCR installation and PATH configuration

   - Error: `paddlepaddle not found`
   - Solution:
   ```bash
   python3 -m pip install paddlepaddle #(CPU) 
   python3 -m pip install paddlepaddle-gpu #(CUDA 9 or CUDA 10)
   ```

3. **Dependencies Conflicts**
   ```bash
   # If you encounter dependency conflicts, try:
   pip install -r requirements.txt --no-deps
   pip install -r requirements.txt --no-cache-dir
   ```

### Optional Components

1. **GPU Support** (for faster OCR)
   ```bash
   # Install CUDA toolkit if using NVIDIA GPU
   pip install paddlepaddle-gpu
   ```

### Development Setup

1. **Install Pre-commit Hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Updating

To update to the latest version:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Docker Installation (Alternative)

If you prefer using Docker:

```bash
# Build the image
docker build -t unstructured-data-extraction .

# Run the container
docker run -v $(pwd)/data:/app/data unstructured-data-extraction
```
### System Requirements

- **Minimum Requirements**:
  - RAM: 8GB
  - CPU: 4 cores
  - Storage: 10GB free space

- **Recommended Requirements**:
  - RAM: 16GB
  - CPU: 8 cores
  - Storage: 20GB free space
  - GPU: NVIDIA GPU with CUDA support (optional)

### Need Help?

If you encounter any installation issues:
1. Check the [Common Issues](https://github.com/AyeshaAmjad0828/Unstructured-Data-Extraction?tab=readme-ov-file#common-issues)
2. Open an issue on GitHub
3. Contact the maintainers

   
