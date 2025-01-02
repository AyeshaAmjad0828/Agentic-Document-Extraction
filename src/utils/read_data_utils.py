#pip install pdfplumber PyPDF2 paddleocr python-docx docx2txt pillow opencv-python

import os
from typing import Text, List
from pathlib import Path

# PDF processing
import pdfplumber
import PyPDF2
from paddleocr import PaddleOCR

# Document processing
from docx import Document
import docx2txt

# Image processing
from PIL import Image
import cv2
from paddleocr import PaddleOCR

def read_searchable_pdf(file_path: str) -> str:
    """
    Extract text from searchable PDFs while preserving layout.
    Uses both pdfplumber (for layout) and PyPDF2 (as fallback)
    """
    try:
        # First attempt with pdfplumber for better layout preservation
        with pdfplumber.open(file_path) as pdf:
            text = []
            for page in pdf.pages:
                text.append(page.extract_text(x_tolerance=3, y_tolerance=3))
        return '\n'.join(filter(None, text))
    except Exception:
        # Fallback to PyPDF2
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

def read_scanned_pdf(file_path: str) -> str:
    """
    Extract text from scanned PDFs using PaddleOCR.
    More accurate than Tesseract and supports multiple languages.
    """
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    pdf = cv2.imread(file_path)
    result = ocr.ocr(pdf, cls=True)
    
    # Extract and organize text while preserving structure
    text = []
    for idx, line in enumerate(result):
        if line:
            # Sort by vertical position to maintain reading order
            line = sorted(line, key=lambda x: x[0][0][1])  # Sort by y-coordinate
            line_text = ' '.join([word[1][0] for word in line])
            text.append(line_text)
    
    return '\n'.join(text)

def read_document(file_path: str) -> str:
    """
    Extract text from DOC/DOCX files while preserving formatting.
    Uses both python-docx and docx2txt for better results.
    """
    try:
        # Try docx2txt first for better formatting preservation
        text = docx2txt.process(file_path)
        if text.strip():
            return text
    except Exception:
        # Fallback to python-docx
        doc = Document(file_path)
        full_text = []
        
        # Process paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        
        # Process tables
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text for cell in row.cells]
                full_text.append(' | '.join(row_text))
        
        return '\n'.join(full_text)

def read_text_file(file_path: str) -> str:
    """Read text files with various encodings."""
    encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read file with any of the encodings: {encodings}")

def read_image(file_path: str) -> str:
    """
    Extract text from images using PaddleOCR.
    Supports multiple languages and better accuracy than Tesseract.
    """
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    image = cv2.imread(file_path)
    result = ocr.ocr(image, cls=True)
    
    # Extract and organize text while preserving structure
    text = []
    if result:
        for idx, line in enumerate(result):
            if line:
                # Sort by vertical position to maintain reading order
                line = sorted(line, key=lambda x: x[0][0][1])  # Sort by y-coordinate
                line_text = ' '.join([word[1][0] for word in line])
                text.append(line_text)
    
    return '\n'.join(text)

def process_document(file_path: str) -> str:
    """
    Process any supported document type and extract text while preserving structure.
    """
    file_extension = Path(file_path).suffix.lower()
    
    try:
        if file_extension in ['.pdf']:
            # Try searchable PDF first, fall back to OCR if needed
            text = read_searchable_pdf(file_path)
            if not text.strip():
                text = read_scanned_pdf(file_path)
            return text
        elif file_extension in ['.doc', '.docx']:
            return read_document(file_path)
        elif file_extension in ['.txt', '.csv', '.md']:
            return read_text_file(file_path)
        elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return read_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        raise Exception(f"Error processing {file_path}: {str(e)}")