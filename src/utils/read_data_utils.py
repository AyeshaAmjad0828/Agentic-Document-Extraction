### Document Reader (Reads from pdf, scanned pdf, txt, doc, docx, .png, .jpg, .jpeg, .tiff, .bmp)

#pip install pdfplumber PyPDF2 paddleocr python-docx docx2txt pillow opencv-python paddlepaddle PyMuPDF

import os
import numpy as np
from typing import Text, List, Dict, Union
from pathlib import Path
import json

# PDF processing
import pdfplumber
import PyPDF2
from paddleocr import PaddleOCR
import fitz  # PyMuPDF

# Document processing
from docx import Document
import docx2txt

# Image processing
from PIL import Image
import cv2
import io
import base64


class DocumentReader:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
    def read_document(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """
        Read document and process each page exactly once
        
        Args:
            file_path (str): Path to the document
            
        Returns:
            dict: Contains 'text' (full document text), 'pages' (list of page texts), 
                and 'num_pages' (total pages)
        """
        file_extension = Path(file_path).suffix.lower()
        
        try:
            # First, determine if it's a PDF and what type
            if file_extension == '.pdf':
                # Try to open with pdfplumber first to check if searchable
                with pdfplumber.open(file_path) as pdf:
                    # Check first page for text
                    first_page = pdf.pages[0]
                    test_text = first_page.extract_text()
                    
                    if test_text and test_text.strip():
                        print("Processing as searchable PDF...")
                        return self.read_searchable_pdf(file_path)
                    else:
                        print("Processing as scanned PDF...")
                        return self.read_scanned_pdf(file_path)
                        
            elif file_extension in ['.doc', '.docx']:
                print("Processing as Word document...")
                return self.read_document_file(file_path)
                
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            raise Exception(f"Error processing {file_path}: {str(e)}")

    def read_searchable_pdf(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """Extract text from searchable PDFs while preserving layout and pages"""
        try:
            pages_text = []
            full_text = []
            
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    # Extract tables first
                    tables = page.extract_tables()
                    tables_text = []
                    
                    for table in tables:
                        table_rows = []
                        for row in table:
                            # Clean and align row data
                            cleaned_row = [str(cell).strip() if cell else '' for cell in row]
                            table_rows.append('\t'.join(cleaned_row))
                        tables_text.append('\n'.join(table_rows))
                    
                    # Extract regular text with layout preservation
                    text = page.extract_text(
                        x_tolerance=5,
                        y_tolerance=5,
                        layout=True,
                        preserve_blank_chars=True
                    )
                    
                    # Combine tables and text while preserving structure
                    page_content = []
                    if text:
                        page_content.append(text)
                    if tables_text:
                        page_content.extend(tables_text)
                    
                    formatted_page = '\n\n'.join(page_content)
                    pages_text.append(formatted_page)
                    full_text.append(formatted_page)
                
                return {
                    'text': '\n\n'.join(filter(None, full_text)),
                    'pages': pages_text,
                    'num_pages': len(pages_text),
                    'layout_preserved': True
                }
                
        except Exception as e:
            print(f"pdfplumber failed: {e}, trying PyPDF2")
            try:
                pages_text = []
                full_text = []
                
                # Try PyMuPDF (fitz) for better layout preservation
                doc = fitz.open(file_path)
                for page in doc:
                    # Get text blocks with position information
                    blocks = page.get_text("blocks")
                    # Sort blocks by vertical position, then horizontal
                    blocks.sort(key=lambda b: (b[1], b[0]))
                    
                    page_text = []
                    current_y = None
                    current_line = []
                    
                    for b in blocks:
                        if current_y is None:
                            current_y = b[1]
                        
                        # If significant vertical change, start new line
                        if abs(b[1] - current_y) > 5:  # Adjust threshold as needed
                            if current_line:
                                page_text.append(' '.join(current_line))
                            current_line = []
                            current_y = b[1]
                        
                        current_line.append(b[4])
                    
                    # Add last line if exists
                    if current_line:
                        page_text.append(' '.join(current_line))
                    
                    formatted_page = '\n'.join(page_text)
                    pages_text.append(formatted_page)
                    full_text.append(formatted_page)
                
                return {
                    'text': '\n\n'.join(filter(None, full_text)),
                    'pages': pages_text,
                    'num_pages': len(pages_text),
                    'layout_preserved': True
                }
                
            except Exception as e2:
                raise Exception(f"Both PDF readers failed: {e2}")

    def read_scanned_pdf(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """Extract text from scanned PDFs using OCR, preserving layout"""
        try:
            pdf_document = fitz.open(file_path)
            num_pages = len(pdf_document)
            print(f"Total pages detected: {num_pages}")
            
            pages_text = []
            
            for page_num in range(num_pages):
                print(f"Processing page {page_num + 1}/{num_pages} with OCR")
                page = pdf_document[page_num]
                pix = page.get_pixmap()
                
                # Convert pixmap to image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # Perform OCR with layout analysis
                result = self.ocr.ocr(img_cv, cls=True)
                
                if result:
                    # Create a grid to maintain spatial relationships
                    page_height = img_cv.shape[0]
                    page_width = img_cv.shape[1]
                    
                    # Extract text blocks with position information
                    blocks = []
                    for line in result:
                        if line:
                            for word in line:
                                ((x1, y1), (x2, y2), (x3, y3), (x4, y4)), (text, conf) = word
                                
                                # Calculate center position and width
                                center_x = (min(x1, x2, x3, x4) + max(x1, x2, x3, x4)) / 2
                                center_y = (min(y1, y2, y3, y4) + max(y1, y2, y3, y4)) / 2
                                width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
                                
                                blocks.append({
                                    'text': text,
                                    'x': center_x,
                                    'y': center_y,
                                    'width': width,
                                    'original_coords': ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
                                })
                    
                    # Sort blocks by vertical position first
                    blocks.sort(key=lambda b: b['y'])
                    
                    # Group blocks into lines based on y-position
                    lines = []
                    current_line = []
                    current_y = None
                    y_threshold = page_height * 0.005  # Adjust threshold based on page height
                    
                    for block in blocks:
                        if current_y is None:
                            current_y = block['y']
                            current_line.append(block)
                        elif abs(block['y'] - current_y) <= y_threshold:
                            current_line.append(block)
                        else:
                            # Sort current line by x-position
                            current_line.sort(key=lambda b: b['x'])
                            
                            # Calculate spacing between blocks
                            formatted_line = []
                            last_x = 0
                            for b in current_line:
                                # Add appropriate spacing
                                spaces = int((b['x'] - last_x) / (page_width * 0.0075))  # Adjust divisor for spacing
                                formatted_line.append(" " * spaces + b['text'])
                                last_x = b['x'] + b['width']
                            
                            lines.append("".join(formatted_line))
                            current_line = [block]
                            current_y = block['y']
                    
                    # Process last line
                    if current_line:
                        current_line.sort(key=lambda b: b['x'])
                        formatted_line = []
                        last_x = 0
                        for b in current_line:
                            spaces = int((b['x'] - last_x) / (page_width * 0.02))
                            formatted_line.append(" " * spaces + b['text'])
                            last_x = b['x'] + b['width']
                        lines.append("".join(formatted_line))
                    
                    # Join lines with appropriate vertical spacing
                    page_text = "\n".join(lines)
                else:
                    page_text = ""
                
                pages_text.append(page_text)
            
            return {
                'text': '\n\n'.join(filter(None, pages_text)),
                'pages': pages_text,
                'num_pages': num_pages,
                'layout_preserved': True
            }
            
        except Exception as e:
            raise Exception(f"OCR processing failed: {e}")

    def read_document_file(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """Extract text from DOC/DOCX files while preserving sections"""
        try:
            doc = Document(file_path)
            pages_text = []
            current_page = []
            paragraph_count = 0
            paragraphs_per_page = 10  # Adjust this value based on your needs
            
            # Process paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    current_page.append(para.text)
                    paragraph_count += 1
                    
                    # Start new page after certain number of paragraphs
                    if paragraph_count >= paragraphs_per_page:
                        pages_text.append('\n'.join(current_page))
                        current_page = []
                        paragraph_count = 0
            
            # Add remaining paragraphs as last page
            if current_page:
                pages_text.append('\n'.join(current_page))
            
            return {
                'text': '\n'.join(pages_text),
                'pages': pages_text,
                'num_pages': len(pages_text)
            }
            
        except Exception as e:
            raise Exception(f"Document processing failed: {e}")
        
        
    
    def read_image(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """
        Extract text from images using PaddleOCR with layout preservation.
        Supports formats: PNG, JPG, JPEG, TIFF, BMP
        """
        supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in supported_formats:
            raise ValueError(f"Unsupported image format: {file_extension}")
        
        try:
            print(f"Processing image: {file_path}")
            
            # Read image
            if file_extension == '.tiff':
                image = Image.open(file_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError(f"Failed to read image: {file_path}")
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Perform OCR
            result = self.ocr.ocr(image, cls=True)
            
            if not result:
                return {
                    'text': '',
                    'pages': [''],
                    'num_pages': 1,
                    'layout_preserved': True
                }
            
            # Process OCR results with layout preservation
            blocks = []
            for line in result:
                if line:
                    for word in line:
                        ((x1, y1), (x2, y2), (x3, y3), (x4, y4)), (text, conf) = word
                        
                        # Calculate center position and dimensions
                        center_x = (min(x1, x2, x3, x4) + max(x1, x2, x3, x4)) / 2
                        center_y = (min(y1, y2, y3, y4) + max(y1, y2, y3, y4)) / 2
                        block_width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
                        
                        blocks.append({
                            'text': text,
                            'x': center_x,
                            'y': center_y,
                            'width': block_width,
                            'confidence': conf
                        })
            
            # Sort blocks by vertical position
            blocks.sort(key=lambda b: b['y'])
            
            # Group blocks into lines with adaptive thresholding
            lines = []
            current_line = []
            current_y = None
            y_threshold = height * 0.015  # Adaptive threshold based on image height
            
            for block in blocks:
                if current_y is None:
                    current_y = block['y']
                    current_line.append(block)
                elif abs(block['y'] - current_y) <= y_threshold:
                    current_line.append(block)
                else:
                    # Process current line
                    if current_line:
                        # Sort blocks in line by x-position
                        current_line.sort(key=lambda b: b['x'])
                        
                        # Calculate horizontal spacing
                        formatted_line = []
                        last_x = 0
                        min_space_width = width * 0.01  # Minimum space width
                        
                        for b in current_line:
                            # Calculate number of spaces needed
                            if last_x > 0:
                                space_count = max(1, int((b['x'] - last_x) / min_space_width))
                                formatted_line.append(" " * space_count)
                            
                            formatted_line.append(b['text'])
                            last_x = b['x'] + b['width']
                        
                        lines.append("".join(formatted_line))
                    
                    # Start new line
                    current_line = [block]
                    current_y = block['y']
            
            # Process last line
            if current_line:
                current_line.sort(key=lambda b: b['x'])
                formatted_line = []
                last_x = 0
                for b in current_line:
                    if last_x > 0:
                        space_count = max(1, int((b['x'] - last_x) / min_space_width))
                        formatted_line.append(" " * space_count)
                    formatted_line.append(b['text'])
                    last_x = b['x'] + b['width']
                lines.append("".join(formatted_line))
            
            # Add appropriate vertical spacing between lines
            formatted_text = ""
            last_y = 0
            for i, line in enumerate(lines):
                if i > 0:
                    # Calculate vertical spacing
                    current_y = blocks[i]['y']
                    line_spacing = int((current_y - last_y) / (height * 0.02))
                    formatted_text += "\n" * max(1, line_spacing)
                formatted_text += line
                last_y = blocks[i]['y']
            
            return {
                'text': formatted_text,
                'pages': [formatted_text],
                'num_pages': 1,
                'layout_preserved': True,
                'confidence_scores': [block['confidence'] for block in blocks]
            }
            
        except Exception as e:
            raise Exception(f"Error processing image {file_path}: {str(e)}")

    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """Check if file is a supported image format"""
        supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        return Path(file_path).suffix.lower() in supported_formats
    

    def convert_image_to_pdf(self, image_path: str, output_pdf_path: str = None) -> str:
        """
        Convert image to PDF format
        
        Args:
            image_path: Path to source image
            output_pdf_path: Optional path for output PDF. If None, uses same name as image
            
        Returns:
            str: Path to created PDF file
        """
        try:
            # If output path not specified, create one
            if output_pdf_path is None:
                output_pdf_path = str(Path(image_path).with_suffix('.pdf'))
                
            # Open image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Save as PDF
            image.save(output_pdf_path, 'PDF', resolution=300.0)
            
            return output_pdf_path
            
        except Exception as e:
            raise Exception(f"Error converting image to PDF: {str(e)}")

    def read_image_as_pdf(self, image_path: str) -> Dict[str, Union[str, List[str]]]:
        """
        Read image by first converting to PDF and then using scanned PDF processing
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Contains extracted text with layout information
        """
        try:
            # Convert image to PDF
            pdf_path = self.convert_image_to_pdf(image_path)
            
            # Process using scanned PDF function
            result = self.read_scanned_pdf(pdf_path)
            
            # Clean up temporary PDF
            if Path(pdf_path).exists():
                Path(pdf_path).unlink()
                
            return result
            
        except Exception as e:
            raise Exception(f"Error processing image as PDF: {str(e)}")