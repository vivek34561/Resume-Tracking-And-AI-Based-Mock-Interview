"""File handling utilities for PDF and TXT extraction."""

import io
import PyPDF2


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file.
    
    Args:
        pdf_file: File object or BytesIO object containing PDF data
        
    Returns:
        Extracted text as string
    """
    try:
        if hasattr(pdf_file, 'getvalue'):
            pdf_data = pdf_file.getvalue()
            pdf_file_like = io.BytesIO(pdf_data)
            reader = PyPDF2.PdfReader(pdf_file_like)
        else:
            reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        print(f"Error in extracting text from PDF: {e}")
        return ""


def extract_text_from_txt(txt_file):
    """Extract text from a TXT file.
    
    Args:
        txt_file: File object or path to text file
        
    Returns:
        Extracted text as string
    """
    try:
        if hasattr(txt_file, 'getvalue'):
            return txt_file.getvalue().decode('utf-8')
        else:
            with open(txt_file, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"Error extracting text from text file: {e}")
        return ""


def extract_text_from_file(file):
    """Extract text from a file (PDF or TXT).
    
    Args:
        file: File object with a 'name' attribute or file path
        
    Returns:
        Extracted text as string
    """
    if hasattr(file, 'name'):
        ext = file.name.split('.')[-1].lower()
    else:
        ext = str(file).split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(file)
    elif ext == 'txt':
        return extract_text_from_txt(file)
    else:
        print(f"Unsupported file extension: {ext}")
        return ""
