import fitz  # PyMuPDF
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

def process_pdf_from_url(pdf_url: str) -> List[str]:
    """Downloads a PDF, extracts text, and splits it into chunks."""
    try:
        # arXiv PDF links are often the abstract page, change to the /pdf/ endpoint
        pdf_url = pdf_url.replace("/abs/", "/pdf/")
        if not pdf_url.endswith('.pdf'):
            pdf_url += '.pdf'
            
        # Download the PDF content
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()  # Will raise an exception for bad status codes
        
        pdf_document = fitz.open(stream=response.content, filetype="pdf")
        
        full_text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            full_text += page.get_text()
        
        if not full_text:
            return []

        # Use LangChain to split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_text(full_text)
        return chunks

    except Exception as e:
        print(f"Failed to process PDF from {pdf_url}. Error: {e}")
        return [] # Return empty list on failure