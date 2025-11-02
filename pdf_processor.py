"""
PDF Processing and Document Loading
"""
import os
from pathlib import Path
from typing import List, Dict
import PyPDF2
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm


class PDFProcessor:
    """Handles PDF loading and text extraction"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_pdf_pypdf2(self, pdf_path: str) -> str:
        """Load PDF using PyPDF2"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def load_pdf_pymupdf(self, pdf_path: str) -> str:
        """Load PDF using PyMuPDF (better quality)"""
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    
    def load_single_pdf(self, pdf_path: str, use_pymupdf: bool = True) -> List[Document]:
        """Load a single PDF and split into chunks"""
        try:
            if use_pymupdf:
                text = self.load_pdf_pymupdf(pdf_path)
            else:
                text = self.load_pdf_pypdf2(pdf_path)
            
            # Create documents with metadata
            filename = os.path.basename(pdf_path)
            documents = self.text_splitter.create_documents(
                [text],
                metadatas=[{"source": filename, "path": pdf_path}]
            )
            return documents
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {str(e)}")
            return []
    
    def load_pdf_directory(self, directory_path: str) -> List[Document]:
        """Load all PDFs from a directory"""
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        
        if not pdf_files:
            print(f"Warning: No PDF files found in {directory_path}")
            return []
        
        all_documents = []
        print(f"Loading {len(pdf_files)} PDF files...")
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            documents = self.load_single_pdf(str(pdf_path))
            all_documents.extend(documents)
        
        print(f"Loaded {len(all_documents)} document chunks from {len(pdf_files)} PDFs")
        return all_documents
    
    def get_document_stats(self, documents: List[Document]) -> Dict:
        """Get statistics about loaded documents"""
        total_chars = sum(len(doc.page_content) for doc in documents)
        sources = set(doc.metadata.get('source', 'unknown') for doc in documents)
        
        return {
            'total_chunks': len(documents),
            'total_characters': total_chars,
            'total_sources': len(sources),
            'avg_chunk_size': total_chars / len(documents) if documents else 0,
            'sources': list(sources)
        }
