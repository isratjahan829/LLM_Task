"""
RAG System - Combines retrieval and generation
"""
from typing import Dict, List, Optional
from pdf_processor import PDFProcessor
from vector_store import VectorStore
import config


class RAGSystem:
    """Retrieval-Augmented Generation System"""
    
    def __init__(self, 
                 pdf_path: Optional[str] = None,
                 vector_db_path: Optional[str] = None,
                 load_existing: bool = False):
        """
        Initialize RAG system
        
        Args:
            pdf_path: Path to PDF directory
            vector_db_path: Path to save/load vector database
            load_existing: If True, load existing vector database
        """
        self.pdf_processor = PDFProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.vector_store = VectorStore(embedding_model_name=config.EMBEDDING_MODEL)
        
        self.pdf_path = pdf_path or config.PDF_DATA_PATH
        self.vector_db_path = vector_db_path or config.VECTOR_DB_PATH
        
        if load_existing:
            self.load_vector_db()
        else:
            self.build_vector_db()
    
    def build_vector_db(self):
        """Build vector database from PDFs"""
        print("Building vector database from PDFs...")
        
        # Load PDFs
        documents = self.pdf_processor.load_pdf_directory(self.pdf_path)
        
        if not documents:
            raise ValueError(f"No documents loaded from {self.pdf_path}")
        
        # Print statistics
        stats = self.pdf_processor.get_document_stats(documents)
        print(f"Document Statistics: {stats}")
        
        # Build vector index
        self.vector_store.build_index(documents)
        
        # Save index
        self.vector_store.save_index(self.vector_db_path)
    
    def load_vector_db(self):
        """Load existing vector database"""
        print("Loading existing vector database...")
        self.vector_store.load_index(self.vector_db_path)
    
    def retrieve_context(self, query: str, top_k: int = None) -> str:
        """Retrieve relevant context for a query"""
        top_k = top_k or config.TOP_K_RETRIEVAL
        return self.vector_store.get_relevant_context(query, k=top_k)
    
    def create_prompt(self, query: str, context: str) -> str:
        """Create prompt with context for the LLM"""
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context from PDF documents. 
If the answer cannot be found in the context, say "I cannot answer this question based on the provided documents."

Context from PDFs:
{context}

Question: {query}

Answer (based only on the context above):"""
        return prompt
    
    def get_retrieved_documents(self, query: str, top_k: int = None) -> List[tuple]:
        """Get the actual retrieved documents with scores"""
        top_k = top_k or config.TOP_K_RETRIEVAL
        return self.vector_store.similarity_search(query, k=top_k)
