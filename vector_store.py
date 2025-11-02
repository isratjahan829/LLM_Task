"""
Vector Store Management using FAISS
"""
import os
import pickle
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain.schema import Document


class VectorStore:
    """Manages vector embeddings and similarity search"""
    
    def __init__(self, embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []
        self.embeddings = None
    
    def create_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Create embeddings for documents"""
        texts = [doc.page_content for doc in documents]
        print(f"Creating embeddings for {len(texts)} documents...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return np.array(embeddings).astype('float32')
    
    def build_index(self, documents: List[Document]):
        """Build FAISS index from documents"""
        self.documents = documents
        self.embeddings = self.create_embeddings(documents)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
        print(f"Built FAISS index with {len(documents)} documents")
    
    def save_index(self, save_path: str):
        """Save FAISS index and documents to disk"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_path, "index.faiss"))
        
        # Save documents and embeddings
        with open(os.path.join(save_path, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(save_path, "embeddings.pkl"), 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        print(f"Saved vector store to {save_path}")
    
    def load_index(self, load_path: str):
        """Load FAISS index and documents from disk"""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(load_path, "index.faiss"))
        
        # Load documents and embeddings
        with open(os.path.join(load_path, "documents.pkl"), 'rb') as f:
            self.documents = pickle.load(f)
        
        with open(os.path.join(load_path, "embeddings.pkl"), 'rb') as f:
            self.embeddings = pickle.load(f)
        
        print(f"Loaded vector store from {load_path}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[tuple]:
        """Search for most similar documents
        
        Returns:
            List of tuples (Document, distance)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Return documents with their distances
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(distance)))
        
        return results
    
    def get_relevant_context(self, query: str, k: int = 5) -> str:
        """Get concatenated context from top-k similar documents"""
        results = self.similarity_search(query, k)
        context_parts = []
        
        for doc, distance in results:
            context_parts.append(doc.page_content)
        
        return "\n\n".join(context_parts)
    
    def compute_cosine_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts"""
        embeddings = self.embedding_model.encode([text1, text2])
        
        # Normalize vectors
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (norm1 * norm2)
        return float(similarity)
