"""
Configuration settings for the RAG system
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
PDF_DATA_PATH = os.getenv('PDF_DATA_PATH', './data/pdfs')
VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', './data/vector_db')
RESULTS_PATH = os.getenv('RESULTS_PATH', './results')

# Model Configuration
MODELS = {
    'mistral': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.2',
        'type': 'local',  # or 'api'
        'max_tokens': 512,
        'temperature': 0.7,
    },
    'qwen3': {
        'name': 'Qwen/Qwen2-7B-Instruct',
        'type': 'local',
        'max_tokens': 512,
        'temperature': 0.7,
    },
    'llama': {
        'name': 'meta-llama/Llama-2-7b-chat-hf',
        'type': 'local',
        'max_tokens': 512,
        'temperature': 0.7,
    }
}

# Embedding Model
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5

# Evaluation Configuration
NUM_TRIALS = 3
EVALUATION_METRICS = [
    'latency',
    'cosine_similarity',
    'f1_bertscore',
    'completeness',
    'hallucination',
    'irrelevance',
    'meteor',
    'bleu',
]

# Device Configuration
DEVICE = os.getenv('DEVICE', 'cpu')
USE_LOCAL_MODELS = os.getenv('USE_LOCAL_MODELS', 'True').lower() == 'true'

# Create directories
Path(PDF_DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
