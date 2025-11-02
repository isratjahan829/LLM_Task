# RAG-Based PDF Question Answering System with Comprehensive Evaluation

A complete **Retrieval-Augmented Generation (RAG)** system that answers questions based on PDF documents using multiple LLM models (Mistral AI, Qwen3, Llama) with comprehensive evaluation metrics and performance comparison visualizations.

## ?? Features

- **Multi-Model Support**: Compare performance across Mistral, Qwen3, and Llama models
- **PDF Processing**: Automatically extract and chunk text from PDF documents
- **Vector Search**: Fast similarity search using FAISS vector database
- **Comprehensive Evaluation**: 10+ metrics including:
  - Latency measurement
  - Cosine Similarity
  - BERTScore (F1, Precision, Recall)
  - Completeness
  - Hallucination Detection
  - Irrelevance Detection
  - METEOR
  - BLEU
  - ROUGE scores
- **Multi-Trial Testing**: Run multiple trials to measure consistency
- **Beautiful Visualizations**: Automatic generation of comparison charts and tables

## ?? Requirements

```bash
Python 3.8+
CUDA (optional, for GPU acceleration)
```

## ?? Installation

1. **Clone the repository**
```bash
git clone <your-repo>
cd <repo-directory>
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (optional)
```bash
cp .env.example .env
# Edit .env with your configuration
```

## ?? Project Structure

```
.
??? config.py                 # Configuration settings
??? pdf_processor.py         # PDF loading and chunking
??? vector_store.py          # FAISS vector database management
??? rag_system.py            # RAG system implementation
??? llm_models.py            # Multi-model LLM integration
??? evaluation_metrics.py    # Comprehensive evaluation metrics
??? visualization.py         # Performance visualization
??? main.py                  # Main execution script
??? requirements.txt         # Python dependencies
??? data/
    ??? pdfs/               # Place your PDF files here
    ??? vector_db/          # Vector database storage
    ??? results/            # Evaluation results and charts
```

## ?? Usage

### Step 1: Prepare Your PDFs

Place your PDF documents in the `data/pdfs/` directory:

```bash
mkdir -p data/pdfs
cp your-documents/*.pdf data/pdfs/
```

### Step 2: Build Vector Database

First run will automatically build the vector database from your PDFs:

```bash
python main.py --rebuild-db
```

### Step 3: Create Test Questions

Edit `main.py` and modify the `create_sample_test_cases()` function with your test questions:

```python
def create_sample_test_cases() -> List[Dict]:
    return [
        {
            'question': 'What is machine learning?',
            'reference_answer': 'Machine learning is a subset of AI that enables systems to learn from data.'
        },
        {
            'question': 'What are neural networks?',
            'reference_answer': 'Neural networks are computing systems inspired by biological neural networks.'
        }
    ]
```

### Step 4: Run Evaluation

Run the complete evaluation pipeline:

```bash
# Evaluate all models
python main.py

# Evaluate specific models
python main.py --models mistral llama

# Custom number of trials
python main.py --num-trials 5

# Load existing vector database
python main.py --load-existing
```

## ?? Command Line Arguments

- `--pdf-path PATH`: Path to PDF directory (default: ./data/pdfs)
- `--load-existing`: Load existing vector database (faster)
- `--rebuild-db`: Rebuild vector database from scratch
- `--num-trials N`: Number of trials for evaluation (default: 3)
- `--models MODEL1 MODEL2`: Specific models to evaluate (default: all)

## ?? Evaluation Metrics Explained

### Performance Metrics

1. **Latency**: Time taken to generate response (in seconds)
2. **Cosine Similarity**: Semantic similarity between reference and response (0-1, higher is better)
3. **BERTScore F1**: Contextual similarity using BERT embeddings (0-1, higher is better)

### Quality Metrics

4. **Completeness**: How much of the reference answer is covered (0-1, higher is better)
5. **Hallucination**: Extent of fabricated information (0-1, lower is better)
6. **Irrelevance**: How off-topic the response is (0-1, lower is better)

### NLP Metrics

7. **METEOR**: Machine Translation evaluation metric (0-1, higher is better)
8. **BLEU**: Bilingual Evaluation Understudy score (0-1, higher is better)
9. **ROUGE-1/2/L**: Recall-Oriented Understudy for Gisting Evaluation (0-1, higher is better)

## ?? Output and Results

After running the evaluation, you'll find in `results/`:

1. **comprehensive_comparison.png**: All metrics compared across models
2. **comparison_*.png**: Individual metric comparisons
3. **trial_consistency.png**: Consistency across multiple trials
4. **summary_table.csv**: Tabular summary of all metrics
5. **detailed_results.json**: Complete raw results in JSON format

### Example Output

```
Model Comparison Summary:
??????????????????????????????????????????????????????????????
? Model    ? Avg Latency(s) ? Cosine Similarity? BERTScore F1?
??????????????????????????????????????????????????????????????
? mistral  ? 2.35 ? 0.12    ? 0.823            ? 0.891       ?
? qwen3    ? 2.18 ? 0.09    ? 0.847            ? 0.903       ?
? llama    ? 2.52 ? 0.15    ? 0.812            ? 0.885       ?
??????????????????????????????????????????????????????????????
```

## ?? Configuration

Edit `config.py` to customize:

```python
# Model configuration
MODELS = {
    'mistral': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.2',
        'max_tokens': 512,
        'temperature': 0.7,
    },
    # Add more models...
}

# RAG configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5

# Evaluation configuration
NUM_TRIALS = 3
```

## ?? Advanced Usage

### Using API-Based Models

If you want to use API-based models instead of local models:

1. Add API keys to `.env`:
```bash
MISTRAL_API_KEY=your_key_here
```

2. Modify model configuration in `config.py`:
```python
'mistral': {
    'type': 'api',
    'api_key': os.getenv('MISTRAL_API_KEY'),
    ...
}
```

### Custom Evaluation Metrics

Add your own metrics in `evaluation_metrics.py`:

```python
def compute_custom_metric(self, response: str, reference: str) -> float:
    # Your custom metric implementation
    return score
```

### Custom Visualization

Extend `visualization.py` to create custom charts:

```python
def plot_custom_chart(self, data):
    # Your visualization code
    pass
```

## ?? Troubleshooting

### Out of Memory Errors

If you encounter CUDA out of memory:

1. Use quantization (already enabled by default)
2. Reduce batch size
3. Use CPU instead: Set `DEVICE=cpu` in `.env`
4. Evaluate models one at a time

### Model Download Issues

Models are downloaded from HuggingFace. Ensure you have:

1. Internet connection
2. Sufficient disk space (~15GB per model)
3. HuggingFace account (for gated models like Llama)

### PDF Loading Issues

If PDFs fail to load:

1. Ensure PDFs are not encrypted
2. Try switching between PyPDF2 and PyMuPDF
3. Check PDF file permissions

## ?? Citation

If you use this system in your research, please cite:

```bibtex
@software{rag_evaluation_system,
  title={RAG-Based PDF Question Answering with Comprehensive Evaluation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/rag-evaluation}
}
```

## ?? Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## ?? License

This project is licensed under the MIT License - see LICENSE file for details.

## ?? Acknowledgments

- HuggingFace for model hosting
- LangChain for RAG utilities
- FAISS for efficient similarity search
- BERTScore, METEOR, BLEU for evaluation metrics

## ?? Contact

For questions or issues, please open a GitHub issue or contact [your-email].

---

**Note**: This system requires significant computational resources for running multiple LLM models. GPU with at least 16GB VRAM is recommended for optimal performance.

## ?? Quick Start Example

```python
from rag_system import RAGSystem
from llm_models import ModelManager

# Initialize system
rag = RAGSystem(pdf_path='./data/pdfs', load_existing=False)

# Load a model
model_manager = ModelManager()
model = model_manager.get_model('mistral')

# Ask a question
query = "What is the main topic of the documents?"
context = rag.retrieve_context(query)
prompt = rag.create_prompt(query, context)
response, latency = model.generate(prompt)

print(f"Response: {response}")
print(f"Latency: {latency:.2f}s")
```

## ?? Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [BERTScore Paper](https://arxiv.org/abs/1904.09675)

---

**Happy Evaluating! ??**
