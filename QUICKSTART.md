# Quick Start Guide

Get started with the RAG evaluation system in 5 minutes!

## ?? Installation (2 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK data (required for evaluation metrics)
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

## ?? Prepare Data (1 minute)

### Option A: Use Sample PDFs

```bash
# Install reportlab for creating sample PDFs
pip install reportlab

# Create sample PDFs
python create_sample_pdf.py
```

### Option B: Use Your Own PDFs

```bash
# Copy your PDFs to the data directory
mkdir -p data/pdfs
cp /path/to/your/pdfs/*.pdf data/pdfs/
```

## ?? Run Evaluation (2 minutes)

### Step 1: Customize Test Questions

Edit `main.py` and update the test questions:

```python
def create_sample_test_cases() -> List[Dict]:
    return [
        {
            'question': 'What is machine learning?',
            'reference_answer': 'Machine learning is a subset of AI that enables systems to learn from data.'
        },
        # Add more questions...
    ]
```

### Step 2: Run the System

```bash
# First run - builds vector database
python main.py --rebuild-db --num-trials 3

# Subsequent runs - uses cached database
python main.py --load-existing

# Evaluate specific models only
python main.py --models mistral qwen3
```

## ?? View Results

Results are saved in the `results/` directory:

- `comprehensive_comparison.png` - All metrics comparison
- `summary_table.csv` - Tabular results
- `detailed_results.json` - Complete raw data

## ?? Common Issues

### Issue: Out of Memory

**Solution**: Use CPU instead of GPU

```bash
# Edit .env file
echo "DEVICE=cpu" > .env
python main.py
```

### Issue: Models Not Downloading

**Solution**: Set HuggingFace cache directory

```bash
export HF_HOME=/path/to/large/storage
python main.py
```

### Issue: No PDFs Found

**Solution**: Verify PDF location

```bash
ls data/pdfs/  # Should show your PDF files
```

## ?? Next Steps

1. **Read the full README.md** for detailed documentation
2. **Check example_usage.py** for code examples
3. **Customize config.py** for your needs
4. **Add more evaluation metrics** in evaluation_metrics.py

## ?? Minimal Working Example

```python
from rag_system import RAGSystem
from llm_models import ModelManager

# Initialize
rag = RAGSystem(pdf_path='./data/pdfs', load_existing=False)
model_manager = ModelManager()

# Query
query = "What is the main topic?"
context = rag.retrieve_context(query)
prompt = rag.create_prompt(query, context)

# Generate
response, latency = model_manager.generate_response('mistral', prompt)
print(f"Response: {response}")
print(f"Latency: {latency:.2f}s")
```

## ?? Getting Help

- Check README.md for detailed documentation
- Review example_usage.py for code examples
- Open an issue on GitHub for bugs
- Read error messages carefully - they usually indicate the problem

---

**Ready to evaluate! ??**
