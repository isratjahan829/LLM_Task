"""
Test script to verify installation
"""
import sys

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    errors = []
    
    packages = [
        ('langchain', 'langchain'),
        ('PyPDF2', 'pypdf2'),
        ('fitz', 'pymupdf'),
        ('sentence_transformers', 'sentence-transformers'),
        ('faiss', 'faiss-cpu'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('transformers', 'transformers'),
        ('torch', 'torch'),
        ('bert_score', 'bert-score'),
        ('rouge_score', 'rouge-score'),
        ('nltk', 'nltk'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('sklearn', 'scikit-learn'),
    ]
    
    for package, install_name in packages:
        try:
            __import__(package)
            print(f"  ? {install_name}")
        except ImportError as e:
            print(f"  ? {install_name} - {str(e)}")
            errors.append(install_name)
    
    return errors


def test_nltk_data():
    """Test NLTK data availability"""
    print("\nTesting NLTK data...")
    import nltk
    
    datasets = ['punkt', 'wordnet']
    errors = []
    
    for dataset in datasets:
        try:
            nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else f'corpora/{dataset}')
            print(f"  ? {dataset}")
        except LookupError:
            print(f"  ? {dataset} - not found")
            errors.append(dataset)
    
    return errors


def test_directories():
    """Test required directories"""
    print("\nTesting directories...")
    import os
    
    dirs = [
        './data',
        './data/pdfs',
        './data/vector_db',
        './results',
    ]
    
    errors = []
    
    for directory in dirs:
        if os.path.exists(directory):
            print(f"  ? {directory}")
        else:
            print(f"  ? {directory} - not found")
            errors.append(directory)
    
    return errors


def test_modules():
    """Test custom modules"""
    print("\nTesting custom modules...")
    errors = []
    
    modules = [
        'config',
        'pdf_processor',
        'vector_store',
        'rag_system',
        'llm_models',
        'evaluation_metrics',
        'visualization',
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"  ? {module}.py")
        except Exception as e:
            print(f"  ? {module}.py - {str(e)}")
            errors.append(module)
    
    return errors


def main():
    print("="*60)
    print("RAG EVALUATION SYSTEM - INSTALLATION TEST")
    print("="*60)
    print()
    
    all_errors = []
    
    # Test imports
    errors = test_imports()
    all_errors.extend(errors)
    
    # Test NLTK data
    errors = test_nltk_data()
    all_errors.extend(errors)
    
    # Test directories
    errors = test_directories()
    all_errors.extend(errors)
    
    # Test custom modules
    errors = test_modules()
    all_errors.extend(errors)
    
    # Summary
    print("\n" + "="*60)
    if all_errors:
        print("INSTALLATION TEST FAILED")
        print("="*60)
        print("\nErrors found:")
        for error in all_errors:
            print(f"  - {error}")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        print("\nFor NLTK data:")
        print("  python -c \"import nltk; nltk.download('punkt'); nltk.download('wordnet')\"")
        sys.exit(1)
    else:
        print("INSTALLATION TEST PASSED")
        print("="*60)
        print("\nAll dependencies are installed correctly!")
        print("You can now run the system:")
        print("  python main.py --rebuild-db")
        sys.exit(0)


if __name__ == "__main__":
    main()
