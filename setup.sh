#!/bin/bash

# Setup script for RAG Evaluation System

echo "=========================================="
echo "RAG Evaluation System - Setup"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Download NLTK data
echo ""
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p data/pdfs
mkdir -p data/vector_db
mkdir -p results

# Create .env from example
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo ".env file created. Please edit it with your configuration."
fi

# Create sample PDFs (optional)
echo ""
read -p "Create sample PDF files for testing? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing reportlab..."
    pip install reportlab
    
    echo "Creating sample PDFs..."
    python create_sample_pdf.py
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Place your PDFs in: data/pdfs/"
echo "3. Edit main.py to add your test questions"
echo "4. Run evaluation: python main.py --rebuild-db"
echo ""
echo "For quick start guide, see: QUICKSTART.md"
echo "For full documentation, see: README.md"
echo ""
