"""
Create sample PDF files for testing
Requires: pip install reportlab
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY
import os


def create_sample_pdf_1(filename):
    """Create first sample PDF about Machine Learning"""
    
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
    )
    
    story.append(Paragraph("Introduction to Machine Learning", title_style))
    story.append(Spacer(1, 12))
    
    # Content
    content = [
        "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience.",
        
        "There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning involves training a model on labeled data, where the correct output is known. The model learns to map inputs to outputs based on example input-output pairs.",
        
        "Unsupervised learning deals with unlabeled data, where the system tries to learn the patterns and structure from the data without explicit guidance. Common unsupervised learning techniques include clustering and dimensionality reduction.",
        
        "Reinforcement learning is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward. This approach is commonly used in robotics, game playing, and autonomous systems.",
        
        "Deep learning is a specialized branch of machine learning that uses artificial neural networks with multiple layers. These deep neural networks have shown remarkable success in image recognition, natural language processing, and speech recognition tasks.",
        
        "Key concepts in machine learning include training data, test data, features, labels, overfitting, underfitting, and model evaluation metrics such as accuracy, precision, recall, and F1 score.",
    ]
    
    for para in content:
        story.append(Paragraph(para, styles['BodyText']))
        story.append(Spacer(1, 12))
    
    doc.build(story)
    print(f"Created: {filename}")


def create_sample_pdf_2(filename):
    """Create second sample PDF about Natural Language Processing"""
    
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
    )
    
    story.append(Paragraph("Natural Language Processing: An Overview", title_style))
    story.append(Spacer(1, 12))
    
    # Content
    content = [
        "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a valuable way.",
        
        "NLP combines computational linguistics with machine learning and deep learning models. Common NLP tasks include text classification, named entity recognition, sentiment analysis, machine translation, and question answering.",
        
        "Word embeddings are a fundamental concept in modern NLP. They represent words as dense vectors in a continuous vector space, where semantically similar words are mapped to nearby points. Popular word embedding techniques include Word2Vec, GloVe, and FastText.",
        
        "Transformer architecture has revolutionized NLP since its introduction in 2017. Models like BERT, GPT, and T5 use self-attention mechanisms to process sequential data more effectively than traditional recurrent neural networks.",
        
        "Large Language Models (LLMs) like GPT-4 and PaLM have demonstrated impressive capabilities in understanding and generating human-like text. These models are pre-trained on massive amounts of text data and can be fine-tuned for specific tasks.",
        
        "Key challenges in NLP include handling ambiguity, understanding context, dealing with multiple languages, and ensuring ethical use of language models. Recent research focuses on improving model efficiency, reducing bias, and enhancing interpretability.",
    ]
    
    for para in content:
        story.append(Paragraph(para, styles['BodyText']))
        story.append(Spacer(1, 12))
    
    doc.build(story)
    print(f"Created: {filename}")


def create_sample_pdf_3(filename):
    """Create third sample PDF about RAG Systems"""
    
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
    )
    
    story.append(Paragraph("Retrieval-Augmented Generation Systems", title_style))
    story.append(Spacer(1, 12))
    
    # Content
    content = [
        "Retrieval-Augmented Generation (RAG) is an architecture that combines the strengths of retrieval-based and generation-based approaches in natural language processing. It enhances language models by providing them with relevant external knowledge during generation.",
        
        "The RAG system typically consists of three main components: a document retriever, a vector database, and a language model generator. The retriever searches for relevant documents based on the input query, and the generator produces responses conditioned on both the query and retrieved documents.",
        
        "Vector databases play a crucial role in RAG systems. They store document embeddings and enable efficient similarity search using techniques like FAISS or Annoy. This allows the system to quickly find the most relevant documents for a given query.",
        
        "Benefits of RAG systems include improved factual accuracy, ability to use up-to-date information, reduced hallucination, and transparency through citations. The system can ground its responses in retrieved documents rather than relying solely on parametric knowledge.",
        
        "Evaluation of RAG systems requires multiple metrics. Traditional metrics like BLEU and ROUGE measure surface-level similarity, while semantic metrics like BERTScore capture meaning. Additionally, metrics for hallucination detection and relevance are crucial for assessing RAG quality.",
        
        "Applications of RAG systems include question answering, document search, customer support chatbots, and knowledge-intensive tasks. Industries like healthcare, legal, and finance benefit from RAG's ability to provide accurate, document-grounded responses.",
    ]
    
    for para in content:
        story.append(Paragraph(para, styles['BodyText']))
        story.append(Spacer(1, 12))
    
    doc.build(story)
    print(f"Created: {filename}")


def main():
    """Create sample PDFs"""
    # Create data/pdfs directory if it doesn't exist
    os.makedirs('./data/pdfs', exist_ok=True)
    
    print("Creating sample PDF files...")
    print("="*60)
    
    create_sample_pdf_1('./data/pdfs/machine_learning_intro.pdf')
    create_sample_pdf_2('./data/pdfs/nlp_overview.pdf')
    create_sample_pdf_3('./data/pdfs/rag_systems.pdf')
    
    print("="*60)
    print("Sample PDFs created successfully!")
    print("\nYou can now run the RAG system:")
    print("  python main.py --rebuild-db")


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("Error: reportlab not installed")
        print("Install with: pip install reportlab")
