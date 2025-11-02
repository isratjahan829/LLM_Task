"""
Example usage scripts for the RAG system
"""
from rag_system import RAGSystem
from llm_models import ModelManager
from evaluation_metrics import EvaluationMetrics
import config


def example_basic_usage():
    """Example 1: Basic RAG query"""
    print("="*60)
    print("EXAMPLE 1: Basic RAG Query")
    print("="*60)
    
    # Initialize RAG system
    rag = RAGSystem(load_existing=True)
    
    # Ask a question
    query = "What is the main topic discussed in the documents?"
    context = rag.retrieve_context(query)
    
    print(f"\nQuery: {query}")
    print(f"\nRetrieved Context ({len(context)} chars):")
    print(context[:500] + "...")
    
    # Create prompt
    prompt = rag.create_prompt(query, context)
    print(f"\nPrompt created (length: {len(prompt)} chars)")


def example_single_model_inference():
    """Example 2: Generate response with single model"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Single Model Inference")
    print("="*60)
    
    # Initialize components
    rag = RAGSystem(load_existing=True)
    model_manager = ModelManager()
    
    # Prepare query
    query = "What are the key points mentioned?"
    context = rag.retrieve_context(query)
    prompt = rag.create_prompt(query, context)
    
    # Generate response
    print(f"\nQuery: {query}")
    print("\nGenerating response with Mistral model...")
    
    response, latency = model_manager.generate_response('mistral', prompt)
    
    print(f"\nResponse: {response}")
    print(f"Latency: {latency:.2f} seconds")


def example_compare_models():
    """Example 3: Compare multiple models on same query"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Compare Multiple Models")
    print("="*60)
    
    # Initialize
    rag = RAGSystem(load_existing=True)
    model_manager = ModelManager()
    
    query = "What is discussed in the documents?"
    context = rag.retrieve_context(query)
    prompt = rag.create_prompt(query, context)
    
    print(f"\nQuery: {query}\n")
    
    # Test each model
    models = ['mistral', 'qwen3', 'llama']
    
    for model_name in models:
        print(f"\n{'-'*60}")
        print(f"Model: {model_name}")
        print(f"{'-'*60}")
        
        response, latency = model_manager.generate_response(model_name, prompt)
        
        print(f"Latency: {latency:.2f}s")
        print(f"Response: {response[:200]}...")


def example_evaluate_response():
    """Example 4: Evaluate response quality"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Evaluate Response Quality")
    print("="*60)
    
    # Initialize
    rag = RAGSystem(load_existing=True)
    model_manager = ModelManager()
    evaluator = EvaluationMetrics()
    
    # Generate response
    query = "What is the main topic?"
    reference = "The main topic is about artificial intelligence and machine learning."
    
    context = rag.retrieve_context(query)
    prompt = rag.create_prompt(query, context)
    
    response, latency = model_manager.generate_response('mistral', prompt)
    
    # Evaluate
    print(f"\nQuery: {query}")
    print(f"Response: {response}")
    print(f"\nReference: {reference}")
    
    metrics = evaluator.compute_all_metrics(
        query=query,
        response=response,
        reference=reference,
        context=context,
        latency=latency,
        embeddings_model=rag.vector_store.embedding_model
    )
    
    print("\nEvaluation Metrics:")
    print(f"  Latency: {metrics['latency']:.2f}s")
    print(f"  Cosine Similarity: {metrics['cosine_similarity']:.3f}")
    print(f"  BERTScore F1: {metrics['bertscore_f1']:.3f}")
    print(f"  Completeness: {metrics['completeness']:.3f}")
    print(f"  Hallucination: {metrics['hallucination']:.3f}")
    print(f"  Irrelevance: {metrics['irrelevance']:.3f}")
    print(f"  METEOR: {metrics['meteor']:.3f}")
    print(f"  BLEU: {metrics['bleu']:.3f}")


def example_custom_retrieval():
    """Example 5: Custom retrieval with different k values"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Retrieval Parameters")
    print("="*60)
    
    rag = RAGSystem(load_existing=True)
    
    query = "What is mentioned about machine learning?"
    
    print(f"\nQuery: {query}\n")
    
    # Try different k values
    for k in [3, 5, 10]:
        print(f"\n{'-'*60}")
        print(f"Top-K = {k}")
        print(f"{'-'*60}")
        
        results = rag.get_retrieved_documents(query, top_k=k)
        
        print(f"Retrieved {len(results)} documents")
        for idx, (doc, distance) in enumerate(results[:2], 1):
            print(f"\nDocument {idx} (distance: {distance:.4f}):")
            print(f"  Source: {doc.metadata.get('source', 'unknown')}")
            print(f"  Content: {doc.page_content[:150]}...")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" RAG SYSTEM - EXAMPLE USAGE")
    print("="*70)
    
    try:
        # Run examples
        example_basic_usage()
        # Uncomment to run other examples
        # example_single_model_inference()
        # example_compare_models()
        # example_evaluate_response()
        # example_custom_retrieval()
        
        print("\n" + "="*70)
        print(" EXAMPLES COMPLETED")
        print("="*70)
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("\nMake sure you have:")
        print("1. PDF files in data/pdfs/")
        print("2. Built vector database (run with --rebuild-db)")
        print("3. Sufficient resources to load models")
