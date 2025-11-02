"""
Main execution script for RAG system with evaluation
"""
import argparse
import sys
from typing import List, Dict
import config
from rag_system import RAGSystem
from llm_models import ModelManager
from evaluation_metrics import TrialEvaluator
from visualization import PerformanceVisualizer


class RAGEvaluationPipeline:
    """Complete pipeline for RAG evaluation"""
    
    def __init__(self, 
                 pdf_path: str = None,
                 load_existing_db: bool = True,
                 num_trials: int = 3):
        """
        Initialize the evaluation pipeline
        
        Args:
            pdf_path: Path to PDF directory
            load_existing_db: Whether to load existing vector database
            num_trials: Number of trials for evaluation
        """
        print("\n" + "="*60)
        print("INITIALIZING RAG EVALUATION PIPELINE")
        print("="*60)
        
        # Initialize RAG system
        print("\n1. Setting up RAG system...")
        self.rag_system = RAGSystem(
            pdf_path=pdf_path,
            load_existing=load_existing_db
        )
        
        # Initialize model manager
        print("\n2. Setting up model manager...")
        self.model_manager = ModelManager()
        
        # Initialize evaluator
        print("\n3. Setting up evaluator...")
        self.evaluator = TrialEvaluator(num_trials=num_trials)
        
        # Initialize visualizer
        print("\n4. Setting up visualizer...")
        self.visualizer = PerformanceVisualizer(results_dir=config.RESULTS_PATH)
        
        print("\n" + "="*60)
        print("INITIALIZATION COMPLETE")
        print("="*60)
    
    def evaluate_models(self, 
                       test_questions: List[Dict],
                       models_to_test: List[str] = None):
        """
        Evaluate multiple models on test questions
        
        Args:
            test_questions: List of dicts with 'question' and 'reference_answer'
            models_to_test: List of model keys to test (default: all models)
        """
        if models_to_test is None:
            models_to_test = list(config.MODELS.keys())
        
        print("\n" + "="*60)
        print(f"EVALUATING {len(models_to_test)} MODELS ON {len(test_questions)} QUESTIONS")
        print("="*60)
        
        all_results = []
        
        for question_idx, test_case in enumerate(test_questions, 1):
            query = test_case['question']
            reference = test_case['reference_answer']
            
            print(f"\n{'='*60}")
            print(f"QUESTION {question_idx}/{len(test_questions)}")
            print(f"{'='*60}")
            print(f"Q: {query}")
            print(f"Reference Answer: {reference}")
            
            # Retrieve context
            context = self.rag_system.retrieve_context(query)
            prompt = self.rag_system.create_prompt(query, context)
            
            print(f"\nRetrieved {len(context)} characters of context")
            
            # Evaluate each model
            question_results = []
            
            for model_name in models_to_test:
                print(f"\n{'-'*60}")
                print(f"EVALUATING: {model_name}")
                print(f"{'-'*60}")
                
                try:
                    # Define generation function for this model
                    def generate_fn(prompt_text):
                        return self.model_manager.generate_response(model_name, prompt_text)
                    
                    # Run trials and evaluate
                    result = self.evaluator.run_trials(
                        model_name=model_name,
                        generate_fn=lambda q: generate_fn(prompt),
                        query=query,
                        reference=reference,
                        context=context,
                        embeddings_model=self.rag_system.vector_store.embedding_model
                    )
                    
                    question_results.append(result)
                    
                    # Print summary
                    agg = result['aggregated_metrics']
                    print(f"\nSummary for {model_name}:")
                    print(f"  Avg Latency: {agg['latency_mean']:.2f}s (?{agg['latency_std']:.2f})")
                    print(f"  Cosine Sim: {agg['cosine_similarity_mean']:.3f}")
                    print(f"  BERTScore F1: {agg['bertscore_f1_mean']:.3f}")
                    print(f"  Completeness: {agg['completeness_mean']:.3f}")
                    print(f"  Hallucination: {agg['hallucination_mean']:.3f}")
                    print(f"  Irrelevance: {agg['irrelevance_mean']:.3f}")
                    
                except Exception as e:
                    print(f"ERROR evaluating {model_name}: {str(e)}")
                    continue
            
            all_results.extend(question_results)
        
        return all_results
    
    def run_complete_evaluation(self, 
                               test_questions: List[Dict],
                               models_to_test: List[str] = None):
        """Run complete evaluation and generate visualizations"""
        
        # Evaluate models
        results = self.evaluate_models(test_questions, models_to_test)
        
        # Generate visualizations
        if results:
            self.visualizer.generate_all_visualizations(results)
        
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up resources...")
        self.model_manager.unload_all()


def create_sample_test_cases() -> List[Dict]:
    """Create sample test cases - REPLACE WITH YOUR OWN"""
    return [
        {
            'question': 'What is the main topic discussed in the documents?',
            'reference_answer': 'The documents discuss various technical concepts and methodologies.'
        },
        {
            'question': 'What are the key findings mentioned?',
            'reference_answer': 'The key findings include multiple insights and recommendations.'
        }
    ]


def main():
    parser = argparse.ArgumentParser(description='RAG System Evaluation Pipeline')
    parser.add_argument('--pdf-path', type=str, default=None,
                       help='Path to PDF directory')
    parser.add_argument('--load-existing', action='store_true', default=True,
                       help='Load existing vector database')
    parser.add_argument('--rebuild-db', action='store_true',
                       help='Rebuild vector database from scratch')
    parser.add_argument('--num-trials', type=int, default=3,
                       help='Number of trials for evaluation')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Models to evaluate (default: all)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGEvaluationPipeline(
        pdf_path=args.pdf_path,
        load_existing_db=not args.rebuild_db,
        num_trials=args.num_trials
    )
    
    # Create test cases - MODIFY THIS WITH YOUR TEST QUESTIONS
    test_questions = create_sample_test_cases()
    
    print("\n" + "="*60)
    print("IMPORTANT: Using sample test cases!")
    print("Please modify create_sample_test_cases() with your actual questions")
    print("="*60)
    
    try:
        # Run evaluation
        results = pipeline.run_complete_evaluation(
            test_questions=test_questions,
            models_to_test=args.models
        )
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60)
        print(f"Results saved to: {config.RESULTS_PATH}")
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\n\nERROR during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        pipeline.cleanup()


if __name__ == "__main__":
    main()
