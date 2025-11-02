"""
Comprehensive Evaluation Metrics for RAG System
"""
import time
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import evaluate

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class EvaluationMetrics:
    """Comprehensive evaluation metrics for RAG responses"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_cosine_similarity(self, text1: str, text2: str, embeddings_model) -> float:
        """Compute cosine similarity between two texts using embeddings"""
        try:
            embeddings = embeddings_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error computing cosine similarity: {e}")
            return 0.0
    
    def compute_bleu(self, reference: str, candidate: str) -> float:
        """Compute BLEU score"""
        try:
            reference_tokens = nltk.word_tokenize(reference.lower())
            candidate_tokens = nltk.word_tokenize(candidate.lower())
            
            smoothing = SmoothingFunction().method1
            score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
            return float(score)
        except Exception as e:
            print(f"Error computing BLEU: {e}")
            return 0.0
    
    def compute_meteor(self, reference: str, candidate: str) -> float:
        """Compute METEOR score"""
        try:
            reference_tokens = nltk.word_tokenize(reference.lower())
            candidate_tokens = nltk.word_tokenize(candidate.lower())
            score = meteor_score([reference_tokens], candidate_tokens)
            return float(score)
        except Exception as e:
            print(f"Error computing METEOR: {e}")
            return 0.0
    
    def compute_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Compute ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure,
            }
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def compute_bertscore(self, reference: str, candidate: str) -> Dict[str, float]:
        """Compute BERTScore (Precision, Recall, F1)"""
        try:
            P, R, F1 = bert_score([candidate], [reference], lang='en', verbose=False)
            return {
                'precision': float(P[0]),
                'recall': float(R[0]),
                'f1': float(F1[0])
            }
        except Exception as e:
            print(f"Error computing BERTScore: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def compute_completeness(self, reference: str, candidate: str) -> float:
        """
        Compute completeness score - how much of the reference is covered
        Uses ROUGE-L as a proxy for completeness
        """
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            # Use recall of ROUGE-L as completeness measure
            return float(scores['rougeL'].recall)
        except Exception as e:
            print(f"Error computing completeness: {e}")
            return 0.0
    
    def detect_hallucination(self, context: str, response: str, embeddings_model) -> float:
        """
        Detect hallucination by checking if response content is grounded in context
        Returns a score between 0 (no hallucination) and 1 (high hallucination)
        """
        try:
            # If response says it cannot answer, no hallucination
            if "cannot answer" in response.lower() or "don't know" in response.lower():
                return 0.0
            
            # Compute semantic similarity between response and context
            similarity = self.compute_cosine_similarity(response, context, embeddings_model)
            
            # Hallucination score is inverse of similarity
            # High similarity = low hallucination
            hallucination_score = 1.0 - similarity
            return float(hallucination_score)
        except Exception as e:
            print(f"Error detecting hallucination: {e}")
            return 0.5
    
    def compute_irrelevance(self, query: str, response: str, embeddings_model) -> float:
        """
        Compute irrelevance score - how off-topic the response is
        Returns a score between 0 (relevant) and 1 (irrelevant)
        """
        try:
            # Compute semantic similarity between query and response
            similarity = self.compute_cosine_similarity(query, response, embeddings_model)
            
            # Irrelevance is inverse of similarity
            irrelevance_score = 1.0 - similarity
            return float(irrelevance_score)
        except Exception as e:
            print(f"Error computing irrelevance: {e}")
            return 0.5
    
    def compute_all_metrics(self, 
                           query: str,
                           response: str,
                           reference: str,
                           context: str,
                           latency: float,
                           embeddings_model) -> Dict[str, float]:
        """Compute all evaluation metrics"""
        
        metrics = {
            'latency': latency,
        }
        
        # Similarity metrics
        metrics['cosine_similarity'] = self.compute_cosine_similarity(
            reference, response, embeddings_model
        )
        
        # BLEU and METEOR
        metrics['bleu'] = self.compute_bleu(reference, response)
        metrics['meteor'] = self.compute_meteor(reference, response)
        
        # ROUGE scores
        rouge_scores = self.compute_rouge(reference, response)
        metrics.update(rouge_scores)
        
        # BERTScore
        bertscore = self.compute_bertscore(reference, response)
        metrics['bertscore_f1'] = bertscore['f1']
        metrics['bertscore_precision'] = bertscore['precision']
        metrics['bertscore_recall'] = bertscore['recall']
        
        # Completeness
        metrics['completeness'] = self.compute_completeness(reference, response)
        
        # Hallucination and Irrelevance
        metrics['hallucination'] = self.detect_hallucination(context, response, embeddings_model)
        metrics['irrelevance'] = self.compute_irrelevance(query, response, embeddings_model)
        
        return metrics


class TrialEvaluator:
    """Run multiple trials and aggregate results"""
    
    def __init__(self, num_trials: int = 3):
        self.num_trials = num_trials
        self.metrics_calculator = EvaluationMetrics()
    
    def run_trials(self,
                   model_name: str,
                   generate_fn,
                   query: str,
                   reference: str,
                   context: str,
                   embeddings_model) -> Dict:
        """Run multiple trials and aggregate metrics"""
        
        all_trials = []
        
        print(f"\nRunning {self.num_trials} trials for {model_name}...")
        
        for trial_num in range(1, self.num_trials + 1):
            print(f"  Trial {trial_num}/{self.num_trials}...", end=" ")
            
            # Generate response
            response, latency = generate_fn(query)
            
            # Compute metrics
            metrics = self.metrics_calculator.compute_all_metrics(
                query=query,
                response=response,
                reference=reference,
                context=context,
                latency=latency,
                embeddings_model=embeddings_model
            )
            
            trial_result = {
                'trial_number': trial_num,
                'response': response,
                'metrics': metrics
            }
            
            all_trials.append(trial_result)
            print(f"Latency: {latency:.2f}s")
        
        # Aggregate metrics across trials
        aggregated_metrics = self._aggregate_metrics(all_trials)
        
        return {
            'model_name': model_name,
            'trials': all_trials,
            'aggregated_metrics': aggregated_metrics
        }
    
    def _aggregate_metrics(self, trials: List[Dict]) -> Dict:
        """Aggregate metrics across trials"""
        all_metrics = [trial['metrics'] for trial in trials]
        
        aggregated = {}
        
        # Get all metric keys
        metric_keys = all_metrics[0].keys()
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated
