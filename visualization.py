"""
Visualization utilities for comparing model performance
"""
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path


class PerformanceVisualizer:
    """Create visualizations for model comparison"""
    
    def __init__(self, results_dir: str = './results'):
        self.results_dir = results_dir
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_comparison_bars(self, 
                            comparison_data: Dict,
                            metric_name: str,
                            save_name: str = None):
        """Create bar chart comparing models on a specific metric"""
        
        models = list(comparison_data.keys())
        values = [comparison_data[model] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Model Comparison: {metric_name}', fontsize=14, fontweight='bold')
        plt.ylabel(metric_name, fontsize=12)
        plt.xlabel('Model', fontsize=12)
        plt.xticks(rotation=15)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.results_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_all_metrics_comparison(self, results: List[Dict]):
        """Create comprehensive comparison of all metrics across models"""
        
        # Prepare data
        metrics_to_plot = [
            'latency_mean',
            'cosine_similarity_mean',
            'bertscore_f1_mean',
            'completeness_mean',
            'hallucination_mean',
            'irrelevance_mean',
            'meteor_mean',
            'bleu_mean'
        ]
        
        metric_labels = {
            'latency_mean': 'Latency (s)',
            'cosine_similarity_mean': 'Cosine Similarity',
            'bertscore_f1_mean': 'F1 BERTScore',
            'completeness_mean': 'Completeness',
            'hallucination_mean': 'Hallucination',
            'irrelevance_mean': 'Irrelevance',
            'meteor_mean': 'METEOR',
            'bleu_mean': 'BLEU'
        }
        
        # Create individual bar charts for each metric
        for metric in metrics_to_plot:
            comparison_data = {}
            for result in results:
                model_name = result['model_name']
                if metric in result['aggregated_metrics']:
                    comparison_data[model_name] = result['aggregated_metrics'][metric]
            
            if comparison_data:
                self.plot_comparison_bars(
                    comparison_data,
                    metric_labels.get(metric, metric),
                    f'comparison_{metric}'
                )
        
        # Create comprehensive multi-metric comparison
        self._plot_multi_metric_comparison(results, metrics_to_plot, metric_labels)
    
    def _plot_multi_metric_comparison(self, 
                                     results: List[Dict],
                                     metrics: List[str],
                                     metric_labels: Dict):
        """Create a comprehensive multi-metric comparison chart"""
        
        # Prepare data for plotting
        data = []
        for result in results:
            model_name = result['model_name']
            for metric in metrics:
                if metric in result['aggregated_metrics']:
                    value = result['aggregated_metrics'][metric]
                    data.append({
                        'Model': model_name,
                        'Metric': metric_labels.get(metric, metric),
                        'Value': value
                    })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            metric_label = metric_labels.get(metric, metric)
            metric_data = df[df['Metric'] == metric_label]
            
            if not metric_data.empty:
                ax = axes[idx]
                models = metric_data['Model'].values
                values = metric_data['Value'].values
                
                bars = ax.bar(models, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                ax.set_title(metric_label, fontsize=11, fontweight='bold')
                ax.set_ylabel('Score', fontsize=10)
                ax.tick_params(axis='x', rotation=15)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Model Performance Comparison', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'comprehensive_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_trial_consistency(self, results: List[Dict]):
        """Plot consistency across trials for each model"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics_to_show = ['latency', 'cosine_similarity', 'bertscore_f1']
        metric_labels = ['Latency (s)', 'Cosine Similarity', 'BERTScore F1']
        
        for ax, metric, label in zip(axes, metrics_to_show, metric_labels):
            for result in results:
                model_name = result['model_name']
                trials = result['trials']
                
                trial_numbers = [t['trial_number'] for t in trials]
                values = [t['metrics'][metric] for t in trials]
                
                ax.plot(trial_numbers, values, marker='o', label=model_name, linewidth=2)
            
            ax.set_title(f'Trial Consistency: {label}', fontweight='bold')
            ax.set_xlabel('Trial Number')
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Model Consistency Across Trials', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'trial_consistency.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_table(self, results: List[Dict]) -> pd.DataFrame:
        """Create a summary table of all results"""
        
        summary_data = []
        
        for result in results:
            model_name = result['model_name']
            agg_metrics = result['aggregated_metrics']
            
            row = {
                'Model': model_name,
                'Avg Latency (s)': f"{agg_metrics.get('latency_mean', 0):.3f} ? {agg_metrics.get('latency_std', 0):.3f}",
                'Cosine Similarity': f"{agg_metrics.get('cosine_similarity_mean', 0):.3f}",
                'BERTScore F1': f"{agg_metrics.get('bertscore_f1_mean', 0):.3f}",
                'Completeness': f"{agg_metrics.get('completeness_mean', 0):.3f}",
                'Hallucination': f"{agg_metrics.get('hallucination_mean', 0):.3f}",
                'Irrelevance': f"{agg_metrics.get('irrelevance_mean', 0):.3f}",
                'METEOR': f"{agg_metrics.get('meteor_mean', 0):.3f}",
                'BLEU': f"{agg_metrics.get('bleu_mean', 0):.3f}",
            }
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # Save to CSV
        csv_path = os.path.join(self.results_dir, 'summary_table.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSummary table saved to: {csv_path}")
        
        return df
    
    def save_detailed_results(self, results: List[Dict]):
        """Save detailed results to JSON"""
        json_path = os.path.join(self.results_dir, 'detailed_results.json')
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Detailed results saved to: {json_path}")
    
    def generate_all_visualizations(self, results: List[Dict]):
        """Generate all visualizations and save results"""
        
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Individual metric comparisons
        self.plot_all_metrics_comparison(results)
        
        # Trial consistency
        self.plot_trial_consistency(results)
        
        # Summary table
        summary_df = self.create_summary_table(results)
        print("\n" + "="*60)
        print("SUMMARY TABLE")
        print("="*60)
        print(summary_df.to_string(index=False))
        
        # Save detailed results
        self.save_detailed_results(results)
        
        print("\n" + "="*60)
        print(f"All visualizations saved to: {self.results_dir}")
        print("="*60)
