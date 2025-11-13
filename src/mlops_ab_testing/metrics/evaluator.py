"""
Main metrics evaluation module.

This module provides the interface for evaluating model performance
and comparing multiple models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging

from mlops_ab_testing.metrics.classification import (
    ClassificationMetrics,
    calculate_classification_metrics
)
from mlops_ab_testing.metrics.regression import (
    RegressionMetrics,
    calculate_regression_metrics
)

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """
    Main evaluator for calculating and comparing model metrics.
    """
    
    def __init__(self, task_type: str = 'auto'):
        """
        Initialize metrics evaluator.
        
        Args:
            task_type: Type of task ('classification', 'regression', or 'auto')
        """
        self.task_type = task_type
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            model_name: Name of the model
            y_true: Ground truth
            y_pred: Predictions
            y_proba: Prediction probabilities (for classification)
            task_type: Override task type
            
        Returns:
            Dictionary of metrics
        """
        task = task_type or self.task_type
        
        # Auto-detect task type if needed
        if task == 'auto':
            task = self._detect_task_type(y_true, y_pred)
        
        logger.info(f"Evaluating {model_name} ({task})...")
        
        if task == 'classification':
            metrics_calc = ClassificationMetrics(y_true, y_pred, y_proba)
            metrics = metrics_calc.to_dict()
        elif task == 'regression':
            metrics_calc = RegressionMetrics(y_true, y_pred)
            metrics = metrics_calc.to_dict()
        else:
            raise ValueError(f"Unknown task type: {task}")
        
        # Store results
        self.results[model_name] = {
            'task_type': task,
            'metrics': metrics['metrics'],
            'n_samples': len(y_true),
            'metadata': {k: v for k, v in metrics.items() if k != 'metrics'}
        }
        
        logger.info(f"  Calculated {len(metrics['metrics'])} metrics")
        
        return self.results[model_name]
    
    def _detect_task_type(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Auto-detect task type based on data.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            
        Returns:
            Task type ('classification' or 'regression')
        """
        # Check if predictions are integers (likely classification)
        unique_values = len(np.unique(y_true))
        
        # Heuristic: if less than 20 unique values and all integers, classify as classification
        if unique_values < 20 and np.all(y_true == y_true.astype(int)):
            return 'classification'
        else:
            return 'regression'
    
    def compare_models(
        self,
        baseline_model: str,
        comparison_models: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            baseline_model: Name of baseline model (e.g., 'champion')
            comparison_models: List of models to compare (None = all except baseline)
            
        Returns:
            DataFrame with comparison
        """
        if baseline_model not in self.results:
            raise ValueError(f"Baseline model '{baseline_model}' not found")
        
        if comparison_models is None:
            comparison_models = [m for m in self.results.keys() if m != baseline_model]
        
        if not comparison_models:
            raise ValueError("No comparison models specified")
        
        # Get baseline metrics
        baseline_metrics = self.results[baseline_model]['metrics']
        
        # Build comparison table
        comparison_data = []
        
        for metric_name in baseline_metrics.keys():
            row = {
                'Metric': metric_name,
                baseline_model: baseline_metrics[metric_name]
            }
            
            for model_name in comparison_models:
                if model_name in self.results:
                    model_metrics = self.results[model_name]['metrics']
                    if metric_name in model_metrics:
                        value = model_metrics[metric_name]
                        baseline_value = baseline_metrics[metric_name]
                        
                        row[model_name] = value
                        
                        # Calculate difference
                        diff = value - baseline_value
                        pct_change = (diff / baseline_value * 100) if baseline_value != 0 else 0
                        row[f'{model_name}_diff'] = diff
                        row[f'{model_name}_pct_change'] = pct_change
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of all evaluated models.
        
        Returns:
            DataFrame with summary statistics
        """
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            
            # Get key metrics based on task type
            if result['task_type'] == 'classification':
                key_metrics = {
                    'Model': model_name,
                    'Task': 'Classification',
                    'Samples': result['n_samples'],
                    'Accuracy': metrics.get('accuracy', np.nan),
                    'F1': metrics.get('f1_score', metrics.get('f1_score_macro', np.nan)),
                    'ROC-AUC': metrics.get('roc_auc', metrics.get('roc_auc_ovr', np.nan))
                }
            else:
                key_metrics = {
                    'Model': model_name,
                    'Task': 'Regression',
                    'Samples': result['n_samples'],
                    'RMSE': metrics.get('rmse', np.nan),
                    'MAE': metrics.get('mae', np.nan),
                    'R²': metrics.get('r2_score', np.nan)
                }
            
            summary_data.append(key_metrics)
        
        return pd.DataFrame(summary_data)
    
    def get_detailed_metrics(self, model_name: str) -> pd.DataFrame:
        """
        Get detailed metrics for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with all metrics
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found")
        
        metrics = self.results[model_name]['metrics']
        
        df = pd.DataFrame([metrics]).T
        df.columns = ['Value']
        df.index.name = 'Metric'
        
        return df
    
    def get_winner(
        self,
        metric_name: str,
        higher_is_better: bool = True
    ) -> str:
        """
        Determine the winning model based on a specific metric.
        
        Args:
            metric_name: Name of the metric to compare
            higher_is_better: Whether higher values are better
            
        Returns:
            Name of the winning model
        """
        if not self.results:
            raise ValueError("No models evaluated")
        
        scores = {}
        for model_name, result in self.results.items():
            if metric_name in result['metrics']:
                scores[model_name] = result['metrics'][metric_name]
        
        if not scores:
            raise ValueError(f"Metric '{metric_name}' not found in any model")
        
        if higher_is_better:
            winner = max(scores.items(), key=lambda x: x[1])
        else:
            winner = min(scores.items(), key=lambda x: x[1])
        
        return winner[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all results to dictionary.
        
        Returns:
            Dictionary with all results
        """
        return {
            'task_type': self.task_type,
            'n_models': len(self.results),
            'models': self.results
        }


def evaluate_ab_test_results(
    results,  # ABTestResults object
    task_type: str = 'auto'
) -> MetricsEvaluator:
    """
    Evaluate A/B test results.
    
    Args:
        results: ABTestResults object from framework.run_test()
        task_type: Type of task ('classification', 'regression', or 'auto')
        
    Returns:
        MetricsEvaluator with computed metrics
    """
    evaluator = MetricsEvaluator(task_type=task_type)
    
    # Evaluate each model
    for model_name, pred_result in results.predictions.items():
        # Get predictions for this model's samples
        sample_indices = pred_result.sample_indices
        y_true = results.ground_truth[sample_indices]
        y_pred = pred_result.predictions
        y_proba = pred_result.probabilities
        
        evaluator.evaluate_model(
            model_name=model_name,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            task_type=task_type
        )
    
    return evaluator