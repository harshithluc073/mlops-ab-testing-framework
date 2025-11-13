"""
Classification metrics for model evaluation.

This module provides comprehensive classification metrics including
accuracy, precision, recall, F1, ROC-AUC, and more.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
import logging

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """Calculator for classification metrics."""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_proba: Optional[np.ndarray] = None,
                 labels: Optional[List[str]] = None):
        """
        Initialize classification metrics calculator.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional, for probabilistic metrics)
            labels: Class labels (optional)
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_proba = np.asarray(y_proba) if y_proba is not None else None
        self.labels = labels
        
        # Detect if binary or multiclass
        self.n_classes = len(np.unique(y_true))
        self.is_binary = self.n_classes == 2
        
        # Cache for computed metrics
        self._metrics_cache: Dict[str, Any] = {}
    
    def calculate_all(self) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        # Basic metrics (always available)
        metrics['accuracy'] = self.accuracy()
        metrics['balanced_accuracy'] = self.balanced_accuracy()
        
        # Metrics that need averaging strategy for multiclass
        if self.is_binary:
            metrics['precision'] = self.precision(average='binary')
            metrics['recall'] = self.recall(average='binary')
            metrics['f1_score'] = self.f1_score(average='binary')
            metrics['specificity'] = self.specificity()
            metrics['sensitivity'] = self.sensitivity()
        else:
            metrics['precision_macro'] = self.precision(average='macro')
            metrics['precision_micro'] = self.precision(average='micro')
            metrics['precision_weighted'] = self.precision(average='weighted')
            metrics['recall_macro'] = self.recall(average='macro')
            metrics['recall_micro'] = self.recall(average='micro')
            metrics['recall_weighted'] = self.recall(average='weighted')
            metrics['f1_score_macro'] = self.f1_score(average='macro')
            metrics['f1_score_micro'] = self.f1_score(average='micro')
            metrics['f1_score_weighted'] = self.f1_score(average='weighted')
        
        # Agreement metrics
        metrics['cohen_kappa'] = self.cohen_kappa()
        metrics['matthews_corrcoef'] = self.matthews_corrcoef()
        
        # Probabilistic metrics (if probabilities available)
        if self.y_proba is not None:
            try:
                if self.is_binary:
                    metrics['roc_auc'] = self.roc_auc()
                    metrics['pr_auc'] = self.pr_auc()
                else:
                    metrics['roc_auc_ovr'] = self.roc_auc(multi_class='ovr')
                    metrics['roc_auc_ovo'] = self.roc_auc(multi_class='ovo')
                
                metrics['log_loss'] = self.log_loss()
                metrics['brier_score'] = self.brier_score()
            except Exception as e:
                logger.warning(f"Could not calculate probabilistic metrics: {e}")
        
        return metrics
    
    def accuracy(self) -> float:
        """Calculate accuracy."""
        return float(accuracy_score(self.y_true, self.y_pred))
    
    def balanced_accuracy(self) -> float:
        """Calculate balanced accuracy (accounts for imbalanced classes)."""
        return float(balanced_accuracy_score(self.y_true, self.y_pred))
    
    def precision(self, average: str = 'binary') -> float:
        """
        Calculate precision.
        
        Args:
            average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')
        """
        return float(precision_score(self.y_true, self.y_pred, average=average, zero_division=0))
    
    def recall(self, average: str = 'binary') -> float:
        """
        Calculate recall (sensitivity).
        
        Args:
            average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')
        """
        return float(recall_score(self.y_true, self.y_pred, average=average, zero_division=0))
    
    def sensitivity(self) -> float:
        """Calculate sensitivity (same as recall for binary classification)."""
        if not self.is_binary:
            raise ValueError("Sensitivity is only defined for binary classification")
        return self.recall(average='binary')
    
    def specificity(self) -> float:
        """Calculate specificity (true negative rate)."""
        if not self.is_binary:
            raise ValueError("Specificity is only defined for binary classification")
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    def f1_score(self, average: str = 'binary') -> float:
        """
        Calculate F1 score.
        
        Args:
            average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')
        """
        return float(f1_score(self.y_true, self.y_pred, average=average, zero_division=0))
    
    def cohen_kappa(self) -> float:
        """Calculate Cohen's Kappa score."""
        return float(cohen_kappa_score(self.y_true, self.y_pred))
    
    def matthews_corrcoef(self) -> float:
        """Calculate Matthews Correlation Coefficient."""
        return float(matthews_corrcoef(self.y_true, self.y_pred))
    
    def roc_auc(self, multi_class: str = 'raise') -> float:
        """
        Calculate ROC AUC score.
        
        Args:
            multi_class: Strategy for multiclass ('raise', 'ovr', 'ovo')
        """
        if self.y_proba is None:
            raise ValueError("Probabilities required for ROC AUC calculation")
        
        if self.is_binary:
            # For binary classification, use positive class probabilities
            if self.y_proba.ndim == 2:
                proba = self.y_proba[:, 1]
            else:
                proba = self.y_proba
            return float(roc_auc_score(self.y_true, proba))
        else:
            # For multiclass
            return float(roc_auc_score(
                self.y_true, 
                self.y_proba, 
                multi_class=multi_class,
                average='macro'
            ))
    
    def pr_auc(self) -> float:
        """Calculate Precision-Recall AUC (average precision)."""
        if self.y_proba is None:
            raise ValueError("Probabilities required for PR AUC calculation")
        
        if not self.is_binary:
            raise ValueError("PR AUC currently only supported for binary classification")
        
        if self.y_proba.ndim == 2:
            proba = self.y_proba[:, 1]
        else:
            proba = self.y_proba
        
        return float(average_precision_score(self.y_true, proba))
    
    def log_loss(self) -> float:
        """Calculate log loss (cross-entropy loss)."""
        if self.y_proba is None:
            raise ValueError("Probabilities required for log loss calculation")
        
        return float(log_loss(self.y_true, self.y_proba))
    
    def brier_score(self) -> float:
        """Calculate Brier score (mean squared error of probabilities)."""
        if self.y_proba is None:
            raise ValueError("Probabilities required for Brier score calculation")
        
        if not self.is_binary:
            raise ValueError("Brier score currently only supported for binary classification")
        
        if self.y_proba.ndim == 2:
            proba = self.y_proba[:, 1]
        else:
            proba = self.y_proba
        
        return float(np.mean((self.y_true - proba) ** 2))
    
    def confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(self.y_true, self.y_pred)
    
    def classification_report(self) -> str:
        """Get detailed classification report."""
        return classification_report(self.y_true, self.y_pred, labels=self.labels)
    
    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve data.
        
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        if self.y_proba is None:
            raise ValueError("Probabilities required for ROC curve")
        
        if not self.is_binary:
            raise ValueError("ROC curve currently only supported for binary classification")
        
        if self.y_proba.ndim == 2:
            proba = self.y_proba[:, 1]
        else:
            proba = self.y_proba
        
        return roc_curve(self.y_true, proba)
    
    def get_precision_recall_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get precision-recall curve data.
        
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        if self.y_proba is None:
            raise ValueError("Probabilities required for PR curve")
        
        if not self.is_binary:
            raise ValueError("PR curve currently only supported for binary classification")
        
        if self.y_proba.ndim == 2:
            proba = self.y_proba[:, 1]
        else:
            proba = self.y_proba
        
        return precision_recall_curve(self.y_true, proba)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary.
        
        Returns:
            Dictionary with all metrics
        """
        result = {
            'metrics': self.calculate_all(),
            'confusion_matrix': self.confusion_matrix().tolist(),
            'n_classes': self.n_classes,
            'is_binary': self.is_binary,
            'n_samples': len(self.y_true)
        }
        
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert metrics to DataFrame.
        
        Returns:
            DataFrame with metrics
        """
        metrics = self.calculate_all()
        df = pd.DataFrame([metrics]).T
        df.columns = ['Value']
        df.index.name = 'Metric'
        return df


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        labels: Class labels (optional)
        metrics: List of specific metrics to calculate (None = all)
        
    Returns:
        Dictionary of metric names to values
    """
    calculator = ClassificationMetrics(y_true, y_pred, y_proba, labels)
    
    if metrics is None:
        return calculator.calculate_all()
    
    # Calculate only requested metrics
    result = {}
    for metric_name in metrics:
        if hasattr(calculator, metric_name):
            try:
                result[metric_name] = getattr(calculator, metric_name)()
            except Exception as e:
                logger.warning(f"Could not calculate {metric_name}: {e}")
        else:
            logger.warning(f"Unknown metric: {metric_name}")
    
    return result