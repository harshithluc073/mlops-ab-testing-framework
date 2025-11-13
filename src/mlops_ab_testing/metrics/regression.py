"""
Regression metrics for model evaluation.

This module provides comprehensive regression metrics including
MSE, RMSE, MAE, R², MAPE, and more.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    mean_absolute_percentage_error,
    median_absolute_error
)
import logging

logger = logging.getLogger(__name__)


class RegressionMetrics:
    """Calculator for regression metrics."""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Initialize regression metrics calculator.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        
        # Validate shapes
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {self.y_true.shape} vs y_pred {self.y_pred.shape}"
            )
        
        # Cache for computed metrics
        self._metrics_cache: Dict[str, Any] = {}
    
    def calculate_all(self) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        # Error metrics
        metrics['mse'] = self.mse()
        metrics['rmse'] = self.rmse()
        metrics['mae'] = self.mae()
        metrics['median_ae'] = self.median_ae()
        metrics['max_error'] = self.max_error()
        
        # Percentage metrics
        try:
            metrics['mape'] = self.mape()
        except Exception as e:
            logger.warning(f"Could not calculate MAPE: {e}")
        
        try:
            metrics['smape'] = self.smape()
        except Exception as e:
            logger.warning(f"Could not calculate SMAPE: {e}")
        
        # Variance metrics
        metrics['r2_score'] = self.r2_score()
        metrics['explained_variance'] = self.explained_variance()
        metrics['adjusted_r2'] = self.adjusted_r2()
        
        # Custom metrics
        metrics['mean_error'] = self.mean_error()
        metrics['std_error'] = self.std_error()
        
        return metrics
    
    def mse(self) -> float:
        """Calculate Mean Squared Error."""
        return float(mean_squared_error(self.y_true, self.y_pred))
    
    def rmse(self) -> float:
        """Calculate Root Mean Squared Error."""
        return float(np.sqrt(self.mse()))
    
    def mae(self) -> float:
        """Calculate Mean Absolute Error."""
        return float(mean_absolute_error(self.y_true, self.y_pred))
    
    def median_ae(self) -> float:
        """Calculate Median Absolute Error."""
        return float(median_absolute_error(self.y_true, self.y_pred))
    
    def max_error(self) -> float:
        """Calculate Maximum Error."""
        return float(max_error(self.y_true, self.y_pred))
    
    def mape(self) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Note: Undefined when y_true contains zeros.
        """
        # Avoid division by zero
        mask = self.y_true != 0
        if not mask.any():
            raise ValueError("MAPE undefined when all true values are zero")
        
        return float(mean_absolute_percentage_error(self.y_true[mask], self.y_pred[mask]))
    
    def smape(self) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.
        
        SMAPE = 100 * mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|))
        """
        numerator = np.abs(self.y_true - self.y_pred)
        denominator = np.abs(self.y_true) + np.abs(self.y_pred)
        
        # Avoid division by zero
        mask = denominator != 0
        if not mask.any():
            raise ValueError("SMAPE undefined when all values are zero")
        
        return float(100 * np.mean(2 * numerator[mask] / denominator[mask]))
    
    def r2_score(self) -> float:
        """Calculate R² (coefficient of determination)."""
        return float(r2_score(self.y_true, self.y_pred))
    
    def adjusted_r2(self, n_features: Optional[int] = None) -> float:
        """
        Calculate Adjusted R².
        
        Args:
            n_features: Number of features (optional)
            
        Returns:
            Adjusted R² score
        """
        r2 = self.r2_score()
        n = len(self.y_true)
        
        if n_features is None:
            # Return regular R² if n_features not provided
            return r2
        
        if n <= n_features + 1:
            return r2
        
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return float(adjusted_r2)
    
    def explained_variance(self) -> float:
        """Calculate Explained Variance Score."""
        return float(explained_variance_score(self.y_true, self.y_pred))
    
    def mean_error(self) -> float:
        """Calculate Mean Error (bias)."""
        return float(np.mean(self.y_pred - self.y_true))
    
    def std_error(self) -> float:
        """Calculate Standard Deviation of Errors."""
        return float(np.std(self.y_pred - self.y_true))
    
    def residuals(self) -> np.ndarray:
        """Get residuals (prediction errors)."""
        return self.y_pred - self.y_true
    
    def relative_error(self) -> np.ndarray:
        """
        Calculate relative error.
        
        Returns:
            Array of relative errors
        """
        mask = self.y_true != 0
        relative_err = np.zeros_like(self.y_true)
        relative_err[mask] = (self.y_pred[mask] - self.y_true[mask]) / self.y_true[mask]
        return relative_err
    
    def mean_relative_error(self) -> float:
        """Calculate mean relative error."""
        rel_err = self.relative_error()
        mask = self.y_true != 0
        return float(np.mean(rel_err[mask])) if mask.any() else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary.
        
        Returns:
            Dictionary with all metrics
        """
        result = {
            'metrics': self.calculate_all(),
            'n_samples': len(self.y_true),
            'mean_true': float(np.mean(self.y_true)),
            'mean_pred': float(np.mean(self.y_pred)),
            'std_true': float(np.std(self.y_true)),
            'std_pred': float(np.std(self.y_pred))
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


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None,
    n_features: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metrics: List of specific metrics to calculate (None = all)
        n_features: Number of features (for adjusted R²)
        
    Returns:
        Dictionary of metric names to values
    """
    calculator = RegressionMetrics(y_true, y_pred)
    
    if metrics is None:
        return calculator.calculate_all()
    
    # Calculate only requested metrics
    result = {}
    for metric_name in metrics:
        if hasattr(calculator, metric_name):
            try:
                if metric_name == 'adjusted_r2':
                    result[metric_name] = calculator.adjusted_r2(n_features)
                else:
                    result[metric_name] = getattr(calculator, metric_name)()
            except Exception as e:
                logger.warning(f"Could not calculate {metric_name}: {e}")
        else:
            logger.warning(f"Unknown metric: {metric_name}")
    
    return result