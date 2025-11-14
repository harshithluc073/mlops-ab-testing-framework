"""
Unit tests for metrics module.
"""

import pytest
import numpy as np
from mlops_ab_testing.metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    calculate_classification_metrics
)


def test_classification_metrics():
    """Test classification metrics."""
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])
    
    metrics = ClassificationMetrics(y_true, y_pred)
    results = metrics.calculate_all()
    
    assert 'accuracy' in results
    assert 'precision' in results
    assert 'recall' in results
    assert 'f1_score' in results
    assert 0 <= results['accuracy'] <= 1


def test_regression_metrics():
    """Test regression metrics."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    metrics = RegressionMetrics(y_true, y_pred)
    results = metrics.calculate_all()
    
    assert 'mse' in results
    assert 'rmse' in results
    assert 'mae' in results
    assert 'r2_score' in results
    assert results['mse'] >= 0
    assert results['rmse'] >= 0


def test_calculate_classification_metrics():
    """Test classification metrics calculation function."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    metrics = calculate_classification_metrics(y_true, y_pred)
    
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])