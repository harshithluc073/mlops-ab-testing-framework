"""
Metrics module for MLOps A/B Testing Framework.

This module provides comprehensive metrics calculation for both
classification and regression tasks.
"""

from mlops_ab_testing.metrics.classification import (
    ClassificationMetrics,
    calculate_classification_metrics
)
from mlops_ab_testing.metrics.regression import (
    RegressionMetrics,
    calculate_regression_metrics
)
from mlops_ab_testing.metrics.evaluator import (
    MetricsEvaluator,
    evaluate_ab_test_results
)

__all__ = [
    'ClassificationMetrics',
    'calculate_classification_metrics',
    'RegressionMetrics',
    'calculate_regression_metrics',
    'MetricsEvaluator',
    'evaluate_ab_test_results',
]