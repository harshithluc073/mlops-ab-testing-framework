"""
Explainability module for MLOps A/B Testing Framework.

This module provides model explainability through SHAP, LIME,
and other interpretation methods.
"""

from mlops_ab_testing.explainability.base import (
    BaseExplainer,
    ExplainerConfig,
    ExplanationResult,
    get_sample_indices,
    aggregate_explanations,
    compare_feature_importance
)
from mlops_ab_testing.explainability.shap_explainer import (
    SHAPExplainer,
    explain_model_with_shap,
    compare_shap_explanations
)
from mlops_ab_testing.explainability.lime_explainer import (
    LIMEExplainer,
    explain_model_with_lime,
    compare_lime_explanations
)
from mlops_ab_testing.explainability.comparison import (
    FeatureComparator,
    analyze_feature_stability,
    get_consensus_features,
    create_importance_matrix,
    summarize_comparison
)

__all__ = [
    'BaseExplainer',
    'ExplainerConfig',
    'ExplanationResult',
    'get_sample_indices',
    'aggregate_explanations',
    'compare_feature_importance',
    'SHAPExplainer',
    'explain_model_with_shap',
    'compare_shap_explanations',
    'LIMEExplainer',
    'explain_model_with_lime',
    'compare_lime_explanations',
    'FeatureComparator',
    'analyze_feature_stability',
    'get_consensus_features',
    'create_importance_matrix',
    'summarize_comparison',
]