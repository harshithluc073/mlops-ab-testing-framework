"""
Explainability visualization module.

This module provides visualization functions for SHAP and LIME explanations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import logging

from mlops_ab_testing.explainability.base import ExplanationResult

logger = logging.getLogger(__name__)


def plot_feature_importance(
    result: ExplanationResult,
    top_n: int = 10,
    title: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        result: ExplanationResult object
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    top_features = result.get_top_features(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['green' if x > 0 else 'red' for x in top_features['Importance']]
    
    ax.barh(range(len(top_features)), top_features['Importance'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title or f'{result.model_name} - Feature Importance ({result.method.upper()})',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_importance_comparison(
    results: Dict[str, ExplanationResult],
    top_n: int = 10,
    title: str = "Feature Importance Comparison",
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Compare feature importance across models.
    
    Args:
        results: Dictionary mapping model names to ExplanationResult
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from mlops_ab_testing.explainability import FeatureComparator
    
    comparator = FeatureComparator()
    comparison_df = comparator.compare_multiple_models(results)
    
    # Get top N by mean importance
    top_df = comparison_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot importance for each model
    x = np.arange(len(top_df))
    width = 0.8 / len(results)
    
    for i, model_name in enumerate(results.keys()):
        importance_col = f'{model_name}_Importance'
        if importance_col in top_df.columns:
            offset = width * i - (width * (len(results) - 1)) / 2
            ax.bar(x + offset, top_df[importance_col], width, label=model_name)
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_df['Feature'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: List[str],
    max_display: int = 10,
    title: str = "SHAP Summary",
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plot SHAP summary (beeswarm-style).
    
    Args:
        shap_values: SHAP values array
        features: Feature values array
        feature_names: Names of features
        max_display: Maximum features to display
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Get top features
    top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, idx in enumerate(top_indices):
        # Plot SHAP values as scatter
        y = np.full(len(shap_values), i)
        colors = features[:, idx]
        
        ax.scatter(shap_values[:, idx], y, c=colors, cmap='RdBu_r',
                  alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([feature_names[i] for i in top_indices])
    ax.set_xlabel('SHAP Value (impact on model output)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdBu_r')
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Feature Value', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_agreement_heatmap(
    results: Dict[str, ExplanationResult],
    title: str = "Model Agreement Heatmap",
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plot heatmap showing feature importance agreement between models.
    
    Args:
        results: Dictionary mapping model names to ExplanationResult
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from mlops_ab_testing.explainability import create_importance_matrix
    
    importance_matrix, model_names, feature_names = create_importance_matrix(results)
    
    # Normalize by row (model)
    importance_matrix_norm = importance_matrix / (np.abs(importance_matrix).sum(axis=1, keepdims=True) + 1e-10)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        importance_matrix_norm,
        xticklabels=feature_names,
        yticklabels=model_names,
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Normalized Importance'},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 300):
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib figure
        filepath: Path to save file
        dpi: DPI for saved image
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved figure to {filepath}")