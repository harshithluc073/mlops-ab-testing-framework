"""
Metrics visualization module.

This module provides visualization functions for metrics comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_plot: Optional[List[str]] = None,
    title: str = "Model Metrics Comparison",
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot comparison of metrics across models.
    
    Args:
        metrics_dict: Dictionary mapping model names to metrics
        metrics_to_plot: List of metrics to plot (None = all)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to DataFrame
    df = pd.DataFrame(metrics_dict).T
    
    if metrics_to_plot:
        df = df[metrics_to_plot]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(df.columns))
    width = 0.8 / len(df)
    
    for i, (model_name, row) in enumerate(df.iterrows()):
        offset = width * i - (width * (len(df) - 1)) / 2
        ax.bar(x + offset, row.values, width, label=model_name)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        labels: Class labels
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels or range(len(cm)),
        yticklabels=labels or range(len(cm)),
        ax=ax
    )
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_roc_curves(
    roc_data: Dict[str, tuple],
    title: str = "ROC Curves Comparison",
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.
    
    Args:
        roc_data: Dictionary mapping model names to (fpr, tpr, auc)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.3)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_metric_distribution(
    data: Dict[str, np.ndarray],
    metric_name: str = "Metric",
    title: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot distribution of metric values.
    
    Args:
        data: Dictionary mapping model names to metric values
        metric_name: Name of the metric
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, values in data.items():
        ax.hist(values, alpha=0.6, label=model_name, bins=30)
    
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title or f'{metric_name} Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_metrics_radar(
    metrics_dict: Dict[str, Dict[str, float]],
    title: str = "Metrics Radar Chart",
    figsize: tuple = (10, 10)
) -> plt.Figure:
    """
    Create radar chart for metrics comparison.
    
    Args:
        metrics_dict: Dictionary mapping model names to metrics
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    df = pd.DataFrame(metrics_dict).T
    
    # Normalize metrics to 0-1 range
    df_norm = (df - df.min()) / (df.max() - df.min())
    df_norm = df_norm.fillna(0)
    
    categories = list(df.columns)
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    for model_name, row in df_norm.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
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