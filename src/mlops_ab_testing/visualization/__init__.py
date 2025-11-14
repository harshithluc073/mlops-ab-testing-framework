"""
Visualization module for MLOps A/B Testing Framework.

This module provides visualization and reporting capabilities.
"""

from mlops_ab_testing.visualization.metrics_plots import (
    plot_metrics_comparison,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_metric_distribution,
    plot_metrics_radar
)
from mlops_ab_testing.visualization.explainability_plots import (
    plot_feature_importance,
    plot_importance_comparison,
    plot_shap_summary,
    plot_agreement_heatmap
)
from mlops_ab_testing.visualization.report_generator import (
    HTMLReportGenerator,
    generate_ab_test_report
)

__all__ = [
    'plot_metrics_comparison',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_metric_distribution',
    'plot_metrics_radar',
    'plot_feature_importance',
    'plot_importance_comparison',
    'plot_shap_summary',
    'plot_agreement_heatmap',
    'HTMLReportGenerator',
    'generate_ab_test_report',
]