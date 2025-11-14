"""
Feature importance comparison utilities.

This module provides tools for comparing feature importance
across different models and explanation methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

from mlops_ab_testing.explainability.base import ExplanationResult

logger = logging.getLogger(__name__)


class FeatureComparator:
    """Compare feature importance across multiple models."""
    
    def __init__(self):
        """Initialize feature comparator."""
        self.comparisons: Dict[str, pd.DataFrame] = {}
    
    def compare_two_models(
        self,
        result_a: ExplanationResult,
        result_b: ExplanationResult,
        model_a_name: Optional[str] = None,
        model_b_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare feature importance between two models.
        
        Args:
            result_a: ExplanationResult for model A
            result_b: ExplanationResult for model B
            model_a_name: Name for model A (uses result name if None)
            model_b_name: Name for model B (uses result name if None)
            
        Returns:
            DataFrame with comparison
        """
        model_a_name = model_a_name or result_a.model_name
        model_b_name = model_b_name or result_b.model_name
        
        # Ensure feature names match
        if result_a.feature_names != result_b.feature_names:
            raise ValueError("Feature names don't match between models")
        
        # Create comparison dataframe
        df = pd.DataFrame({
            'Feature': result_a.feature_names,
            f'{model_a_name}_Importance': result_a.feature_importance,
            f'{model_b_name}_Importance': result_b.feature_importance
        })
        
        # Calculate differences
        df['Difference'] = df[f'{model_b_name}_Importance'] - df[f'{model_a_name}_Importance']
        df['Abs_Difference'] = np.abs(df['Difference'])
        
        # Calculate percentage change (handle division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_change = (df['Difference'] / df[f'{model_a_name}_Importance'].abs()) * 100
            pct_change = np.where(np.isfinite(pct_change), pct_change, 0)
        df['Pct_Change'] = pct_change
        
        # Add rankings
        df[f'{model_a_name}_Rank'] = df[f'{model_a_name}_Importance'].abs().rank(ascending=False)
        df[f'{model_b_name}_Rank'] = df[f'{model_b_name}_Importance'].abs().rank(ascending=False)
        df['Rank_Change'] = df[f'{model_a_name}_Rank'] - df[f'{model_b_name}_Rank']
        
        # Sort by absolute difference
        df = df.sort_values('Abs_Difference', ascending=False).reset_index(drop=True)
        
        # Store comparison
        comparison_key = f"{model_a_name}_vs_{model_b_name}"
        self.comparisons[comparison_key] = df
        
        return df
    
    def compare_multiple_models(
        self,
        results: Dict[str, ExplanationResult]
    ) -> pd.DataFrame:
        """
        Compare feature importance across multiple models.
        
        Args:
            results: Dictionary mapping model names to ExplanationResult objects
            
        Returns:
            DataFrame with all model comparisons
        """
        if len(results) < 2:
            raise ValueError("Need at least 2 models to compare")
        
        # Get feature names from first result
        first_result = list(results.values())[0]
        feature_names = first_result.feature_names
        
        # Build comparison dataframe
        data = {'Feature': feature_names}
        
        for model_name, result in results.items():
            if result.feature_names != feature_names:
                raise ValueError(f"Feature names don't match for {model_name}")
            
            data[f'{model_name}_Importance'] = result.feature_importance
            data[f'{model_name}_Rank'] = result.feature_importance.argsort()[::-1].argsort() + 1
        
        df = pd.DataFrame(data)
        
        # Calculate statistics across models
        importance_cols = [col for col in df.columns if col.endswith('_Importance')]
        df['Mean_Importance'] = df[importance_cols].mean(axis=1)
        df['Std_Importance'] = df[importance_cols].std(axis=1)
        df['Min_Importance'] = df[importance_cols].min(axis=1)
        df['Max_Importance'] = df[importance_cols].max(axis=1)
        
        # Sort by mean importance
        df = df.sort_values('Mean_Importance', key=abs, ascending=False).reset_index(drop=True)
        
        return df
    
    def get_agreement_score(
        self,
        result_a: ExplanationResult,
        result_b: ExplanationResult,
        top_n: int = 10
    ) -> Dict[str, float]:
        """
        Calculate agreement score between two models.
        
        Args:
            result_a: ExplanationResult for model A
            result_b: ExplanationResult for model B
            top_n: Number of top features to consider
            
        Returns:
            Dictionary with agreement metrics
        """
        # Get top N features for each model
        top_a = np.argsort(np.abs(result_a.feature_importance))[-top_n:]
        top_b = np.argsort(np.abs(result_b.feature_importance))[-top_n:]
        
        # Calculate overlap
        overlap = len(set(top_a) & set(top_b))
        overlap_pct = (overlap / top_n) * 100
        
        # Calculate rank correlation (Spearman)
        from scipy.stats import spearmanr
        correlation, p_value = spearmanr(
            np.abs(result_a.feature_importance),
            np.abs(result_b.feature_importance)
        )
        
        return {
            'top_n': top_n,
            'overlap_count': overlap,
            'overlap_percentage': overlap_pct,
            'rank_correlation': correlation,
            'correlation_p_value': p_value
        }
    
    def find_disagreements(
        self,
        result_a: ExplanationResult,
        result_b: ExplanationResult,
        threshold: float = 0.1
    ) -> pd.DataFrame:
        """
        Find features where models disagree significantly.
        
        Args:
            result_a: ExplanationResult for model A
            result_b: ExplanationResult for model B
            threshold: Threshold for significant disagreement (as fraction)
            
        Returns:
            DataFrame with disagreeing features
        """
        comparison = self.compare_two_models(result_a, result_b)
        
        # Features where models disagree significantly
        max_importance = max(
            np.abs(result_a.feature_importance).max(),
            np.abs(result_b.feature_importance).max()
        )
        
        threshold_value = threshold * max_importance
        
        disagreements = comparison[comparison['Abs_Difference'] > threshold_value].copy()
        
        return disagreements
    
    def get_feature_rankings(
        self,
        results: Dict[str, ExplanationResult]
    ) -> pd.DataFrame:
        """
        Get feature rankings across all models.
        
        Args:
            results: Dictionary mapping model names to ExplanationResult objects
            
        Returns:
            DataFrame with feature rankings
        """
        feature_names = list(results.values())[0].feature_names
        
        rankings = {'Feature': feature_names}
        
        for model_name, result in results.items():
            # Get ranking (1 = most important)
            importance_abs = np.abs(result.feature_importance)
            ranking = importance_abs.argsort()[::-1].argsort() + 1
            rankings[f'{model_name}_Rank'] = ranking
        
        df = pd.DataFrame(rankings)
        
        # Add average rank
        rank_cols = [col for col in df.columns if col.endswith('_Rank')]
        df['Avg_Rank'] = df[rank_cols].mean(axis=1)
        df['Std_Rank'] = df[rank_cols].std(axis=1)
        
        # Sort by average rank
        df = df.sort_values('Avg_Rank').reset_index(drop=True)
        
        return df


def analyze_feature_stability(
    results: List[ExplanationResult],
    model_name: str = 'model'
) -> pd.DataFrame:
    """
    Analyze feature importance stability across multiple runs.
    
    Args:
        results: List of ExplanationResult objects from multiple runs
        model_name: Name of the model
        
    Returns:
        DataFrame with stability metrics
    """
    if not results:
        raise ValueError("No results provided")
    
    feature_names = results[0].feature_names
    n_features = len(feature_names)
    n_runs = len(results)
    
    # Collect importance scores
    importance_matrix = np.zeros((n_runs, n_features))
    for i, result in enumerate(results):
        importance_matrix[i, :] = result.feature_importance
    
    # Calculate statistics
    df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Importance': importance_matrix.mean(axis=0),
        'Std_Importance': importance_matrix.std(axis=0),
        'Min_Importance': importance_matrix.min(axis=0),
        'Max_Importance': importance_matrix.max(axis=0),
        'CV': importance_matrix.std(axis=0) / (np.abs(importance_matrix.mean(axis=0)) + 1e-10)
    })
    
    # Add stability score (lower CV = more stable)
    df['Stability_Score'] = 1 / (1 + df['CV'])
    
    # Sort by mean importance
    df = df.sort_values('Mean_Importance', key=abs, ascending=False).reset_index(drop=True)
    
    return df


def get_consensus_features(
    results: Dict[str, ExplanationResult],
    top_n: int = 10,
    min_agreement: int = 2
) -> List[str]:
    """
    Get features that appear in top N for multiple models.
    
    Args:
        results: Dictionary mapping model names to ExplanationResult objects
        top_n: Number of top features to consider per model
        min_agreement: Minimum number of models that must agree
        
    Returns:
        List of consensus feature names
    """
    feature_counts = {}
    
    for model_name, result in results.items():
        # Get top N features
        top_indices = np.argsort(np.abs(result.feature_importance))[-top_n:]
        top_features = [result.feature_names[i] for i in top_indices]
        
        # Count occurrences
        for feature in top_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    # Filter by minimum agreement
    consensus = [
        feature for feature, count in feature_counts.items()
        if count >= min_agreement
    ]
    
    # Sort by count (descending)
    consensus.sort(key=lambda f: feature_counts[f], reverse=True)
    
    return consensus


def create_importance_matrix(
    results: Dict[str, ExplanationResult]
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Create importance matrix for visualization.
    
    Args:
        results: Dictionary mapping model names to ExplanationResult objects
        
    Returns:
        Tuple of (importance_matrix, model_names, feature_names)
    """
    model_names = list(results.keys())
    feature_names = list(results.values())[0].feature_names
    
    n_models = len(model_names)
    n_features = len(feature_names)
    
    importance_matrix = np.zeros((n_models, n_features))
    
    for i, model_name in enumerate(model_names):
        importance_matrix[i, :] = results[model_name].feature_importance
    
    return importance_matrix, model_names, feature_names


def summarize_comparison(
    result_a: ExplanationResult,
    result_b: ExplanationResult,
    model_a_name: Optional[str] = None,
    model_b_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get summary of model comparison.
    
    Args:
        result_a: ExplanationResult for model A
        result_b: ExplanationResult for model B
        model_a_name: Name for model A
        model_b_name: Name for model B
        
    Returns:
        Dictionary with summary statistics
    """
    model_a_name = model_a_name or result_a.model_name
    model_b_name = model_b_name or result_b.model_name
    
    comparator = FeatureComparator()
    comparison = comparator.compare_two_models(result_a, result_b, model_a_name, model_b_name)
    agreement = comparator.get_agreement_score(result_a, result_b, top_n=10)
    
    # Get top features for each
    top_a = result_a.get_top_features(5)['Feature'].tolist()
    top_b = result_b.get_top_features(5)['Feature'].tolist()
    
    # Most different features
    most_different = comparison.head(5)['Feature'].tolist()
    
    summary = {
        'model_a': model_a_name,
        'model_b': model_b_name,
        'agreement': agreement,
        'top_features': {
            model_a_name: top_a,
            model_b_name: top_b,
            'shared': list(set(top_a) & set(top_b))
        },
        'most_different_features': most_different,
        'n_features': len(result_a.feature_names)
    }
    
    return summary