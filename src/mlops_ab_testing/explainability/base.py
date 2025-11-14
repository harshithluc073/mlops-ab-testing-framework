"""
Base explainer class and utilities for model explainability.

This module provides the foundation for all explainers (SHAP, LIME, etc.)
with common interfaces and utilities.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseExplainer(ABC):
    """Abstract base class for all explainers."""
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        model_type: str = 'auto'
    ):
        """
        Initialize base explainer.
        
        Args:
            model: The model to explain
            feature_names: Names of features
            model_type: Type of model ('tree', 'linear', 'deep', 'auto')
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        
        if model_type == 'auto':
            self.model_type = self._detect_model_type()
        
        logger.info(f"Initialized {self.__class__.__name__} for {self.model_type} model")
    
    def _detect_model_type(self) -> str:
        """
        Detect model type from model class.
        
        Returns:
            Model type string
        """
        model_class = type(self.model).__name__.lower()
        
        # Tree-based models
        tree_keywords = ['tree', 'forest', 'boost', 'gbm', 'xgb', 'lgb', 'catboost']
        if any(keyword in model_class for keyword in tree_keywords):
            return 'tree'
        
        # Linear models
        linear_keywords = ['linear', 'logistic', 'ridge', 'lasso', 'elasticnet']
        if any(keyword in model_class for keyword in linear_keywords):
            return 'linear'
        
        # Deep learning models
        deep_keywords = ['neural', 'keras', 'pytorch', 'tensorflow']
        if any(keyword in model_class for keyword in deep_keywords):
            return 'deep'
        
        # Default to kernel for unknown models
        return 'kernel'
    
    @abstractmethod
    def explain_global(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Generate global explanations for the model.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary with global explanations
        """
        pass
    
    @abstractmethod
    def explain_local(self, X: np.ndarray, indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Generate local explanations for specific instances.
        
        Args:
            X: Input data
            indices: Indices of instances to explain (None = all)
            
        Returns:
            Dictionary with local explanations
        """
        pass
    
    def explain(self, X: np.ndarray, mode: str = 'global') -> Dict[str, Any]:
        """
        Generate explanations.
        
        Args:
            X: Input data
            mode: Explanation mode ('global' or 'local')
            
        Returns:
            Dictionary with explanations
        """
        if mode == 'global':
            return self.explain_global(X)
        elif mode == 'local':
            return self.explain_local(X)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'global' or 'local'")
    
    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        """
        Validate and convert input data.
        
        Args:
            X: Input data
            
        Returns:
            Validated numpy array
        """
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X = X.values
        
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        return X
    
    def _get_feature_names(self, X: Union[np.ndarray, pd.DataFrame]) -> List[str]:
        """
        Get feature names from data or generate default names.
        
        Args:
            X: Input data
            
        Returns:
            List of feature names
        """
        if self.feature_names is not None:
            return self.feature_names
        
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        
        # Generate default feature names
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        return [f'feature_{i}' for i in range(n_features)]


class ExplainerConfig:
    """Configuration for explainers."""
    
    def __init__(
        self,
        method: str = 'shap',
        sample_size: int = 100,
        n_samples: int = 1000,
        random_seed: int = 42,
        **kwargs
    ):
        """
        Initialize explainer configuration.
        
        Args:
            method: Explainer method ('shap', 'lime')
            sample_size: Number of samples for explanation
            n_samples: Number of samples for SHAP/LIME (for sampling-based methods)
            random_seed: Random seed
            **kwargs: Additional method-specific parameters
        """
        self.method = method
        self.sample_size = sample_size
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.kwargs = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'method': self.method,
            'sample_size': self.sample_size,
            'n_samples': self.n_samples,
            'random_seed': self.random_seed,
            **self.kwargs
        }


class ExplanationResult:
    """Container for explanation results."""
    
    def __init__(
        self,
        method: str,
        model_name: str,
        feature_names: List[str],
        feature_importance: Optional[np.ndarray] = None,
        shap_values: Optional[np.ndarray] = None,
        base_values: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize explanation result.
        
        Args:
            method: Explanation method used
            model_name: Name of the model
            feature_names: Names of features
            feature_importance: Global feature importance scores
            shap_values: SHAP values (if applicable)
            base_values: Base/expected values (if applicable)
            data: Original data
            metadata: Additional metadata
        """
        self.method = method
        self.model_name = model_name
        self.feature_names = feature_names
        self.feature_importance = feature_importance
        self.shap_values = shap_values
        self.base_values = base_values
        self.data = data
        self.metadata = metadata or {}
    
    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features
            
        Returns:
            DataFrame with top features and their importance
        """
        if self.feature_importance is None:
            raise ValueError("No feature importance available")
        
        # Ensure feature_importance is 1D
        feature_importance = self.feature_importance
        if feature_importance.ndim > 1:
            feature_importance = feature_importance.flatten()
        
        # Get absolute importance
        abs_importance = np.abs(feature_importance)
        
        # Get top N indices
        top_indices = np.argsort(abs_importance)[-n:][::-1]
        
        # Convert to Python list for safe indexing
        top_indices_list = [int(idx) for idx in top_indices]
        
        df = pd.DataFrame({
            'Feature': [self.feature_names[i] for i in top_indices_list],
            'Importance': [feature_importance[i] for i in top_indices_list],
            'Abs_Importance': [abs_importance[i] for i in top_indices_list]
        })
        
        return df
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = {
            'method': self.method,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'metadata': self.metadata
        }
        
        if self.feature_importance is not None:
            result['feature_importance'] = self.feature_importance.tolist()
        
        if self.shap_values is not None:
            result['shap_values'] = self.shap_values.tolist()
        
        if self.base_values is not None:
            result['base_values'] = self.base_values.tolist()
        
        return result


def get_sample_indices(
    n_total: int,
    sample_size: int,
    random_seed: int = 42,
    strategy: str = 'random'
) -> np.ndarray:
    """
    Get sample indices for explanation.
    
    Args:
        n_total: Total number of samples
        sample_size: Number of samples to select
        random_seed: Random seed
        strategy: Sampling strategy ('random', 'first', 'last')
        
    Returns:
        Array of selected indices
    """
    if sample_size >= n_total:
        return np.arange(n_total)
    
    if strategy == 'random':
        rng = np.random.RandomState(random_seed)
        return rng.choice(n_total, size=sample_size, replace=False)
    elif strategy == 'first':
        return np.arange(sample_size)
    elif strategy == 'last':
        return np.arange(n_total - sample_size, n_total)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def aggregate_explanations(
    explanations: List[ExplanationResult]
) -> pd.DataFrame:
    """
    Aggregate multiple explanation results.
    
    Args:
        explanations: List of ExplanationResult objects
        
    Returns:
        DataFrame with aggregated importance scores
    """
    if not explanations:
        return pd.DataFrame()
    
    # Collect all feature importance scores
    importance_dict = {}
    
    for exp in explanations:
        if exp.feature_importance is not None:
            for fname, importance in zip(exp.feature_names, exp.feature_importance):
                if fname not in importance_dict:
                    importance_dict[fname] = []
                importance_dict[fname].append(importance)
    
    # Calculate statistics
    results = []
    for fname, values in importance_dict.items():
        results.append({
            'Feature': fname,
            'Mean_Importance': np.mean(values),
            'Std_Importance': np.std(values),
            'Min_Importance': np.min(values),
            'Max_Importance': np.max(values),
            'N_Models': len(values)
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('Mean_Importance', ascending=False, key=abs)
    
    return df


def compare_feature_importance(
    model_a_importance: np.ndarray,
    model_b_importance: np.ndarray,
    feature_names: List[str],
    model_a_name: str = 'Model A',
    model_b_name: str = 'Model B'
) -> pd.DataFrame:
    """
    Compare feature importance between two models.
    
    Args:
        model_a_importance: Feature importance for model A
        model_b_importance: Feature importance for model B
        feature_names: Names of features
        model_a_name: Name of model A
        model_b_name: Name of model B
        
    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame({
        'Feature': feature_names,
        f'{model_a_name}_Importance': model_a_importance,
        f'{model_b_name}_Importance': model_b_importance
    })
    
    # Calculate difference
    df['Difference'] = df[f'{model_b_name}_Importance'] - df[f'{model_a_name}_Importance']
    df['Abs_Difference'] = np.abs(df['Difference'])
    
    # Sort by absolute difference
    df = df.sort_values('Abs_Difference', ascending=False)
    
    return df