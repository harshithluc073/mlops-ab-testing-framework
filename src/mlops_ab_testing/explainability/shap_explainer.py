"""
SHAP (SHapley Additive exPlanations) integration.

This module provides SHAP-based explainability for models including
TreeExplainer for tree-based models and KernelExplainer for any model.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
import logging

from mlops_ab_testing.explainability.base import (
    BaseExplainer,
    ExplanationResult,
    get_sample_indices
)

logger = logging.getLogger(__name__)


class SHAPExplainer(BaseExplainer):
    """SHAP-based explainer for machine learning models."""
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        model_type: str = 'auto',
        explainer_type: str = 'auto'
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: The model to explain
            feature_names: Names of features
            model_type: Type of model ('tree', 'linear', 'deep', 'auto')
            explainer_type: Type of SHAP explainer ('tree', 'kernel', 'linear', 'auto')
        """
        super().__init__(model, feature_names, model_type)
        
        self.explainer_type = explainer_type
        if explainer_type == 'auto':
            self.explainer_type = self._select_explainer_type()
        
        self.explainer = None
        self.expected_value = None
        
        logger.info(f"Using SHAP {self.explainer_type} explainer")
    
    def _select_explainer_type(self) -> str:
        """
        Select appropriate SHAP explainer based on model type.
        
        Returns:
            Explainer type string
        """
        if self.model_type == 'tree':
            return 'tree'
        elif self.model_type == 'linear':
            return 'linear'
        elif self.model_type == 'deep':
            return 'deep'
        else:
            return 'kernel'
    
    def _initialize_explainer(self, X: np.ndarray):
        """
        Initialize SHAP explainer with background data.
        
        Args:
            X: Background data for explainer
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP not installed. Install with: pip install shap"
            )
        
        if self.explainer is not None:
            return  # Already initialized
        
        logger.info(f"Initializing SHAP {self.explainer_type} explainer...")
        
        try:
            if self.explainer_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model)
                
            elif self.explainer_type == 'kernel':
                # Use a sample as background for kernel explainer
                background_size = min(100, len(X))
                background = shap.sample(X, background_size)
                self.explainer = shap.KernelExplainer(self.model.predict, background)
                
            elif self.explainer_type == 'linear':
                self.explainer = shap.LinearExplainer(self.model, X)
                
            elif self.explainer_type == 'deep':
                # For deep learning models
                background_size = min(100, len(X))
                background = X[:background_size]
                self.explainer = shap.DeepExplainer(self.model, background)
                
            else:
                raise ValueError(f"Unknown explainer type: {self.explainer_type}")
            
            # Get expected value
            if hasattr(self.explainer, 'expected_value'):
                self.expected_value = self.explainer.expected_value
            
            logger.info("SHAP explainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            # Fallback to kernel explainer
            logger.info("Falling back to KernelExplainer...")
            background_size = min(100, len(X))
            background = shap.sample(X, background_size) if len(X) > background_size else X
            self.explainer = shap.KernelExplainer(self.model.predict, background)
            self.explainer_type = 'kernel'
    
    def explain_global(
        self,
        X: np.ndarray,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate global SHAP explanations.
        
        Args:
            X: Input data
            sample_size: Number of samples to use (None = all)
            
        Returns:
            Dictionary with global explanations
        """
        X = self._validate_data(X)
        feature_names = self._get_feature_names(X)
        
        # Initialize explainer if needed
        self._initialize_explainer(X)
        
        # Sample data if needed
        if sample_size and sample_size < len(X):
            indices = get_sample_indices(len(X), sample_size)
            X_sample = X[indices]
        else:
            X_sample = X
        
        logger.info(f"Computing SHAP values for {len(X_sample)} samples...")
        
        # Calculate SHAP values
        try:
            shap_values = self.explainer.shap_values(X_sample)
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            raise
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            # For multi-class, take the first class or average
            shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
        
        # Calculate mean absolute SHAP values (global importance)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Sort features by importance
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        
        logger.info(f"Top 5 features: {[feature_names[i] for i in sorted_idx[:5]]}")
        
        return {
            'shap_values': shap_values,
            'feature_importance': mean_abs_shap,
            'feature_names': feature_names,
            'expected_value': self.expected_value,
            'data': X_sample,
            'n_samples': len(X_sample)
        }
    
    def explain_local(
        self,
        X: np.ndarray,
        indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Generate local SHAP explanations for specific instances.
        
        Args:
            X: Input data
            indices: Indices of instances to explain (None = all)
            
        Returns:
            Dictionary with local explanations
        """
        X = self._validate_data(X)
        feature_names = self._get_feature_names(X)
        
        # Initialize explainer if needed
        self._initialize_explainer(X)
        
        # Select instances
        if indices is not None:
            X_explain = X[indices]
        else:
            X_explain = X
        
        logger.info(f"Computing local SHAP explanations for {len(X_explain)} instances...")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_explain)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        return {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'expected_value': self.expected_value,
            'data': X_explain,
            'indices': indices,
            'n_instances': len(X_explain)
        }
    
    def get_explanation_result(
        self,
        X: np.ndarray,
        model_name: str,
        mode: str = 'global',
        sample_size: Optional[int] = None
    ) -> ExplanationResult:
        """
        Get explanation result in standardized format.
        
        Args:
            X: Input data
            model_name: Name of the model
            mode: Explanation mode ('global' or 'local')
            sample_size: Number of samples for global explanation
            
        Returns:
            ExplanationResult object
        """
        if mode == 'global':
            explanation = self.explain_global(X, sample_size)
        else:
            explanation = self.explain_local(X)
        
        result = ExplanationResult(
            method='shap',
            model_name=model_name,
            feature_names=explanation['feature_names'],
            feature_importance=explanation.get('feature_importance'),
            shap_values=explanation['shap_values'],
            base_values=np.full(len(X), explanation.get('expected_value', 0)) 
                if explanation.get('expected_value') is not None else None,
            data=explanation['data'],
            metadata={
                'explainer_type': self.explainer_type,
                'n_samples': explanation.get('n_samples', explanation.get('n_instances'))
            }
        )
        
        return result


def explain_model_with_shap(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    model_name: str = 'model',
    sample_size: int = 100,
    explainer_type: str = 'auto'
) -> ExplanationResult:
    """
    Explain a model using SHAP (convenience function).
    
    Args:
        model: The model to explain
        X: Input data
        feature_names: Names of features
        model_name: Name of the model
        sample_size: Number of samples to use
        explainer_type: Type of SHAP explainer
        
    Returns:
        ExplanationResult object
    """
    explainer = SHAPExplainer(
        model=model,
        feature_names=feature_names,
        explainer_type=explainer_type
    )
    
    result = explainer.get_explanation_result(
        X=X,
        model_name=model_name,
        mode='global',
        sample_size=sample_size
    )
    
    return result


def compare_shap_explanations(
    model_a: Any,
    model_b: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    model_a_name: str = 'Model A',
    model_b_name: str = 'Model B',
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Compare SHAP explanations between two models.
    
    Args:
        model_a: First model
        model_b: Second model
        X: Input data
        feature_names: Names of features
        model_a_name: Name of first model
        model_b_name: Name of second model
        sample_size: Number of samples to use
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing SHAP explanations: {model_a_name} vs {model_b_name}")
    
    # Explain model A
    result_a = explain_model_with_shap(
        model=model_a,
        X=X,
        feature_names=feature_names,
        model_name=model_a_name,
        sample_size=sample_size
    )
    
    # Explain model B
    result_b = explain_model_with_shap(
        model=model_b,
        X=X,
        feature_names=feature_names,
        model_name=model_b_name,
        sample_size=sample_size
    )
    
    # Compare feature importance
    from mlops_ab_testing.explainability import compare_feature_importance
    
    comparison = compare_feature_importance(
        model_a_importance=result_a.feature_importance,
        model_b_importance=result_b.feature_importance,
        feature_names=result_a.feature_names,
        model_a_name=model_a_name,
        model_b_name=model_b_name
    )
    
    return {
        'model_a': result_a,
        'model_b': result_b,
        'comparison': comparison
    }