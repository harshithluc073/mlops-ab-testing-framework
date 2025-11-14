"""
LIME (Local Interpretable Model-agnostic Explanations) integration.

This module provides LIME-based explainability for any model type.
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


class LIMEExplainer(BaseExplainer):
    """LIME-based explainer for machine learning models."""
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        model_type: str = 'auto',
        mode: str = 'tabular'
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: The model to explain
            feature_names: Names of features
            model_type: Type of model (not used for LIME, model-agnostic)
            mode: LIME mode ('tabular' or 'text')
        """
        super().__init__(model, feature_names, model_type)
        
        self.mode = mode
        self.explainer = None
        
        logger.info(f"Initialized LIME explainer in {mode} mode")
    
    def _initialize_explainer(self, X: np.ndarray, training_data: Optional[np.ndarray] = None):
        """
        Initialize LIME explainer with training data.
        
        Args:
            X: Data to explain
            training_data: Training data for LIME (uses X if not provided)
        """
        try:
            import lime
            import lime.lime_tabular
        except ImportError:
            raise ImportError(
                "LIME not installed. Install with: pip install lime"
            )
        
        if self.explainer is not None:
            return  # Already initialized
        
        logger.info("Initializing LIME explainer...")
        
        # Use X as training data if not provided
        if training_data is None:
            training_data = X
        
        # Get feature names
        feature_names = self._get_feature_names(X)
        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()
        
        # Initialize LIME tabular explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            mode='classification',  # or 'regression'
            random_state=42
        )
        
        logger.info("LIME explainer initialized")
    
    def explain_global(
        self,
        X: np.ndarray,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate global LIME explanations by aggregating local explanations.
        
        Args:
            X: Input data
            sample_size: Number of samples to use (None = all)
            
        Returns:
            Dictionary with aggregated explanations
        """
        X = self._validate_data(X)
        feature_names = self._get_feature_names(X)
        
        # Ensure feature_names is a list
        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()
        
        # Initialize explainer
        self._initialize_explainer(X)
        
        # Sample data if needed
        if sample_size and sample_size < len(X):
            indices = get_sample_indices(len(X), sample_size)
            X_sample = X[indices]
        else:
            X_sample = X
        
        logger.info(f"Computing LIME explanations for {len(X_sample)} samples...")
        
        # Collect feature importance from multiple instances
        all_importances = []
        
        # Limit to avoid too many explanations
        max_samples = min(len(X_sample), 50)
        sample_indices = np.random.choice(len(X_sample), max_samples, replace=False)
        
        for idx in sample_indices:
            try:
                # Get explanation for this instance
                exp = self.explainer.explain_instance(
                    X_sample[idx],
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    num_features=len(feature_names)
                )
                
                # Extract feature importance
                importance_dict = dict(exp.as_list())
                
                # Convert feature names to indices and get values
                instance_importance = np.zeros(len(feature_names))
                for feature_desc, value in importance_dict.items():
                    # Feature description is like "feature_0 <= 0.5"
                    # Extract feature name
                    feature_name = feature_desc.split()[0]
                    if feature_name in feature_names:
                        idx_feat = feature_names.index(feature_name)
                        instance_importance[idx_feat] = value
                
                all_importances.append(instance_importance)
                
            except Exception as e:
                logger.warning(f"Failed to explain instance {idx}: {e}")
                continue
        
        # Aggregate importance scores
        if all_importances:
            mean_importance = np.mean(all_importances, axis=0)
        else:
            mean_importance = np.zeros(len(feature_names))
        
        logger.info(f"Aggregated {len(all_importances)} local explanations")
        
        return {
            'feature_importance': mean_importance,
            'feature_names': feature_names,
            'data': X_sample,
            'n_samples': len(X_sample),
            'n_explained': len(all_importances)
        }
    
    def explain_local(
        self,
        X: np.ndarray,
        indices: Optional[List[int]] = None,
        num_features: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate local LIME explanations for specific instances.
        
        Args:
            X: Input data
            indices: Indices of instances to explain (None = all)
            num_features: Number of features to show in explanation
            
        Returns:
            Dictionary with local explanations
        """
        X = self._validate_data(X)
        feature_names = self._get_feature_names(X)
        
        # Ensure feature_names is a list
        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()
        
        # Initialize explainer
        self._initialize_explainer(X)
        
        # Select instances
        if indices is not None:
            X_explain = X[indices]
        else:
            X_explain = X
        
        if num_features is None:
            num_features = len(feature_names)
        
        logger.info(f"Computing local LIME explanations for {len(X_explain)} instances...")
        
        # Get explanations for each instance
        explanations = []
        importance_matrix = []
        
        for i in range(len(X_explain)):
            try:
                exp = self.explainer.explain_instance(
                    X_explain[i],
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    num_features=num_features
                )
                
                explanations.append(exp)
                
                # Convert to importance array
                importance_dict = dict(exp.as_list())
                instance_importance = np.zeros(len(feature_names))
                
                for feature_desc, value in importance_dict.items():
                    feature_name = feature_desc.split()[0]
                    if feature_name in feature_names:
                        idx_feat = feature_names.index(feature_name)
                        instance_importance[idx_feat] = value
                
                importance_matrix.append(instance_importance)
                
            except Exception as e:
                logger.warning(f"Failed to explain instance {i}: {e}")
                importance_matrix.append(np.zeros(len(feature_names)))
        
        importance_matrix = np.array(importance_matrix)
        
        return {
            'explanations': explanations,
            'importance_matrix': importance_matrix,
            'feature_names': feature_names,
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
        
        # Get feature importance - handle both global and local modes
        feature_importance = explanation.get('feature_importance')
        if feature_importance is None:
            # For local mode, aggregate importance matrix
            importance_matrix = explanation.get('importance_matrix')
            if importance_matrix is not None:
                feature_importance = np.mean(np.abs(importance_matrix), axis=0)
        
        result = ExplanationResult(
            method='lime',
            model_name=model_name,
            feature_names=explanation['feature_names'],
            feature_importance=feature_importance,
            data=explanation['data'],
            metadata={
                'mode': self.mode,
                'n_samples': explanation.get('n_samples', explanation.get('n_instances')),
                'n_explained': explanation.get('n_explained', explanation.get('n_instances'))
            }
        )
        
        return result


def explain_model_with_lime(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    model_name: str = 'model',
    sample_size: int = 50
) -> ExplanationResult:
    """
    Explain a model using LIME (convenience function).
    
    Args:
        model: The model to explain
        X: Input data
        feature_names: Names of features
        model_name: Name of the model
        sample_size: Number of samples to explain
        
    Returns:
        ExplanationResult object
    """
    explainer = LIMEExplainer(
        model=model,
        feature_names=feature_names
    )
    
    result = explainer.get_explanation_result(
        X=X,
        model_name=model_name,
        mode='global',
        sample_size=sample_size
    )
    
    return result


def compare_lime_explanations(
    model_a: Any,
    model_b: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    model_a_name: str = 'Model A',
    model_b_name: str = 'Model B',
    sample_size: int = 50
) -> Dict[str, Any]:
    """
    Compare LIME explanations between two models.
    
    Args:
        model_a: First model
        model_b: Second model
        X: Input data
        feature_names: Names of features
        model_a_name: Name of first model
        model_b_name: Name of second model
        sample_size: Number of samples to explain
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing LIME explanations: {model_a_name} vs {model_b_name}")
    
    # Explain model A
    result_a = explain_model_with_lime(
        model=model_a,
        X=X,
        feature_names=feature_names,
        model_name=model_a_name,
        sample_size=sample_size
    )
    
    # Explain model B
    result_b = explain_model_with_lime(
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