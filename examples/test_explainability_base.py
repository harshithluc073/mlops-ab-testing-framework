"""
Test script for explainability base module.
"""

import numpy as np
from mlops_ab_testing.explainability import (
    ExplainerConfig,
    ExplanationResult,
    get_sample_indices,
    compare_feature_importance
)


def test_explainer_config():
    """Test ExplainerConfig."""
    print("Testing ExplainerConfig...")
    
    config = ExplainerConfig(
        method='shap',
        sample_size=100,
        n_samples=1000
    )
    
    print(f"  Method: {config.method}")
    print(f"  Sample size: {config.sample_size}")
    print(f"  Config dict: {config.to_dict()}")
    print("  [OK] ExplainerConfig works!")


def test_explanation_result():
    """Test ExplanationResult."""
    print("\nTesting ExplanationResult...")
    
    feature_names = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
    feature_importance = np.array([0.5, 0.3, 0.15, 0.05])
    
    result = ExplanationResult(
        method='shap',
        model_name='test_model',
        feature_names=feature_names,
        feature_importance=feature_importance
    )
    
    print(f"  Method: {result.method}")
    print(f"  Model: {result.model_name}")
    
    # Get top features
    top_features = result.get_top_features(n=2)
    print(f"\n  Top 2 features:")
    print(top_features.to_string(index=False))
    
    print("\n  [OK] ExplanationResult works!")


def test_sample_indices():
    """Test get_sample_indices."""
    print("\nTesting get_sample_indices...")
    
    indices = get_sample_indices(
        n_total=100,
        sample_size=10,
        strategy='random'
    )
    
    print(f"  Selected {len(indices)} indices from 100")
    print(f"  Indices: {indices[:5]}...")
    print("  [OK] get_sample_indices works!")


def test_compare_importance():
    """Test compare_feature_importance."""
    print("\nTesting compare_feature_importance...")
    
    feature_names = ['age', 'income', 'score', 'credit']
    importance_a = np.array([0.4, 0.3, 0.2, 0.1])
    importance_b = np.array([0.3, 0.4, 0.2, 0.1])
    
    comparison = compare_feature_importance(
        model_a_importance=importance_a,
        model_b_importance=importance_b,
        feature_names=feature_names,
        model_a_name='Champion',
        model_b_name='Challenger'
    )
    
    print("\n  Feature Importance Comparison:")
    print(comparison.to_string(index=False))
    print("\n  [OK] compare_feature_importance works!")


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Explainability Base Module")
    print("="*60)
    
    test_explainer_config()
    test_explanation_result()
    test_sample_indices()
    test_compare_importance()
    
    print("\n" + "="*60)
    print("All Tests Passed!")
    print("="*60)
    print("\nStep 4.1 Complete: Base explainer infrastructure ready!")
    print("Next: Step 4.2 will add SHAP integration")


if __name__ == '__main__':
    main()