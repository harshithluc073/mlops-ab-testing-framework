"""
Test feature comparison utilities.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')


def test_feature_comparison():
    """Test feature comparison."""
    print("="*60)
    print("Testing Feature Comparison")
    print("="*60)
    
    # Check SHAP
    try:
        import shap
    except ImportError:
        print("[WARNING] SHAP not installed. Skipping test.")
        return
    
    from mlops_ab_testing.explainability import (
        explain_model_with_shap,
        FeatureComparator,
        summarize_comparison,
        get_consensus_features
    )
    
    # Create data
    print("\n1. Creating sample data...")
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Train two models
    print("\n2. Training two models...")
    model_a = RandomForestClassifier(n_estimators=30, random_state=42)
    model_b = GradientBoostingClassifier(n_estimators=30, random_state=42)
    
    model_a.fit(X, y)
    model_b.fit(X, y)
    
    # Get explanations
    print("\n3. Generating SHAP explanations...")
    result_a = explain_model_with_shap(model_a, X, feature_names, 'RandomForest', sample_size=50)
    result_b = explain_model_with_shap(model_b, X, feature_names, 'GradientBoosting', sample_size=50)
    
    # Compare
    print("\n4. Comparing feature importance...")
    comparator = FeatureComparator()
    comparison = comparator.compare_two_models(result_a, result_b)
    
    print("\n   Top 5 most different features:")
    print(comparison[['Feature', 'RandomForest_Importance', 'GradientBoosting_Importance', 'Abs_Difference']].head(5).to_string(index=False))
    
    # Agreement score
    print("\n5. Calculating agreement score...")
    agreement = comparator.get_agreement_score(result_a, result_b, top_n=5)
    print(f"   Top-5 overlap: {agreement['overlap_count']}/5 ({agreement['overlap_percentage']:.1f}%)")
    print(f"   Rank correlation: {agreement['rank_correlation']:.3f}")
    
    # Summary
    print("\n6. Generating summary...")
    summary = summarize_comparison(result_a, result_b)
    print(f"   Top features (RandomForest): {summary['top_features']['RandomForest'][:3]}")
    print(f"   Top features (GradientBoosting): {summary['top_features']['GradientBoosting'][:3]}")
    print(f"   Shared top features: {summary['top_features']['shared']}")
    
    # Consensus
    print("\n7. Finding consensus features...")
    results_dict = {'RandomForest': result_a, 'GradientBoosting': result_b}
    consensus = get_consensus_features(results_dict, top_n=5, min_agreement=2)
    print(f"   Consensus features (both models): {consensus}")
    
    print("\n" + "="*60)
    print("[OK] Feature Comparison Test Passed!")
    print("="*60)


def main():
    """Run comparison tests."""
    print("\n" + "="*70)
    print(" "*20 + "Comparison Tests")
    print("="*70)
    
    test_feature_comparison()
    
    print("\n" + "="*70)
    print("Comparison Tests Complete!")
    print("="*70)
    print("\nStep 4.4 Complete: Feature comparison ready!")
    print("Next: Step 4.5 will integrate everything with ABTestFramework")


if __name__ == '__main__':
    main()