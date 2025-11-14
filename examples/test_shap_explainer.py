"""
Test SHAP explainer with real models.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def test_shap_with_tree_model():
    """Test SHAP explainer with a tree-based model."""
    print("="*60)
    print("Testing SHAP with Tree Model")
    print("="*60)
    
    # Check if SHAP is installed
    try:
        import shap
        print("[OK] SHAP is installed")
    except ImportError:
        print("[WARNING] SHAP not installed. Install with: pip install shap")
        print("Skipping SHAP tests...")
        return
    
    from mlops_ab_testing.explainability import SHAPExplainer, explain_model_with_shap
    
    # Create sample data
    print("\n1. Creating sample data...")
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train model
    print("\n2. Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"   Model accuracy: {accuracy:.3f}")
    
    # Create SHAP explainer
    print("\n3. Creating SHAP explainer...")
    explainer = SHAPExplainer(
        model=model,
        feature_names=feature_names,
        explainer_type='tree'
    )
    print(f"   Explainer type: {explainer.explainer_type}")
    
    # Global explanation
    print("\n4. Computing global SHAP explanations...")
    result = explainer.get_explanation_result(
        X=X_test,
        model_name='RandomForest',
        mode='global',
        sample_size=100
    )
    
    print(f"   Computed SHAP values shape: {result.shap_values.shape}")
    print(f"   Feature importance shape: {result.feature_importance.shape}")
    
    # Get top features
    print("\n5. Top 5 most important features:")
    top_features = result.get_top_features(n=5)
    print(top_features.to_string(index=False))
    
    # Local explanation
    print("\n6. Computing local explanation for first sample...")
    local_result = explainer.explain_local(X_test, indices=[0])
    print(f"   Local SHAP values shape: {local_result['shap_values'].shape}")
    
    print("\n   Feature contributions for first prediction:")
    contributions = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Value': local_result['shap_values'][0]
    }).sort_values('SHAP_Value', key=abs, ascending=False)
    print(contributions.head(5).to_string(index=False))
    
    print("\n" + "="*60)
    print("[OK] SHAP Tree Explainer Test Passed!")
    print("="*60)


def test_compare_two_models():
    """Test comparing SHAP explanations between two models."""
    print("\n\n" + "="*60)
    print("Testing SHAP Comparison Between Two Models")
    print("="*60)
    
    try:
        import shap
    except ImportError:
        print("[WARNING] SHAP not installed. Skipping comparison test...")
        return
    
    from mlops_ab_testing.explainability import compare_shap_explanations
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Create data
    print("\n1. Creating sample data...")
    X, y = make_classification(n_samples=300, n_features=8, random_state=42)
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Train two models
    print("\n2. Training two models...")
    model_a = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
    model_b = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
    
    model_a.fit(X, y)
    model_b.fit(X, y)
    
    print(f"   Model A accuracy: {model_a.score(X, y):.3f}")
    print(f"   Model B accuracy: {model_b.score(X, y):.3f}")
    
    # Compare explanations
    print("\n3. Comparing SHAP explanations...")
    comparison = compare_shap_explanations(
        model_a=model_a,
        model_b=model_b,
        X=X,
        feature_names=feature_names,
        model_a_name='RandomForest',
        model_b_name='GradientBoosting',
        sample_size=50
    )
    
    print("\n4. Feature importance comparison:")
    print(comparison['comparison'].head(5).to_string(index=False))
    
    print("\n" + "="*60)
    print("[OK] SHAP Comparison Test Passed!")
    print("="*60)


def main():
    """Run all SHAP tests."""
    print("\n" + "="*70)
    print(" "*20 + "SHAP Explainer Tests")
    print("="*70)
    
    test_shap_with_tree_model()
    test_compare_two_models()
    
    print("\n" + "="*70)
    print("All SHAP Tests Complete!")
    print("="*70)
    print("\nStep 4.2 Complete: SHAP integration ready!")
    print("Next: Step 4.3 will add LIME integration")


if __name__ == '__main__':
    main()