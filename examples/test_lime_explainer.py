"""
Test LIME explainer with real models.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def test_lime_explainer():
    """Test LIME explainer."""
    print("="*60)
    print("Testing LIME Explainer")
    print("="*60)
    
    # Check if LIME is installed
    try:
        import lime
        print("[OK] LIME is installed")
    except ImportError:
        print("[WARNING] LIME not installed. Install with: pip install lime")
        print("Skipping LIME tests...")
        return
    
    from mlops_ab_testing.explainability import LIMEExplainer, explain_model_with_lime
    
    # Create sample data
    print("\n1. Creating sample data...")
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=6,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train model
    print("\n2. Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"   Model accuracy: {accuracy:.3f}")
    
    # Quick explanation
    print("\n3. Quick LIME explanation...")
    result = explain_model_with_lime(
        model=model,
        X=X_test,
        feature_names=feature_names,
        sample_size=20
    )
    
    print(f"   Feature importance computed for {result.metadata['n_explained']} instances")
    
    # Get top features
    print("\n4. Top 5 most important features:")
    top_features = result.get_top_features(n=5)
    print(top_features.to_string(index=False))
    
    # Local explanation
    print("\n5. Local explanation for first instance...")
    explainer = LIMEExplainer(model=model, feature_names=feature_names)
    local = explainer.explain_local(X_test, indices=[0], num_features=5)
    
    print(f"   Explained {len(local['explanations'])} instances")
    print("   Top 5 feature contributions:")
    importance = local['importance_matrix'][0]
    indices = np.argsort(np.abs(importance))[-5:][::-1]
    for idx in indices:
        print(f"      {feature_names[idx]}: {importance[idx]:+.3f}")
    
    print("\n" + "="*60)
    print("[OK] LIME Explainer Test Passed!")
    print("="*60)


def main():
    """Run LIME tests."""
    print("\n" + "="*70)
    print(" "*25 + "LIME Tests")
    print("="*70)
    
    test_lime_explainer()
    
    print("\n" + "="*70)
    print("LIME Tests Complete!")
    print("="*70)
    print("\nStep 4.3 Complete: LIME integration ready!")
    print("Next: Step 4.4 will add feature importance comparison")


if __name__ == '__main__':
    main()