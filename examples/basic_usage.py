"""
Basic Usage Example for MLOps A/B Testing Framework

This example demonstrates the core functionality of the framework.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib

from mlops_ab_testing import ABTestFramework


def create_sample_data():
    """Create sample classification dataset."""
    print("Creating sample dataset...")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create dataframes
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test
    
    # Save test data
    test_df.to_csv('data/raw/test_data.csv', index=False)
    print(f"✓ Saved test data: {len(test_df)} samples")
    
    return train_df, test_df


def train_models(train_df):
    """Train champion and challenger models."""
    print("\nTraining models...")
    
    # Prepare data
    feature_cols = [col for col in train_df.columns if col != 'target']
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    
    # Train champion model (Random Forest)
    print("  Training champion (Random Forest)...")
    champion = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    champion.fit(X_train, y_train)
    joblib.dump(champion, 'models/champion.pkl')
    print(f"  ✓ Champion accuracy: {champion.score(X_train, y_train):.3f}")
    
    # Train challenger model (Gradient Boosting)
    print("  Training challenger (Gradient Boosting)...")
    challenger = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    challenger.fit(X_train, y_train)
    joblib.dump(challenger, 'models/challenger.pkl')
    print(f"  ✓ Challenger accuracy: {challenger.score(X_train, y_train):.3f}")
    
    return champion, challenger


def run_ab_test_simple():
    """Run A/B test with simple API."""
    print("\n" + "="*60)
    print("Running A/B Test (Simple API)")
    print("="*60)
    
    # Initialize framework with simple parameters
    framework = ABTestFramework(
        champion_model='models/champion.pkl',
        challenger_models=['models/challenger.pkl'],
        test_data='data/raw/test_data.csv',
        target_column='target'
    )
    
    # Run test
    results = framework.run_test()
    
    # Display summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Duration: {results.duration:.2f}s")
    print(f"Samples: {results.n_samples}")
    print(f"Models: {', '.join(results.model_names)}")
    
    print("\n" + framework.get_predictions_summary().to_string(index=False))
    
    return results


def run_ab_test_with_config():
    """Run A/B test with configuration file."""
    print("\n" + "="*60)
    print("Running A/B Test (With Config)")
    print("="*60)
    
    # Initialize framework with config
    framework = ABTestFramework(config='configs/config.yaml')
    
    # Run test
    results = framework.run_test()
    
    # Display results
    print("\n" + framework.get_predictions_summary().to_string(index=False))
    
    return results


def main():
    """Main example function."""
    print("="*60)
    print("MLOps A/B Testing Framework - Basic Example")
    print("="*60)
    
    # Create directories
    import os
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Create sample data and train models
    train_df, test_df = create_sample_data()
    champion, challenger = train_models(train_df)
    
    # Step 2: Run A/B test with simple API
    results = run_ab_test_simple()
    
    print("\n" + "="*60)
    print("Example Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Check the test results above")
    print("  2. Run metrics analysis (coming in Step 3)")
    print("  3. Generate visualizations (coming in Step 5)")
    print("  4. Create reports (coming in Step 5)")
    
    return results


if __name__ == '__main__':
    results = main()