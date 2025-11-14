"""
Unit tests for core framework.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from mlops_ab_testing.core.framework import ABTestFramework
from mlops_ab_testing.core.config import ExperimentConfig, TrafficConfig, ModelConfig


def test_config_creation():
    """Test configuration creation."""
    config = ExperimentConfig(
        experiment=ModelConfig(name="test", description="Test experiment"),
        traffic=TrafficConfig(strategy="random", split_ratio=[0.5, 0.5])
    )
    
    assert config.experiment.name == "test"
    assert config.traffic.strategy == "random"
    assert sum(config.traffic.split_ratio) == 1.0


def test_framework_initialization(tmp_path):
    """Test framework initialization."""
    # Create test data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    
    # Save test data
    data_path = tmp_path / "test_data.csv"
    df.to_csv(data_path, index=False)
    
    # Create and save model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    model_path = tmp_path / "model.joblib"
    import joblib
    joblib.dump(model, model_path)
    
    # Initialize framework
    framework = ABTestFramework(
        champion_model=str(model_path),
        challenger_models=[str(model_path)],
        test_data=str(data_path),
        target_column='target'
    )
    
    assert len(framework.models) == 2
    assert framework.test_data is not None


def test_run_test(tmp_path):
    """Test running A/B test."""
    # Create test data
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    
    data_path = tmp_path / "test_data.csv"
    df.to_csv(data_path, index=False)
    
    # Create model
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    
    model_path = tmp_path / "model.joblib"
    import joblib
    joblib.dump(model, model_path)
    
    # Run test
    framework = ABTestFramework(
        champion_model=str(model_path),
        challenger_models=[str(model_path)],
        test_data=str(data_path),
        target_column='target'
    )
    
    results = framework.run_test()
    
    assert results is not None
    assert len(results.predictions) == 2
    assert results.ground_truth is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])