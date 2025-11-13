# core module
"""
Core module for MLOps A/B Testing Framework.

This module contains the main framework classes and utilities.
"""

from mlops_ab_testing.core.config import (
    ConfigManager,
    ABTestConfig,
    ModelConfig,
    ExperimentConfig,
    DataConfig,
    RoutingConfig,
    load_default_config
)
from mlops_ab_testing.core.model_loader import (
    ModelLoader,
    ModelWrapper,
    save_model
)
from mlops_ab_testing.core.router import (
    TrafficRouter,
    EqualRouter,
    StratifiedRouter,
    WeightedRouter,
    SequentialRouter,
    create_router_from_config
)
from mlops_ab_testing.core.framework import (
    ABTestFramework,
    ABTestResults,
    PredictionResult
)

__all__ = [
    # Config
    'ConfigManager',
    'ABTestConfig',
    'ModelConfig',
    'ExperimentConfig',
    'DataConfig',
    'RoutingConfig',
    'load_default_config',
    # Model Loading
    'ModelLoader',
    'ModelWrapper',
    'save_model',
    # Routing
    'TrafficRouter',
    'EqualRouter',
    'StratifiedRouter',
    'WeightedRouter',
    'SequentialRouter',
    'create_router_from_config',
    # Framework
    'ABTestFramework',
    'ABTestResults',
    'PredictionResult',
]