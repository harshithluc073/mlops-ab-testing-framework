"""
Configuration management for MLOps A/B Testing Framework.

This module handles loading, validating, and managing configuration
for A/B test experiments.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Configuration for a single model."""
    
    path: str = Field(..., description="Path to the model file")
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    
    @validator('path')
    def validate_path(cls, v: str) -> str:
        """Validate that the model path exists."""
        if not os.path.exists(v):
            raise ValueError(f"Model file not found: {v}")
        return v


class ExperimentConfig(BaseModel):
    """Configuration for the experiment."""
    
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    tags: List[str] = Field(default_factory=list, description="Experiment tags")
    random_seed: int = Field(42, description="Random seed for reproducibility")


class DataConfig(BaseModel):
    """Configuration for test data."""
    
    test_data_path: str = Field(..., description="Path to test dataset")
    target_column: str = Field(..., description="Name of target column")
    feature_columns: Optional[List[str]] = Field(None, description="Feature column names")
    stratify_column: Optional[str] = Field(None, description="Column for stratification")
    sample_size: Optional[int] = Field(None, description="Number of samples to use")
    
    @validator('test_data_path')
    def validate_data_path(cls, v: str) -> str:
        """Validate that the data file exists."""
        if not os.path.exists(v):
            raise ValueError(f"Data file not found: {v}")
        return v


class RoutingConfig(BaseModel):
    """Configuration for traffic routing."""
    
    strategy: str = Field("equal", description="Routing strategy: equal, stratified, weighted, custom")
    traffic_split: Dict[str, float] = Field(..., description="Traffic split per model")
    stratify_by: Optional[str] = Field(None, description="Column for stratification")
    custom_router: Optional[str] = Field(None, description="Path to custom router function")
    
    @validator('strategy')
    def validate_strategy(cls, v: str) -> str:
        """Validate routing strategy."""
        valid_strategies = ['equal', 'stratified', 'weighted', 'custom']
        if v not in valid_strategies:
            raise ValueError(f"Invalid strategy: {v}. Must be one of {valid_strategies}")
        return v
    
    @validator('traffic_split')
    def validate_traffic_split(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that traffic split sums to 1.0."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Traffic split must sum to 1.0, got {total}")
        return v


class MetricsConfig(BaseModel):
    """Configuration for metrics calculation."""
    
    classification: List[str] = Field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
        description="Classification metrics to calculate"
    )
    regression: List[str] = Field(
        default_factory=lambda: ["mse", "rmse", "mae", "r2_score"],
        description="Regression metrics to calculate"
    )
    custom: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Custom metrics configuration"
    )


class StatisticalTestsConfig(BaseModel):
    """Configuration for statistical tests."""
    
    alpha: float = Field(0.05, description="Significance level", ge=0, le=1)
    method: str = Field("both", description="Method: classical, bayesian, or both")
    
    @validator('method')
    def validate_method(cls, v: str) -> str:
        """Validate statistical method."""
        valid_methods = ['classical', 'bayesian', 'both']
        if v not in valid_methods:
            raise ValueError(f"Invalid method: {v}. Must be one of {valid_methods}")
        return v


class ExplainabilityConfig(BaseModel):
    """Configuration for model explainability."""
    
    enabled: bool = Field(True, description="Enable explainability analysis")
    methods: List[str] = Field(default_factory=lambda: ["shap"], description="Explainability methods")
    shap: Optional[Dict[str, Any]] = Field(None, description="SHAP configuration")
    lime: Optional[Dict[str, Any]] = Field(None, description="LIME configuration")


class VisualizationConfig(BaseModel):
    """Configuration for visualizations."""
    
    enabled: bool = Field(True, description="Enable visualizations")
    formats: List[str] = Field(default_factory=lambda: ["png", "html"], description="Output formats")
    plots: List[str] = Field(
        default_factory=lambda: ["roc_curve", "confusion_matrix", "feature_importance"],
        description="Plot types to generate"
    )
    style: str = Field("seaborn", description="Plot style")


class ReportingConfig(BaseModel):
    """Configuration for report generation."""
    
    enabled: bool = Field(True, description="Enable report generation")
    formats: List[str] = Field(default_factory=lambda: ["html", "markdown"], description="Report formats")
    output_dir: str = Field("reports", description="Output directory for reports")
    sections: List[str] = Field(
        default_factory=lambda: [
            "executive_summary",
            "statistical_analysis",
            "performance_metrics",
            "visualizations",
            "recommendations"
        ],
        description="Report sections to include"
    )


class TrackingConfig(BaseModel):
    """Configuration for experiment tracking."""
    
    enabled: bool = Field(True, description="Enable experiment tracking")
    backend: str = Field("mlflow", description="Tracking backend: mlflow, wandb, or both")
    mlflow: Optional[Dict[str, Any]] = Field(None, description="MLflow configuration")
    wandb: Optional[Dict[str, Any]] = Field(None, description="W&B configuration")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: str = Field("INFO", description="Logging level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    log_file: Optional[str] = Field("logs/ab_testing.log", description="Log file path")
    console_output: bool = Field(True, description="Enable console output")


class ABTestConfig(BaseModel):
    """Main configuration for A/B testing framework."""
    
    experiment: ExperimentConfig
    models: Dict[str, ModelConfig]
    data: DataConfig
    routing: RoutingConfig
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    statistical_tests: StatisticalTestsConfig = Field(default_factory=StatisticalTestsConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @validator('models')
    def validate_models(cls, v: Dict[str, ModelConfig]) -> Dict[str, ModelConfig]:
        """Validate that we have at least champion and one challenger."""
        if 'champion' not in v:
            raise ValueError("Configuration must include a 'champion' model")
        if len(v) < 2:
            raise ValueError("Must have at least champion and one challenger model")
        return v
    
    class Config:
        """Pydantic configuration."""
        extra = 'forbid'  # Raise error on unknown fields


class ConfigManager:
    """Manager for loading and validating configurations."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[ABTestConfig] = None
        
        if self.config_path:
            self.load_config()
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> ABTestConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Validated ABTestConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        # Load YAML
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse and validate with Pydantic
        try:
            # Parse models separately
            models_dict = {}
            if 'models' in config_dict:
                champion_config = config_dict['models'].get('champion', {})
                models_dict['champion'] = ModelConfig(**champion_config)
                
                challengers = config_dict['models'].get('challengers', [])
                for i, challenger in enumerate(challengers):
                    name = challenger.get('name', f'challenger_{i}')
                    models_dict[name] = ModelConfig(**challenger)
            
            # Build full config
            config_dict['models'] = models_dict
            self.config = ABTestConfig(**config_dict)
            
        except Exception as e:
            raise ValueError(f"Invalid configuration: {str(e)}")
        
        return self.config
    
    def load_config_from_dict(self, config_dict: Dict[str, Any]) -> ABTestConfig:
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Validated ABTestConfig object
        """
        self.config = ABTestConfig(**config_dict)
        return self.config
    
    def get_config(self) -> ABTestConfig:
        """
        Get current configuration.
        
        Returns:
            Current ABTestConfig object
            
        Raises:
            ValueError: If config not loaded
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load_config() first.")
        return self.config
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration
        """
        if self.config is None:
            raise ValueError("No configuration to save")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict
        config_dict = self.config.dict()
        
        # Save to YAML
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def update_config(self, updates: Dict[str, Any]) -> ABTestConfig:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates
            
        Returns:
            Updated ABTestConfig object
        """
        if self.config is None:
            raise ValueError("Configuration not loaded")
        
        config_dict = self.config.dict()
        config_dict.update(updates)
        self.config = ABTestConfig(**config_dict)
        
        return self.config
    
    def validate_config(self) -> bool:
        """
        Validate current configuration.
        
        Returns:
            True if valid, raises exception otherwise
        """
        if self.config is None:
            raise ValueError("Configuration not loaded")
        
        # Pydantic already validates on creation, but we can add extra checks
        # Check that all model files exist
        for model_name, model_config in self.config.models.items():
            if not Path(model_config.path).exists():
                raise ValueError(f"Model file not found for {model_name}: {model_config.path}")
        
        # Check that data file exists
        if not Path(self.config.data.test_data_path).exists():
            raise ValueError(f"Data file not found: {self.config.data.test_data_path}")
        
        return True


def load_default_config() -> ABTestConfig:
    """
    Load default configuration.
    
    Returns:
        Default ABTestConfig object
    """
    return ABTestConfig(
        experiment=ExperimentConfig(
            name="default_experiment",
            description="Default A/B test configuration"
        ),
        models={
            'champion': ModelConfig(
                path='models/champion.pkl',
                name='champion',
                version='1.0.0'
            ),
            'challenger': ModelConfig(
                path='models/challenger.pkl',
                name='challenger',
                version='2.0.0'
            )
        },
        data=DataConfig(
            test_data_path='data/test.csv',
            target_column='target'
        ),
        routing=RoutingConfig(
            strategy='equal',
            traffic_split={'champion': 0.5, 'challenger': 0.5}
        )
    )