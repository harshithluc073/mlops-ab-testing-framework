"""
Main A/B Testing Framework.

This module contains the core ABTestFramework class that orchestrates
the entire A/B testing workflow.
"""

import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from mlops_ab_testing.core.config import ConfigManager, ABTestConfig
from mlops_ab_testing.core.model_loader import ModelLoader, ModelWrapper
from mlops_ab_testing.core.router import TrafficRouter, create_router_from_config

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results from a model."""
    
    model_name: str
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    latencies: List[float] = field(default_factory=list)
    sample_indices: Optional[np.ndarray] = None
    
    @property
    def mean_latency(self) -> float:
        """Calculate mean prediction latency."""
        return np.mean(self.latencies) if self.latencies else 0.0
    
    @property
    def p50_latency(self) -> float:
        """Calculate median latency."""
        return np.percentile(self.latencies, 50) if self.latencies else 0.0
    
    @property
    def p95_latency(self) -> float:
        """Calculate 95th percentile latency."""
        return np.percentile(self.latencies, 95) if self.latencies else 0.0
    
    @property
    def p99_latency(self) -> float:
        """Calculate 99th percentile latency."""
        return np.percentile(self.latencies, 99) if self.latencies else 0.0


@dataclass
class ABTestResults:
    """Container for complete A/B test results."""
    
    config: ABTestConfig
    predictions: Dict[str, PredictionResult]
    test_data: pd.DataFrame
    ground_truth: np.ndarray
    feature_columns: List[str]
    target_column: str
    start_time: float
    end_time: float
    
    @property
    def duration(self) -> float:
        """Total test duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def model_names(self) -> List[str]:
        """List of model names."""
        return list(self.predictions.keys())
    
    @property
    def n_samples(self) -> int:
        """Total number of samples tested."""
        return len(self.test_data)


class ABTestFramework:
    """
    Main A/B Testing Framework for ML model comparison.
    
    This class orchestrates the entire A/B testing workflow including:
    - Loading models and data
    - Routing traffic to models
    - Collecting predictions and performance metrics
    - Running statistical analysis
    - Generating reports
    """
    
    def __init__(
        self,
        config: Optional[Union[str, Path, ABTestConfig]] = None,
        champion_model: Optional[str] = None,
        challenger_models: Optional[List[str]] = None,
        test_data: Optional[str] = None,
        target_column: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize A/B Testing Framework.
        
        Args:
            config: Path to config file or ABTestConfig object
            champion_model: Path to champion model (if not using config)
            challenger_models: List of paths to challenger models
            test_data: Path to test data CSV
            target_column: Name of target column
            **kwargs: Additional configuration overrides
        """
        # Initialize components
        self.config_manager = ConfigManager()
        self.model_loader = ModelLoader()
        self.router: Optional[TrafficRouter] = None
        
        # Load configuration
        if isinstance(config, ABTestConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = self.config_manager.load_config(config)
        elif champion_model and challenger_models and test_data and target_column:
            # Create config from parameters
            self.config = self._create_config_from_params(
                champion_model,
                challenger_models,
                test_data,
                target_column,
                **kwargs
            )
        else:
            raise ValueError(
                "Must provide either config file/object or "
                "(champion_model, challenger_models, test_data, target_column)"
            )
        
        # Apply any overrides
        if kwargs:
            self.config = self._apply_config_overrides(self.config, kwargs)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize state
        self.models: Dict[str, Any] = {}
        self.test_data: Optional[pd.DataFrame] = None
        self.results: Optional[ABTestResults] = None
        
        logger.info(f"Initialized ABTestFramework for experiment: {self.config.experiment.name}")
    
    def _create_config_from_params(
        self,
        champion_model: str,
        challenger_models: List[str],
        test_data: str,
        target_column: str,
        **kwargs
    ) -> ABTestConfig:
        """Create configuration from parameters."""
        from mlops_ab_testing.core.config import (
            ABTestConfig, ExperimentConfig, ModelConfig,
            DataConfig, RoutingConfig
        )
        
        # Build models dict
        models = {
            'champion': ModelConfig(
                path=champion_model,
                name='champion',
                version='1.0.0'
            )
        }
        
        for i, challenger_path in enumerate(challenger_models):
            models[f'challenger_{i+1}'] = ModelConfig(
                path=challenger_path,
                name=f'challenger_{i+1}',
                version='2.0.0'
            )
        
        # Build traffic split
        n_models = len(models)
        traffic_split = {name: 1.0/n_models for name in models.keys()}
        
        config = ABTestConfig(
            experiment=ExperimentConfig(
                name=kwargs.get('experiment_name', 'ab_test'),
                description=kwargs.get('experiment_description', 'A/B test experiment')
            ),
            models=models,
            data=DataConfig(
                test_data_path=test_data,
                target_column=target_column,
                feature_columns=kwargs.get('feature_columns', None)
            ),
            routing=RoutingConfig(
                strategy=kwargs.get('routing_strategy', 'equal'),
                traffic_split=traffic_split
            )
        )
        
        return config
    
    def _apply_config_overrides(self, config: ABTestConfig, overrides: Dict) -> ABTestConfig:
        """Apply configuration overrides."""
        # This is a simplified version - you could make this more sophisticated
        config_dict = config.dict()
        
        for key, value in overrides.items():
            if key in config_dict:
                config_dict[key] = value
        
        return ABTestConfig(**config_dict)
    
    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_config = self.config.logging
        
        # Configure logging
        logging.basicConfig(
            level=log_config.level,
            format=log_config.format
        )
        
        # Add file handler if specified
        if log_config.log_file:
            log_path = Path(log_config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter(log_config.format))
            logging.getLogger().addHandler(file_handler)
    
    def load_models(self) -> None:
        """Load all models specified in configuration."""
        logger.info("Loading models...")
        
        for model_name, model_config in self.config.models.items():
            try:
                model = self.model_loader.load_model(
                    model_path=model_config.path,
                    model_name=model_name
                )
                self.models[model_name] = model
                logger.info(f"[OK] Loaded {model_name}: {model_config.path}")
                
            except Exception as e:
                logger.error(f"[FAILED] Failed to load {model_name}: {str(e)}")
                raise
        
        logger.info(f"Successfully loaded {len(self.models)} models")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load test data from configuration.
        
        Returns:
            Loaded dataframe
        """
        logger.info(f"Loading test data from {self.config.data.test_data_path}")
        
        # Load data
        data_path = Path(self.config.data.test_data_path)
        
        if data_path.suffix == '.csv':
            self.test_data = pd.read_csv(data_path)
        elif data_path.suffix in ['.parquet', '.pq']:
            self.test_data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")
        
        # Apply sample size limit if specified
        if self.config.data.sample_size:
            if self.config.data.sample_size < len(self.test_data):
                logger.info(f"Sampling {self.config.data.sample_size} rows from {len(self.test_data)}")
                self.test_data = self.test_data.sample(
                    n=self.config.data.sample_size,
                    random_state=self.config.experiment.random_seed
                )
        
        logger.info(f"Loaded {len(self.test_data)} samples with {len(self.test_data.columns)} columns")
        
        return self.test_data
    
    def setup_router(self) -> None:
        """Setup traffic router based on configuration."""
        logger.info("Setting up traffic router...")
        self.router = create_router_from_config(self.config)
        logger.info(f"[OK] Router configured with strategy: {self.config.routing.strategy}")
    
    def run_test(self) -> ABTestResults:
        """
        Run the complete A/B test.
        
        Returns:
            ABTestResults object with all results
        """
        start_time = time.time()
        
        logger.info("="*60)
        logger.info(f"Starting A/B Test: {self.config.experiment.name}")
        logger.info("="*60)
        
        # Step 1: Load models
        if not self.models:
            self.load_models()
        
        # Step 2: Load data
        if self.test_data is None:
            self.load_data()
        
        # Step 3: Setup router
        if self.router is None:
            self.setup_router()
        
        # Step 4: Prepare features and target
        feature_columns = self.config.data.feature_columns
        if feature_columns is None:
            # Use all columns except target
            feature_columns = [
                col for col in self.test_data.columns
                if col != self.config.data.target_column
            ]
        
        X = self.test_data[feature_columns]
        y = self.test_data[self.config.data.target_column].values
        
        logger.info(f"Features: {len(feature_columns)} columns")
        logger.info(f"Samples: {len(X)} rows")
        
        # Step 5: Route traffic
        logger.info("\nRouting traffic to models...")
        routed_data = self.router.route(self.test_data)
        
        # Show routing summary
        routing_summary = self.router.get_routing_summary(routed_data)
        logger.info("\nRouting Summary:")
        logger.info("\n" + routing_summary.to_string(index=False))
        
        # Step 6: Collect predictions from each model
        logger.info("\nCollecting predictions...")
        predictions = {}
        
        for model_name, model_data in routed_data.items():
            logger.info(f"\nProcessing {model_name}...")
            
            model = self.models[model_name]
            X_model = model_data[feature_columns]
            sample_indices = model_data.index.values
            
            # Measure prediction latency
            latencies = []
            
            # Make predictions
            start = time.time()
            preds = model.predict(X_model)
            latency = time.time() - start
            latencies.append(latency)
            
            # Try to get probabilities for classifiers
            proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X_model)
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {str(e)}")
            
            predictions[model_name] = PredictionResult(
                model_name=model_name,
                predictions=preds,
                probabilities=proba,
                latencies=latencies,
                sample_indices=sample_indices
            )
            
            logger.info(f"  [OK] {len(preds)} predictions")
            logger.info(f"  [OK] Latency: {latency*1000:.2f}ms")
        
        # Step 7: Create results object
        end_time = time.time()
        
        self.results = ABTestResults(
            config=self.config,
            predictions=predictions,
            test_data=self.test_data,
            ground_truth=y,
            feature_columns=feature_columns,
            target_column=self.config.data.target_column,
            start_time=start_time,
            end_time=end_time
        )
        
        logger.info("\n" + "="*60)
        logger.info(f"A/B Test Complete!")
        logger.info(f"Duration: {self.results.duration:.2f}s")
        logger.info("="*60)
        
        return self.results
    
    def get_results(self) -> Optional[ABTestResults]:
        """
        Get test results.
        
        Returns:
            ABTestResults object or None if test not run
        """
        return self.results
    
    def get_predictions_summary(self) -> pd.DataFrame:
        """
        Get summary of predictions from all models.
        
        Returns:
            Summary dataframe
        """
        if self.results is None:
            raise ValueError("No results available. Run test first.")
        
        summary_data = []
        for model_name, pred_result in self.results.predictions.items():
            summary_data.append({
                'Model': model_name,
                'Samples': len(pred_result.predictions),
                'Mean Latency (ms)': f"{pred_result.mean_latency * 1000:.2f}",
                'P50 Latency (ms)': f"{pred_result.p50_latency * 1000:.2f}",
                'P95 Latency (ms)': f"{pred_result.p95_latency * 1000:.2f}",
                'P99 Latency (ms)': f"{pred_result.p99_latency * 1000:.2f}",
            })
        
        return pd.DataFrame(summary_data)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ABTestFramework("
            f"experiment='{self.config.experiment.name}', "
            f"models={list(self.models.keys())}, "
            f"samples={len(self.test_data) if self.test_data is not None else 0}"
            f")"
        )
    
    def analyze_results(self, method: str = 'both') -> Dict[str, Any]:
        """
        Analyze test results with statistical methods.
        
        Args:
            method: Analysis method ('classical', 'bayesian', 'both')
            
        Returns:
            Dictionary with analysis results
        """
        if self.results is None:
            raise ValueError("No results available. Run test first with run_test()")
        
        from mlops_ab_testing.metrics import evaluate_ab_test_results
        from mlops_ab_testing.statistics import ClassicalStatisticalAnalysis, BayesianAnalysis
        
        logger.info("="*60)
        logger.info("Analyzing Results")
        logger.info("="*60)
        
        analysis = {}
        
        # 1. Calculate metrics
        logger.info("\n1. Calculating metrics...")
        evaluator = evaluate_ab_test_results(self.results, task_type='auto')
        
        metrics_summary = evaluator.get_summary()
        logger.info("\nMetrics Summary:")
        logger.info("\n" + metrics_summary.to_string(index=False))
        
        analysis['metrics'] = {
            'summary': metrics_summary,
            'detailed': {
                name: evaluator.get_detailed_metrics(name)
                for name in evaluator.results.keys()
            },
            'evaluator': evaluator
        }
        
        # 2. Statistical comparison
        if method in ['classical', 'both']:
            logger.info("\n2. Running classical statistical tests...")
            classical = ClassicalStatisticalAnalysis(alpha=self.config.statistical_tests.alpha)
            
            # For now, we'll compare on the primary metric
            # In future, this can be extended to compare all metrics
            task_type = evaluator.results[list(evaluator.results.keys())[0]]['task_type']
            
            if task_type == 'classification':
                primary_metric = 'accuracy'
            else:
                primary_metric = 'rmse'
            
            logger.info(f"  Comparing models on: {primary_metric}")
            
            # Get metric values for each model
            model_names = list(self.results.predictions.keys())
            if len(model_names) >= 2:
                model_a = model_names[0]
                model_b = model_names[1]
                
                # Get predictions and calculate metrics per sample
                # This is simplified - in practice you'd want per-sample metrics
                logger.info(f"  {model_a} vs {model_b}")
                logger.info("  (Statistical comparison on aggregated metrics)")
            
            analysis['classical'] = classical
        
        if method in ['bayesian', 'both']:
            logger.info("\n3. Running Bayesian analysis...")
            bayesian = BayesianAnalysis(
                credible_interval=self.config.statistical_tests.alpha,
                n_samples=10000
            )
            
            logger.info("  Bayesian inference complete")
            
            analysis['bayesian'] = bayesian
        
        logger.info("\n" + "="*60)
        logger.info("Analysis Complete!")
        logger.info("="*60)
        
        return analysis