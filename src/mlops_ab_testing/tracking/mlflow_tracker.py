"""
MLflow integration for experiment tracking.

This module provides MLflow tracking capabilities for A/B tests.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracker."""
    
    def __init__(
        self,
        experiment_name: str = "ab_testing",
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI (None = local)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.mlflow = None
        self.run_id = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize MLflow."""
        try:
            import mlflow
            self.mlflow = mlflow
            
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow initialized: experiment='{self.experiment_name}'")
            
        except ImportError:
            logger.warning("MLflow not installed. Install with: pip install mlflow")
            self.mlflow = None
    
    def start_run(self, run_name: Optional[str] = None):
        """Start an MLflow run."""
        if not self.mlflow:
            return
        
        self.mlflow.start_run(run_name=run_name)
        self.run_id = self.mlflow.active_run().info.run_id
        logger.info(f"Started MLflow run: {self.run_id}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if not self.mlflow or not self.mlflow.active_run():
            return
        
        for key, value in params.items():
            self.mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if not self.mlflow or not self.mlflow.active_run():
            return
        
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                self.mlflow.log_metric(key, float(value), step=step)
    
    def log_model(self, model, model_name: str):
        """Log model artifact."""
        if not self.mlflow or not self.mlflow.active_run():
            return
        
        try:
            self.mlflow.sklearn.log_model(model, model_name)
            logger.info(f"Logged model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")
    
    def log_artifact(self, filepath: str):
        """Log artifact file."""
        if not self.mlflow or not self.mlflow.active_run():
            return
        
        self.mlflow.log_artifact(filepath)
    
    def end_run(self):
        """End MLflow run."""
        if not self.mlflow or not self.mlflow.active_run():
            return
        
        self.mlflow.end_run()
        logger.info("Ended MLflow run")
    
    def log_ab_test(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Dict[str, float]],
        winner: Optional[str] = None
    ):
        """
        Log complete A/B test to MLflow.
        
        Args:
            config: Test configuration
            metrics: Dictionary mapping model names to metrics
            winner: Name of winning model
        """
        if not self.mlflow:
            logger.warning("MLflow not available")
            return
        
        self.start_run(run_name="ab_test")
        
        # Log config
        self.log_params({
            'experiment_name': config.get('experiment', {}).get('name', 'unknown'),
            'n_models': len(metrics),
            'winner': winner or 'none'
        })
        
        # Log metrics for each model
        for model_name, model_metrics in metrics.items():
            for metric_name, value in model_metrics.items():
                self.log_metrics({f'{model_name}_{metric_name}': value})
        
        self.end_run()
        logger.info("A/B test logged to MLflow")


def track_with_mlflow(
    func,
    experiment_name: str = "ab_testing",
    auto_log: bool = True
):
    """
    Decorator to track function with MLflow.
    
    Args:
        func: Function to track
        experiment_name: Name of experiment
        auto_log: Enable autologging
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        tracker = MLflowTracker(experiment_name=experiment_name)
        
        if tracker.mlflow and auto_log:
            tracker.mlflow.sklearn.autolog()
        
        tracker.start_run()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            tracker.end_run()
    
    return wrapper