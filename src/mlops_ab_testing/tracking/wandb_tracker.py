"""
Weights & Biases integration for experiment tracking.

This module provides W&B tracking capabilities for A/B tests.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class WandBTracker:
    """Weights & Biases experiment tracker."""
    
    def __init__(
        self,
        project: str = "ab-testing",
        entity: Optional[str] = None
    ):
        """
        Initialize W&B tracker.
        
        Args:
            project: W&B project name
            entity: W&B entity (username or team)
        """
        self.project = project
        self.entity = entity
        self.wandb = None
        self.run = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize W&B."""
        try:
            import wandb
            self.wandb = wandb
            logger.info(f"W&B initialized: project='{self.project}'")
            
        except ImportError:
            logger.warning("W&B not installed. Install with: pip install wandb")
            self.wandb = None
    
    def start_run(
        self,
        name: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """Start a W&B run."""
        if not self.wandb:
            return
        
        self.run = self.wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config or {}
        )
        logger.info(f"Started W&B run: {self.run.id}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if not self.wandb or not self.run:
            return
        
        self.wandb.log(metrics, step=step)
    
    def log_table(self, name: str, data: Any):
        """Log table."""
        if not self.wandb or not self.run:
            return
        
        table = self.wandb.Table(dataframe=data)
        self.wandb.log({name: table})
    
    def log_artifact(self, filepath: str, name: Optional[str] = None):
        """Log artifact."""
        if not self.wandb or not self.run:
            return
        
        artifact = self.wandb.Artifact(name or "artifact", type="dataset")
        artifact.add_file(filepath)
        self.run.log_artifact(artifact)
    
    def finish(self):
        """Finish W&B run."""
        if not self.wandb or not self.run:
            return
        
        self.run.finish()
        logger.info("Finished W&B run")
    
    def log_ab_test(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Dict[str, float]],
        winner: Optional[str] = None
    ):
        """
        Log complete A/B test to W&B.
        
        Args:
            config: Test configuration
            metrics: Dictionary mapping model names to metrics
            winner: Name of winning model
        """
        if not self.wandb:
            logger.warning("W&B not available")
            return
        
        run_config = {
            'experiment_name': config.get('experiment', {}).get('name', 'unknown'),
            'n_models': len(metrics),
            'winner': winner or 'none'
        }
        
        self.start_run(name="ab_test", config=run_config)
        
        # Log metrics for each model
        for model_name, model_metrics in metrics.items():
            metrics_dict = {f'{model_name}/{k}': v for k, v in model_metrics.items()}
            self.log_metrics(metrics_dict)
        
        self.finish()
        logger.info("A/B test logged to W&B")