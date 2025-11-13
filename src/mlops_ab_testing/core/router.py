"""
Traffic routing for A/B testing.

This module handles distributing test data to different models
according to various routing strategies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseRouter(ABC):
    """Abstract base class for traffic routers."""
    
    def __init__(self, traffic_split: Dict[str, float], random_seed: int = 42):
        """
        Initialize router.
        
        Args:
            traffic_split: Dictionary mapping model names to traffic percentages
            random_seed: Random seed for reproducibility
        """
        self.traffic_split = traffic_split
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
        # Validate traffic split
        total = sum(traffic_split.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Traffic split must sum to 1.0, got {total}")
    
    @abstractmethod
    def route(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Route data to different models.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary mapping model names to their data splits
        """
        pass
    
    def get_model_names(self) -> List[str]:
        """Get list of model names."""
        return list(self.traffic_split.keys())


class EqualRouter(BaseRouter):
    """Router that splits traffic equally (with random assignment)."""
    
    def route(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Route data with equal (random) distribution.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary mapping model names to their data splits
        """
        n_samples = len(data)
        model_names = self.get_model_names()
        
        logger.info(f"Routing {n_samples} samples to {len(model_names)} models")
        
        # Create random assignments
        assignments = self.rng.choice(
            model_names,
            size=n_samples,
            p=list(self.traffic_split.values())
        )
        
        # Split data
        routed_data = {}
        for model_name in model_names:
            mask = assignments == model_name
            routed_data[model_name] = data[mask].copy()
            logger.info(
                f"  {model_name}: {len(routed_data[model_name])} samples "
                f"({len(routed_data[model_name])/n_samples*100:.1f}%)"
            )
        
        return routed_data


class StratifiedRouter(BaseRouter):
    """Router that maintains class/group distribution across splits."""
    
    def __init__(
        self,
        traffic_split: Dict[str, float],
        stratify_column: str,
        random_seed: int = 42
    ):
        """
        Initialize stratified router.
        
        Args:
            traffic_split: Dictionary mapping model names to traffic percentages
            stratify_column: Column name to stratify by
            random_seed: Random seed for reproducibility
        """
        super().__init__(traffic_split, random_seed)
        self.stratify_column = stratify_column
    
    def route(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Route data with stratified distribution.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary mapping model names to their data splits
        """
        if self.stratify_column not in data.columns:
            raise ValueError(f"Stratify column '{self.stratify_column}' not found in data")
        
        n_samples = len(data)
        model_names = self.get_model_names()
        
        logger.info(
            f"Routing {n_samples} samples to {len(model_names)} models "
            f"(stratified by '{self.stratify_column}')"
        )
        
        # Initialize result dictionaries
        routed_data = {name: [] for name in model_names}
        
        # Stratify by each unique value
        for stratum_value in data[self.stratify_column].unique():
            stratum_mask = data[self.stratify_column] == stratum_value
            stratum_data = data[stratum_mask]
            stratum_size = len(stratum_data)
            
            # Assign samples from this stratum
            assignments = self.rng.choice(
                model_names,
                size=stratum_size,
                p=list(self.traffic_split.values())
            )
            
            # Split stratum data
            for model_name in model_names:
                mask = assignments == model_name
                routed_data[model_name].append(stratum_data[mask])
        
        # Concatenate splits
        routed_data = {
            name: pd.concat(splits, ignore_index=True)
            for name, splits in routed_data.items()
        }
        
        # Log distribution
        for model_name in model_names:
            logger.info(
                f"  {model_name}: {len(routed_data[model_name])} samples "
                f"({len(routed_data[model_name])/n_samples*100:.1f}%)"
            )
        
        return routed_data


class WeightedRouter(BaseRouter):
    """
    Router with weighted (non-uniform) distribution.
    
    This is similar to EqualRouter but allows for unequal splits.
    """
    
    def route(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Route data with weighted distribution.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary mapping model names to their data splits
        """
        # Weighted routing is the same as equal routing with different probabilities
        n_samples = len(data)
        model_names = self.get_model_names()
        
        logger.info(
            f"Routing {n_samples} samples to {len(model_names)} models "
            f"(weighted: {self.traffic_split})"
        )
        
        # Create weighted assignments
        assignments = self.rng.choice(
            model_names,
            size=n_samples,
            p=list(self.traffic_split.values())
        )
        
        # Split data
        routed_data = {}
        for model_name in model_names:
            mask = assignments == model_name
            routed_data[model_name] = data[mask].copy()
            logger.info(
                f"  {model_name}: {len(routed_data[model_name])} samples "
                f"({len(routed_data[model_name])/n_samples*100:.1f}%)"
            )
        
        return routed_data


class SequentialRouter(BaseRouter):
    """Router that assigns samples sequentially (deterministic)."""
    
    def route(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Route data sequentially.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary mapping model names to their data splits
        """
        n_samples = len(data)
        model_names = self.get_model_names()
        
        logger.info(
            f"Routing {n_samples} samples to {len(model_names)} models (sequential)"
        )
        
        # Calculate split points
        split_points = [0]
        cumsum = 0
        for model_name in model_names[:-1]:
            cumsum += self.traffic_split[model_name]
            split_points.append(int(cumsum * n_samples))
        split_points.append(n_samples)
        
        # Split data
        routed_data = {}
        for i, model_name in enumerate(model_names):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            routed_data[model_name] = data.iloc[start_idx:end_idx].copy()
            logger.info(
                f"  {model_name}: {len(routed_data[model_name])} samples "
                f"(indices {start_idx}-{end_idx})"
            )
        
        return routed_data


class TrafficRouter:
    """Main traffic router that manages routing strategies."""
    
    def __init__(
        self,
        strategy: str,
        traffic_split: Dict[str, float],
        stratify_column: Optional[str] = None,
        random_seed: int = 42
    ):
        """
        Initialize traffic router.
        
        Args:
            strategy: Routing strategy (equal, stratified, weighted, sequential)
            traffic_split: Dictionary mapping model names to traffic percentages
            stratify_column: Column for stratification (required for stratified strategy)
            random_seed: Random seed for reproducibility
        """
        self.strategy = strategy
        self.traffic_split = traffic_split
        self.stratify_column = stratify_column
        self.random_seed = random_seed
        
        # Create appropriate router
        self.router = self._create_router()
    
    def _create_router(self) -> BaseRouter:
        """
        Create router based on strategy.
        
        Returns:
            Router instance
        """
        if self.strategy == 'equal':
            return EqualRouter(self.traffic_split, self.random_seed)
        
        elif self.strategy == 'stratified':
            if not self.stratify_column:
                raise ValueError("stratify_column required for stratified routing")
            return StratifiedRouter(
                self.traffic_split,
                self.stratify_column,
                self.random_seed
            )
        
        elif self.strategy == 'weighted':
            return WeightedRouter(self.traffic_split, self.random_seed)
        
        elif self.strategy == 'sequential':
            return SequentialRouter(self.traffic_split, self.random_seed)
        
        else:
            raise ValueError(
                f"Unknown routing strategy: {self.strategy}. "
                f"Valid strategies: equal, stratified, weighted, sequential"
            )
    
    def route(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Route data to models.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary mapping model names to their data splits
        """
        logger.info(f"Using {self.strategy} routing strategy")
        return self.router.route(data)
    
    def get_routing_summary(self, routed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Get summary of routing distribution.
        
        Args:
            routed_data: Dictionary of routed data
            
        Returns:
            Summary dataframe
        """
        total_samples = sum(len(df) for df in routed_data.values())
        
        summary_data = []
        for model_name, df in routed_data.items():
            n_samples = len(df)
            percentage = (n_samples / total_samples * 100) if total_samples > 0 else 0
            expected_percentage = self.traffic_split.get(model_name, 0) * 100
            
            summary_data.append({
                'Model': model_name,
                'Samples': n_samples,
                'Percentage': f"{percentage:.2f}%",
                'Expected': f"{expected_percentage:.2f}%",
                'Deviation': f"{abs(percentage - expected_percentage):.2f}%"
            })
        
        return pd.DataFrame(summary_data)


def create_router_from_config(config) -> TrafficRouter:
    """
    Create router from configuration object.
    
    Args:
        config: Configuration object with routing settings
        
    Returns:
        TrafficRouter instance
    """
    return TrafficRouter(
        strategy=config.routing.strategy,
        traffic_split=config.routing.traffic_split,
        stratify_column=config.routing.stratify_by,
        random_seed=config.experiment.random_seed
    )