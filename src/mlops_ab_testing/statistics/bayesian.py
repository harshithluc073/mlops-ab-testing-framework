"""
Bayesian inference for A/B testing.

This module provides Bayesian methods for comparing models including
credible intervals and probability of superiority.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class BayesianAnalysis:
    """
    Bayesian analysis for A/B testing.
    """
    
    def __init__(
        self,
        credible_interval: float = 0.95,
        n_samples: int = 10000,
        random_seed: int = 42
    ):
        """
        Initialize Bayesian analysis.
        
        Args:
            credible_interval: Credible interval level (default: 0.95)
            n_samples: Number of posterior samples
            random_seed: Random seed for reproducibility
        """
        self.credible_interval = credible_interval
        self.n_samples = n_samples
        self.rng = np.random.RandomState(random_seed)
        self.results: Dict[str, Dict] = {}
    
    def compare_metrics(
        self,
        metric_a: np.ndarray,
        metric_b: np.ndarray,
        metric_name: str = 'metric'
    ) -> Dict:
        """
        Compare metrics using Bayesian approach.
        
        Args:
            metric_a: Metric values for model A
            metric_b: Metric values for model B
            metric_name: Name of the metric
            
        Returns:
            Dictionary with Bayesian comparison results
        """
        logger.info(f"Bayesian comparison of {metric_name}...")
        
        # Estimate parameters from data
        mean_a, std_a = np.mean(metric_a), np.std(metric_a, ddof=1)
        mean_b, std_b = np.mean(metric_b), np.std(metric_b, ddof=1)
        n_a, n_b = len(metric_a), len(metric_b)
        
        # Sample from posterior distributions (assuming normal likelihood)
        # Using normal approximation for large samples
        posterior_a = self.rng.normal(
            loc=mean_a,
            scale=std_a / np.sqrt(n_a),
            size=self.n_samples
        )
        
        posterior_b = self.rng.normal(
            loc=mean_b,
            scale=std_b / np.sqrt(n_b),
            size=self.n_samples
        )
        
        # Calculate difference
        posterior_diff = posterior_b - posterior_a
        
        # Probability that B > A
        prob_b_better = np.mean(posterior_diff > 0)
        
        # Credible interval for difference
        ci_lower = np.percentile(posterior_diff, (1 - self.credible_interval) / 2 * 100)
        ci_upper = np.percentile(posterior_diff, (1 + self.credible_interval) / 2 * 100)
        
        # Expected loss (risk of choosing wrong model)
        expected_loss_choose_a = np.mean(np.maximum(0, posterior_diff))
        expected_loss_choose_b = np.mean(np.maximum(0, -posterior_diff))
        
        result = {
            'metric_name': metric_name,
            'posterior_mean_a': float(np.mean(posterior_a)),
            'posterior_mean_b': float(np.mean(posterior_b)),
            'posterior_std_a': float(np.std(posterior_a)),
            'posterior_std_b': float(np.std(posterior_b)),
            'difference': {
                'mean': float(np.mean(posterior_diff)),
                'std': float(np.std(posterior_diff)),
                'credible_interval': {
                    'lower': float(ci_lower),
                    'upper': float(ci_upper),
                    'level': self.credible_interval
                }
            },
            'probability_b_better': float(prob_b_better),
            'probability_a_better': float(1 - prob_b_better),
            'expected_loss': {
                'choose_a': float(expected_loss_choose_a),
                'choose_b': float(expected_loss_choose_b),
                'recommended': 'B' if expected_loss_choose_b < expected_loss_choose_a else 'A'
            },
            'practical_significance': self._assess_practical_significance(
                posterior_diff, ci_lower, ci_upper
            )
        }
        
        self.results[metric_name] = result
        return result
    
    def compare_proportions(
        self,
        success_a: int,
        n_a: int,
        success_b: int,
        n_b: int,
        metric_name: str = 'proportion'
    ) -> Dict:
        """
        Compare proportions using Beta-Binomial conjugate prior.
        
        Args:
            success_a: Number of successes in model A
            n_a: Total samples in model A
            success_b: Number of successes in model B
            n_b: Total samples in model B
            metric_name: Name of the metric
            
        Returns:
            Dictionary with Bayesian comparison results
        """
        logger.info(f"Bayesian comparison of {metric_name} (proportions)...")
        
        # Use uniform prior Beta(1, 1)
        alpha_prior, beta_prior = 1, 1
        
        # Posterior is Beta(alpha + successes, beta + failures)
        alpha_a = alpha_prior + success_a
        beta_a = beta_prior + (n_a - success_a)
        
        alpha_b = alpha_prior + success_b
        beta_b = beta_prior + (n_b - success_b)
        
        # Sample from posteriors
        posterior_a = self.rng.beta(alpha_a, beta_a, size=self.n_samples)
        posterior_b = self.rng.beta(alpha_b, beta_b, size=self.n_samples)
        
        # Calculate difference
        posterior_diff = posterior_b - posterior_a
        
        # Probability that B > A
        prob_b_better = np.mean(posterior_diff > 0)
        
        # Credible intervals
        ci_lower = np.percentile(posterior_diff, (1 - self.credible_interval) / 2 * 100)
        ci_upper = np.percentile(posterior_diff, (1 + self.credible_interval) / 2 * 100)
        
        # Expected loss
        expected_loss_choose_a = np.mean(np.maximum(0, posterior_diff))
        expected_loss_choose_b = np.mean(np.maximum(0, -posterior_diff))
        
        result = {
            'metric_name': metric_name,
            'posterior_mean_a': float(np.mean(posterior_a)),
            'posterior_mean_b': float(np.mean(posterior_b)),
            'observed_proportion_a': float(success_a / n_a),
            'observed_proportion_b': float(success_b / n_b),
            'difference': {
                'mean': float(np.mean(posterior_diff)),
                'std': float(np.std(posterior_diff)),
                'credible_interval': {
                    'lower': float(ci_lower),
                    'upper': float(ci_upper),
                    'level': self.credible_interval
                }
            },
            'probability_b_better': float(prob_b_better),
            'probability_a_better': float(1 - prob_b_better),
            'expected_loss': {
                'choose_a': float(expected_loss_choose_a),
                'choose_b': float(expected_loss_choose_b),
                'recommended': 'B' if expected_loss_choose_b < expected_loss_choose_a else 'A'
            }
        }
        
        self.results[metric_name] = result
        return result
    
    def _assess_practical_significance(
        self,
        posterior_diff: np.ndarray,
        ci_lower: float,
        ci_upper: float,
        rope: Tuple[float, float] = (-0.01, 0.01)
    ) -> Dict:
        """
        Assess practical significance using Region of Practical Equivalence (ROPE).
        
        Args:
            posterior_diff: Posterior samples of difference
            ci_lower: Lower bound of credible interval
            ci_upper: Upper bound of credible interval
            rope: Region of practical equivalence (default: ±0.01)
            
        Returns:
            Dictionary with practical significance assessment
        """
        rope_lower, rope_upper = rope
        
        # Probability in ROPE
        prob_in_rope = np.mean((posterior_diff > rope_lower) & (posterior_diff < rope_upper))
        
        # Credible interval overlap with ROPE
        ci_fully_in_rope = (ci_lower >= rope_lower) and (ci_upper <= rope_upper)
        ci_fully_outside_rope = (ci_upper < rope_lower) or (ci_lower > rope_upper)
        
        if ci_fully_outside_rope:
            decision = 'practically_significant'
            interpretation = 'Models are practically different'
        elif ci_fully_in_rope:
            decision = 'practically_equivalent'
            interpretation = 'Models are practically equivalent'
        else:
            decision = 'uncertain'
            interpretation = 'Uncertain - more data needed'
        
        return {
            'rope': {'lower': rope_lower, 'upper': rope_upper},
            'probability_in_rope': float(prob_in_rope),
            'decision': decision,
            'interpretation': interpretation
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to DataFrame.
        
        Returns:
            DataFrame with Bayesian results
        """
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for metric_name, result in self.results.items():
            row = {
                'Metric': metric_name,
                'Mean Diff': result['difference']['mean'],
                'CI Lower': result['difference']['credible_interval']['lower'],
                'CI Upper': result['difference']['credible_interval']['upper'],
                'P(B > A)': result['probability_b_better'],
                'Recommended': result['expected_loss']['recommended']
            }
            
            if 'practical_significance' in result:
                row['Practical Significance'] = result['practical_significance']['decision']
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_winner(self, threshold: float = 0.95) -> Optional[str]:
        """
        Get the winning model based on probability threshold.
        
        Args:
            threshold: Probability threshold for declaring a winner
            
        Returns:
            Winner ('A', 'B', or None if uncertain)
        """
        if not self.results:
            return None
        
        # Average probability across all metrics
        avg_prob_b = np.mean([r['probability_b_better'] for r in self.results.values()])
        
        if avg_prob_b >= threshold:
            return 'B'
        elif avg_prob_b <= (1 - threshold):
            return 'A'
        else:
            return None  # Uncertain


def bayesian_meta_analysis(results_list: list) -> Dict:
    """
    Perform Bayesian meta-analysis across multiple experiments.
    
    Args:
        results_list: List of BayesianAnalysis results
        
    Returns:
        Dictionary with meta-analysis results
    """
    # This is a simplified version
    # In practice, you'd use hierarchical Bayesian models
    
    all_probs = []
    for results in results_list:
        if isinstance(results, dict):
            all_probs.extend([r.get('probability_b_better', 0.5) 
                            for r in results.values()])
    
    meta_prob = np.mean(all_probs) if all_probs else 0.5
    
    return {
        'meta_probability_b_better': float(meta_prob),
        'n_experiments': len(results_list),
        'confidence': 'high' if abs(meta_prob - 0.5) > 0.3 else 'low'
    }