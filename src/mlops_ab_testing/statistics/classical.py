"""
Classical statistical tests for A/B testing.

This module provides hypothesis testing methods including t-tests,
chi-square tests, and confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy import stats
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


class StatisticalTest:
    """Base class for statistical tests."""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical test.
        
        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
    
    def test(self, *args, **kwargs) -> Dict:
        """Run the statistical test."""
        raise NotImplementedError


class TTest(StatisticalTest):
    """T-test for comparing two groups."""
    
    def test(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        paired: bool = False,
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        Perform t-test.
        
        Args:
            group_a: First group data
            group_b: Second group data
            paired: Whether to use paired t-test
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
            
        Returns:
            Dictionary with test results
        """
        if paired:
            statistic, pvalue = stats.ttest_rel(group_a, group_b, alternative=alternative)
        else:
            statistic, pvalue = stats.ttest_ind(group_a, group_b, alternative=alternative)
        
        significant = pvalue < self.alpha
        
        result = {
            'test': 'T-Test (paired)' if paired else 'T-Test (independent)',
            'statistic': float(statistic),
            'p_value': float(pvalue),
            'alpha': self.alpha,
            'significant': significant,
            'conclusion': 'reject H0' if significant else 'fail to reject H0',
            'alternative': alternative,
            'mean_a': float(np.mean(group_a)),
            'mean_b': float(np.mean(group_b)),
            'std_a': float(np.std(group_a, ddof=1)),
            'std_b': float(np.std(group_b, ddof=1)),
            'n_a': len(group_a),
            'n_b': len(group_b)
        }
        
        return result


class ProportionTest(StatisticalTest):
    """Test for comparing two proportions."""
    
    def test(
        self,
        success_a: int,
        n_a: int,
        success_b: int,
        n_b: int,
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        Perform two-proportion z-test.
        
        Args:
            success_a: Number of successes in group A
            n_a: Total samples in group A
            success_b: Number of successes in group B
            n_b: Total samples in group B
            alternative: Alternative hypothesis
            
        Returns:
            Dictionary with test results
        """
        p_a = success_a / n_a
        p_b = success_b / n_b
        
        # Pooled proportion
        p_pool = (success_a + success_b) / (n_a + n_b)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        
        # Z-statistic
        z_stat = (p_a - p_b) / se if se > 0 else 0
        
        # P-value
        if alternative == 'two-sided':
            pvalue = 2 * (1 - norm.cdf(abs(z_stat)))
        elif alternative == 'greater':
            pvalue = 1 - norm.cdf(z_stat)
        else:  # less
            pvalue = norm.cdf(z_stat)
        
        significant = pvalue < self.alpha
        
        result = {
            'test': 'Two-Proportion Z-Test',
            'statistic': float(z_stat),
            'p_value': float(pvalue),
            'alpha': self.alpha,
            'significant': significant,
            'conclusion': 'reject H0' if significant else 'fail to reject H0',
            'alternative': alternative,
            'proportion_a': float(p_a),
            'proportion_b': float(p_b),
            'difference': float(p_a - p_b),
            'pooled_proportion': float(p_pool),
            'n_a': n_a,
            'n_b': n_b
        }
        
        return result


class ChiSquareTest(StatisticalTest):
    """Chi-square test for independence."""
    
    def test(
        self,
        observed: np.ndarray,
        expected: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Perform chi-square test.
        
        Args:
            observed: Observed frequencies (contingency table)
            expected: Expected frequencies (optional)
            
        Returns:
            Dictionary with test results
        """
        if expected is None:
            # Test of independence
            statistic, pvalue, dof, expected = stats.chi2_contingency(observed)
            test_name = 'Chi-Square Test of Independence'
        else:
            # Goodness of fit test
            statistic, pvalue = stats.chisquare(observed.flatten(), expected.flatten())
            dof = len(observed.flatten()) - 1
            test_name = 'Chi-Square Goodness of Fit Test'
        
        significant = pvalue < self.alpha
        
        result = {
            'test': test_name,
            'statistic': float(statistic),
            'p_value': float(pvalue),
            'degrees_of_freedom': int(dof),
            'alpha': self.alpha,
            'significant': significant,
            'conclusion': 'reject H0' if significant else 'fail to reject H0'
        }
        
        return result


class MannWhitneyTest(StatisticalTest):
    """Mann-Whitney U test (non-parametric alternative to t-test)."""
    
    def test(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        Perform Mann-Whitney U test.
        
        Args:
            group_a: First group data
            group_b: Second group data
            alternative: Alternative hypothesis
            
        Returns:
            Dictionary with test results
        """
        statistic, pvalue = stats.mannwhitneyu(
            group_a, group_b, alternative=alternative
        )
        
        significant = pvalue < self.alpha
        
        result = {
            'test': 'Mann-Whitney U Test',
            'statistic': float(statistic),
            'p_value': float(pvalue),
            'alpha': self.alpha,
            'significant': significant,
            'conclusion': 'reject H0' if significant else 'fail to reject H0',
            'alternative': alternative,
            'median_a': float(np.median(group_a)),
            'median_b': float(np.median(group_b)),
            'n_a': len(group_a),
            'n_b': len(group_b)
        }
        
        return result


def confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for mean.
    
    Args:
        data: Data array
        confidence: Confidence level
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    mean = np.mean(data)
    se = stats.sem(data)
    margin = se * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return float(mean - margin), float(mean + margin)


def effect_size_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Args:
        group_a: First group data
        group_b: Second group data
        
    Returns:
        Cohen's d
    """
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    
    # Pooled standard deviation
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
    return float(d)


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values
        alpha: Significance level
        
    Returns:
        List of booleans indicating significance after correction
    """
    corrected_alpha = alpha / len(p_values)
    return [p < corrected_alpha for p in p_values]


class ClassicalStatisticalAnalysis:
    """
    Comprehensive classical statistical analysis for A/B testing.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analysis.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
        self.results: Dict[str, Any] = {}
    
    def compare_metrics(
        self,
        metric_a: np.ndarray,
        metric_b: np.ndarray,
        metric_name: str = 'metric',
        test_type: str = 'auto'
    ) -> Dict:
        """
        Compare a metric between two models.
        
        Args:
            metric_a: Metric values for model A
            metric_b: Metric values for model B
            metric_name: Name of the metric
            test_type: Type of test ('auto', 't-test', 'mann-whitney')
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {metric_name} between models...")
        
        # Determine test type
        if test_type == 'auto':
            # Use Shapiro-Wilk test to check normality
            _, p_a = stats.shapiro(metric_a) if len(metric_a) <= 5000 else (0, 0.05)
            _, p_b = stats.shapiro(metric_b) if len(metric_b) <= 5000 else (0, 0.05)
            
            # If both are normal, use t-test, otherwise Mann-Whitney
            if p_a > 0.05 and p_b > 0.05:
                test_type = 't-test'
            else:
                test_type = 'mann-whitney'
        
        # Run appropriate test
        if test_type == 't-test':
            test = TTest(alpha=self.alpha)
            test_result = test.test(metric_a, metric_b)
        elif test_type == 'mann-whitney':
            test = MannWhitneyTest(alpha=self.alpha)
            test_result = test.test(metric_a, metric_b)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Calculate confidence intervals
        ci_a = confidence_interval(metric_a, confidence=1-self.alpha)
        ci_b = confidence_interval(metric_b, confidence=1-self.alpha)
        
        # Calculate effect size
        effect_size = effect_size_cohens_d(metric_a, metric_b)
        
        result = {
            'metric_name': metric_name,
            'test_result': test_result,
            'confidence_intervals': {
                'model_a': {'lower': ci_a[0], 'upper': ci_a[1]},
                'model_b': {'lower': ci_b[0], 'upper': ci_b[1]}
            },
            'effect_size': {
                'cohens_d': effect_size,
                'interpretation': self._interpret_effect_size(effect_size)
            }
        }
        
        self.results[metric_name] = result
        return result
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to DataFrame.
        
        Returns:
            DataFrame with test results
        """
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for metric_name, result in self.results.items():
            test_result = result['test_result']
            row = {
                'Metric': metric_name,
                'Test': test_result['test'],
                'Statistic': test_result['statistic'],
                'P-Value': test_result['p_value'],
                'Significant': test_result['significant'],
                'Effect Size': result['effect_size']['cohens_d'],
                'Effect Interpretation': result['effect_size']['interpretation']
            }
            rows.append(row)
        
        return pd.DataFrame(rows)