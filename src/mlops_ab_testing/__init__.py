"""
MLOps A/B Testing Framework

An offline and automated framework for simulating A/B tests between different ML model versions,
enabling data-driven and statistically sound evaluation before deployment.
"""

__version__ = "0.1.0"
__author__ = "harshithluc073"
__license__ = "MIT"

from mlops_ab_testing.core.framework import ABTestFramework

__all__ = ["ABTestFramework"]
