"""
Data utilities for MLOps A/B Testing Framework.

This module provides utilities for loading, preprocessing,
and validating data for A/B tests.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def load_data(
    data_path: Union[str, Path],
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Load data from file.
    
    Args:
        data_path: Path to data file
        target_column: Name of target column
        feature_columns: List of feature column names (None = all except target)
        sample_size: Number of samples to load (None = all)
        random_seed: Random seed for sampling
        
    Returns:
        Loaded dataframe
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load based on extension
    if data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    elif data_path.suffix in ['.parquet', '.pq']:
        df = pd.read_parquet(data_path)
    elif data_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(data_path)
    elif data_path.suffix == '.json':
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {data_path}")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_seed)
        logger.info(f"Sampled {sample_size} rows")
    
    # Validate columns
    if target_column and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    if feature_columns:
        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Feature columns not found: {missing_cols}")
    
    return df


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        feature_columns: List of feature columns (None = all except target)
        
    Returns:
        Tuple of (X, y)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    y = df[target_column]
    
    if feature_columns is None:
        # Use all columns except target
        feature_columns = [col for col in df.columns if col != target_column]
    
    X = df[feature_columns]
    
    return X, y


def validate_data(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    check_missing: bool = True,
    check_infinite: bool = True
) -> bool:
    """
    Validate data for common issues.
    
    Args:
        X: Feature dataframe
        y: Target series (optional)
        check_missing: Check for missing values
        check_infinite: Check for infinite values
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    # Check for missing values
    if check_missing:
        missing_features = X.columns[X.isnull().any()].tolist()
        if missing_features:
            n_missing = X[missing_features].isnull().sum().sum()
            logger.warning(
                f"Found {n_missing} missing values in columns: {missing_features}"
            )
    
    # Check for infinite values in numeric columns
    if check_infinite:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        infinite_cols = []
        
        for col in numeric_cols:
            if np.isinf(X[col]).any():
                infinite_cols.append(col)
        
        if infinite_cols:
            logger.warning(f"Found infinite values in columns: {infinite_cols}")
    
    # Validate target if provided
    if y is not None:
        if y.isnull().any():
            logger.warning(f"Found {y.isnull().sum()} missing values in target")
    
    return True


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Summary dataframe
    """
    summary = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null': df.count().values,
        'Null': df.isnull().sum().values,
        'Null %': (df.isnull().sum() / len(df) * 100).values,
        'Unique': df.nunique().values
    })
    
    return summary


def encode_categorical(
    df: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    encoding: str = 'onehot'
) -> pd.DataFrame:
    """
    Encode categorical variables.
    
    Args:
        df: Input dataframe
        categorical_columns: List of categorical columns (None = auto-detect)
        encoding: Encoding method ('onehot' or 'label')
        
    Returns:
        DataFrame with encoded variables
    """
    df = df.copy()
    
    if categorical_columns is None:
        # Auto-detect categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_columns:
        return df
    
    if encoding == 'onehot':
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        logger.info(f"One-hot encoded {len(categorical_columns)} categorical columns")
    
    elif encoding == 'label':
        from sklearn.preprocessing import LabelEncoder
        
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        
        logger.info(f"Label encoded {len(categorical_columns)} categorical columns")
    
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    
    return df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean',
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Handle missing values in dataframe.
    
    Args:
        df: Input dataframe
        strategy: Strategy ('mean', 'median', 'mode', 'drop', 'fill')
        fill_value: Value to fill (for 'fill' strategy)
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if strategy == 'drop':
        df = df.dropna()
        logger.info("Dropped rows with missing values")
    
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        logger.info("Filled missing values with mean")
    
    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        logger.info("Filled missing values with median")
    
    elif strategy == 'mode':
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        logger.info("Filled missing values with mode")
    
    elif strategy == 'fill':
        if fill_value is None:
            raise ValueError("fill_value required for 'fill' strategy")
        df = df.fillna(fill_value)
        logger.info(f"Filled missing values with {fill_value}")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df