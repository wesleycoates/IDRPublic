"""
Feature Engineering Module

This module contains functions for creating new features from existing data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional


def add_polynomial_features(df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
    """
    Create polynomial features for specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to process
    columns : list of str
        List of column names to create polynomial features for
    degree : int, default=2
        The degree of the polynomial features
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added polynomial features
    """
    # Make a copy of the input DataFrame
    result_df = df.copy()
    
    # Validate columns
    valid_columns = [col for col in columns if col in df.columns]
    if len(valid_columns) < len(columns):
        missing = set(columns) - set(valid_columns)
        print(f"Warning: Some columns not found in the data: {missing}")
    
    # Create polynomial features
    for col in valid_columns:
        for d in range(2, degree + 1):
            result_df[f"{col}^{d}"] = result_df[col] ** d
    
    print(f"Added {len(valid_columns) * (degree - 1)} polynomial features")
    return result_df


def add_interaction_features(df: pd.DataFrame, column_pairs: List[tuple]) -> pd.DataFrame:
    """
    Create interaction features for pairs of columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to process
    column_pairs : list of tuples
        List of column pairs to create interaction features for
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added interaction features
    """
    # Make a copy of the input DataFrame
    result_df = df.copy()
    
    # Create interaction features
    valid_pairs = 0
    for col1, col2 in column_pairs:
        if col1 in df.columns and col2 in df.columns:
            result_df[f"{col1}_{col2}_interaction"] = result_df[col1] * result_df[col2]
            valid_pairs += 1
        else:
            print(f"Warning: Column pair ({col1}, {col2}) contains invalid column(s)")
    
    print(f"Added {valid_pairs} interaction features")
    return result_df


def add_ratio_features(df: pd.DataFrame, column_pairs: List[tuple]) -> pd.DataFrame:
    """
    Create ratio features for pairs of columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to process
    column_pairs : list of tuples
        List of column pairs (numerator, denominator) to create ratio features for
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added ratio features
    """
    # Make a copy of the input DataFrame
    result_df = df.copy()
    
    # Create ratio features
    valid_pairs = 0
    for numerator, denominator in column_pairs:
        if numerator in df.columns and denominator in df.columns:
            # Handle division by zero
            result_df[f"{numerator}_to_{denominator}_ratio"] = (
                result_df[numerator] / result_df[denominator].replace(0, np.nan)
            )
            valid_pairs += 1
        else:
            print(f"Warning: Column pair ({numerator}, {denominator}) contains invalid column(s)")
    
    print(f"Added {valid_pairs} ratio features")
    return result_df


def add_binned_features(df: pd.DataFrame, columns: Dict[str, int]) -> pd.DataFrame:
    """
    Create binned features for specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to process
    columns : dict
        Dictionary of column names and number of bins
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added binned features
    """
    # Make a copy of the input DataFrame
    result_df = df.copy()
    
    # Create binned features
    valid_columns = 0
    for col, bins in columns.items():
        if col in df.columns:
            result_df[f"{col}_binned"] = pd.qcut(
                result_df[col], 
                q=bins, 
                labels=False, 
                duplicates='drop'
            )
            valid_columns += 1
        else:
            print(f"Warning: Column {col} not found in the data")
    
    print(f"Added {valid_columns} binned features")
    return result_df


def add_date_features(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """
    Extract features from date columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to process
    date_columns : list of str
        List of column names containing dates
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added date features
    """
    # Make a copy of the input DataFrame
    result_df = df.copy()
    
    # Create date features
    features_added = 0
    for col in date_columns:
        if col in df.columns:
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_dtype(result_df[col]):
                    result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
                
                # Extract features
                result_df[f"{col}_year"] = result_df[col].dt.year
                result_df[f"{col}_month"] = result_df[col].dt.month
                result_df[f"{col}_day"] = result_df[col].dt.day
                result_df[f"{col}_day_of_week"] = result_df[col].dt.dayofweek
                result_df[f"{col}_quarter"] = result_df[col].dt.quarter
                result_df[f"{col}_is_weekend"] = result_df[col].dt.dayofweek >= 5
                
                features_added += 6
            except Exception as e:
                print(f"Error extracting date features from {col}: {e}")
        else:
            print(f"Warning: Column {col} not found in the data")
    
    print(f"Added {features_added} date features")
    return result_df


def add_text_features(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    Extract features from text columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to process
    text_columns : list of str
        List of column names containing text
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added text features
    """
    # Make a copy of the input DataFrame
    result_df = df.copy()
    
    # Create text features
    features_added = 0
    for col in text_columns:
        if col in df.columns:
            try:
                # Convert to string
                result_df[col] = result_df[col].astype(str)
                
                # Character count
                result_df[f"{col}_char_count"] = result_df[col].str.len()
                
                # Word count
                result_df[f"{col}_word_count"] = result_df[col].str.split().str.len()
                
                # Uppercase count
                result_df[f"{col}_uppercase_count"] = result_df[col].str.count(r'[A-Z]')
                
                # Digit count
                result_df[f"{col}_digit_count"] = result_df[col].str.count(r'[0-9]')
                
                features_added += 4
            except Exception as e:
                print(f"Error extracting text features from {col}: {e}")
        else:
            print(f"Warning: Column {col} not found in the data")
    
    print(f"Added {features_added} text features")
    return result_df


def add_aggregation_features(df: pd.DataFrame, group_by: str, agg_columns: List[str], 
                           agg_functions: List[str] = ['mean', 'min', 'max', 'std']) -> pd.DataFrame:
    """
    Add aggregation features by grouping data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to process
    group_by : str
        Column to group by
    agg_columns : list of str
        Columns to aggregate
    agg_functions : list of str, default=['mean', 'min', 'max', 'std']
        Aggregation functions to apply
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added aggregation features
    """
    # Make a copy of the input DataFrame
    result_df = df.copy()
    
    # Validate columns
    if group_by not in df.columns:
        print(f"Warning: Group by column {group_by} not found in the data")
        return result_df
    
    valid_agg_columns = [col for col in agg_columns if col in df.columns]
    if len(valid_agg_columns) < len(agg_columns):
        missing = set(agg_columns) - set(valid_agg_columns)
        print(f"Warning: Some aggregation columns not found in the data: {missing}")
    
    # Skip if no valid aggregation columns
    if not valid_agg_columns:
        print("No valid aggregation columns. Skipping.")
        return result_df
    
    # Create aggregation features
    features_added = 0
    
    # Create dictionary of columns and functions to aggregate
    agg_dict = {col: agg_functions for col in valid_agg_columns}
    
    # Calculate aggregations
    grouped = df.groupby(group_by).agg(agg_dict)
    
    # Flatten multi-index columns
    grouped.columns = [f"{col}_{func}_by_{group_by}" for col, func in grouped.columns]
    
    # Reset index to make the group_by column available for joining
    grouped = grouped.reset_index()
    
    # Join aggregations with original DataFrame
    result_df = pd.merge(result_df, grouped, on=group_by, how='left')
    
    features_added = len(valid_agg_columns) * len(agg_functions)
    print(f"Added {features_added} aggregation features")
    
    return result_df


def add_lag_features(df: pd.DataFrame, id_column: str, date_column: str, 
                   lag_columns: List[str], lag_periods: List[int]) -> pd.DataFrame:
    """
    Add lag features for time series data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to process
    id_column : str
        Column that identifies the entity
    date_column : str
        Column containing the date or time
    lag_columns : list of str
        Columns to create lag features for
    lag_periods : list of int
        Periods to lag
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added lag features
    """
    # Make a copy of the input DataFrame
    result_df = df.copy()
    
    # Validate columns
    required_columns = [id_column, date_column] + lag_columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Some required columns not found in the data: {missing_columns}")
        return result_df
    
    # Convert date column to datetime if not already
    if not pd.api.types.is_datetime64_dtype(result_df[date_column]):
        result_df[date_column] = pd.to_datetime(result_df[date_column], errors='coerce')
    
    # Sort by id and date
    result_df = result_df.sort_values([id_column, date_column])
    
    # Create lag features
    features_added = 0
    
    for col in lag_columns:
        for period in lag_periods:
            # Create lag feature name
            lag_name = f"{col}_lag_{period}"
            
            # Create lag feature
            result_df[lag_name] = result_df.groupby(id_column)[col].shift(period)
            
            features_added += 1
    
    print(f"Added {features_added} lag features")
    return result_df


if __name__ == "__main__":
    # This will execute when you run this script directly
    print("This is a module for feature engineering.")
    print("Example usage:")
    print("from feature_engineering import add_polynomial_features")
    print("enhanced_df = add_polynomial_features(df, ['feature1', 'feature2'])")