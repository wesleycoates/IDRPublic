"""
Utility functions for data visualization.

This module contains helper functions for data visualization and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from typing import Optional, List, Tuple, Dict, Union

from .core import save_figure


def visualize_dataset_summary(df: pd.DataFrame, 
                            title: str = "Dataset Summary Visualization",
                            figsize: Tuple[int, int] = (18, 12), 
                            max_categories: int = 10,
                            save_path: Optional[str] = None):
    """
    Create a comprehensive visualization dashboard for a dataset, showing key statistics,
    data types, missing values, and distributions.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to visualize
    title : str, default="Dataset Summary Visualization"
        Title of the dashboard
    figsize : tuple of int, default=(18, 12)
        Figure size (width, height) in inches
    max_categories : int, default=10
        Maximum number of categories to show for categorical variables
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Create figure with a complex grid layout
    fig = plt.figure(figsize=figsize)
    
    # Define grid layout
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 2, 2])
    
    # 1. Dataset shape and data types (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    # Create text summary
    n_rows, n_cols = df.shape
    dtypes_count = df.dtypes.value_counts()
    dtype_summary = ", ".join([f"{count} {dtype}" for dtype, count in dtypes_count.items()])
    
    summary_text = (
        f"Dataset Shape: {n_rows} rows × {n_cols} columns\n\n"
        f"Data Types: {dtype_summary}\n\n"
        f"Memory Usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB"
    )
    
    ax1.text(0.05, 0.95, summary_text, va='top', fontsize=12,
            transform=ax1.transAxes)
    
    # 2. Missing values (top-middle)
    ax2 = fig.add_subplot(gs[0, 1:3])
    
    # Calculate missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_percent = missing / len(df) * 100
    
    if len(missing) > 0:
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Values': missing.values,
            'Percent': missing_percent.values
        }).sort_values('Missing Values', ascending=False)
        
        # Plot missing values
        sns.barplot(x='Column', y='Percent', data=missing_df, ax=ax2, palette='YlOrRd')
        ax2.set_title('Missing Values (%)', fontsize=14)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_ylabel('Percent Missing')
    else:
        ax2.text(0.5, 0.5, "No Missing Values", ha='center', va='center', fontsize=14)
        ax2.set_title('Missing Values', fontsize=14)
        ax2.axis('off')
    
    # 3. Correlation heatmap of numeric variables (top-right)
    ax3 = fig.add_subplot(gs[0, 3])
    
    # Get numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr()
        
        # If too many variables, show a condensed version
        if len(corr) > 10:
            # Find most correlated pairs
            corr_unstack = corr.abs().unstack()
            corr_unstack = corr_unstack[corr_unstack < 1]  # Remove self-correlations
            top_corr = corr_unstack.nlargest(10)
            
            ax3.text(0.5, 0.5, "Too many variables for heatmap.\nTop correlations:", 
                    ha='center', va='center', fontsize=12)
            
            for i, ((var1, var2), val) in enumerate(top_corr.items()):
                ax3.text(0.1, 0.3 - i*0.06, f"{var1} — {var2}: {corr.loc[var1, var2]:.2f}", 
                        va='center', fontsize=10)
        else:
            # Create heatmap
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                      linewidths=0.5, vmin=-1, vmax=1, ax=ax3, annot_kws={"size": 8})
        
        ax3.set_title('Correlation Matrix', fontsize=14)
    else:
        ax3.text(0.5, 0.5, "Insufficient numeric\ncolumns for correlation", 
                ha='center', va='center', fontsize=14)
        ax3.set_title('Correlation Matrix', fontsize=14)
        ax3.axis('off')
    
    # 4. Numeric distributions (middle row)
    ax4 = fig.add_subplot(gs[1, :])
    
    if len(numeric_df.columns) > 0:
        # Create a subplot for each numeric column
        n_numeric = len(numeric_df.columns)
        n_cols_vis = min(n_numeric, 5)  # Limit to 5 columns at most
        
        # Create subplots
        subgs = gridspec.GridSpecFromSubplotSpec(1, n_cols_vis, subplot_spec=gs[1, :])
        
        for i, col in enumerate(numeric_df.columns[:n_cols_vis]):
            subax = fig.add_subplot(subgs[i])
            sns.histplot(df[col], kde=True, ax=subax)
            subax.set_title(col, fontsize=12)
            
            # Add basic stats
            stats_text = (
                f"Mean: {df[col].mean():.2f}\n"
                f"Median: {df[col].median():.2f}\n"
                f"Std: {df[col].std():.2f}"
            )
            
            subax.text(0.95, 0.95, stats_text, transform=subax.transAxes,
                      va='top', ha='right', fontsize=9,
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
            
            # Set tighter bounds for x-axis
            q1, q3 = df[col].quantile([0.05, 0.95])
            iqr = q3 - q1
            subax.set_xlim([q1 - 1.5*iqr, q3 + 1.5*iqr])
    else:
        ax4.text(0.5, 0.5, "No numeric columns to visualize", ha='center', va='center', fontsize=14)
        ax4.axis('off')
    
    # 5. Categorical distributions (bottom row)
    ax5 = fig.add_subplot(gs[2, :])
    
    # Get categorical columns
    cat_df = df.select_dtypes(include=['object', 'category', 'bool'])
    
    if len(cat_df.columns) > 0:
        # Create a subplot for each categorical column (up to 4)
        n_cat = len(cat_df.columns)
        n_cols_vis = min(n_cat, 4)  # Limit to 4 columns at most
        
        # Create subplots
        subgs = gridspec.GridSpecFromSubplotSpec(1, n_cols_vis, subplot_spec=gs[2, :])
        
        for i, col in enumerate(cat_df.columns[:n_cols_vis]):
            subax = fig.add_subplot(subgs[i])
            
            # Get value counts
            value_counts = df[col].value_counts()
            
            # Limit to top categories if too many
            if len(value_counts) > max_categories:
                other_count = value_counts.iloc[max_categories:].sum()
                value_counts = value_counts.iloc[:max_categories]
                value_counts['Other'] = other_count
            
            # Calculate percentages
            total = value_counts.sum()
            percentages = (value_counts / total * 100).round(1)
            
            # Create bar chart
            bars = subax.bar(value_counts.index, value_counts.values, 
                           color=sns.color_palette('viridis', len(value_counts)))
            
            # Add percentage labels
            for j, (bar, percentage) in enumerate(zip(bars, percentages)):
                height = bar.get_height()
                subax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         f'{percentage}%', ha='center', va='bottom', fontsize=8)
            
            # Set title and format x-axis
            subax.set_title(col, fontsize=12)
            # First get the current ticks and set them explicitly
            ticks = subax.get_xticks()
            subax.set_xticks(ticks)
            # Then set the labels
            subax.set_xticklabels(subax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            subax.set_ylabel('Count')
            
            # Add count and unique info
            info_text = f"Count: {len(df[col])}\nUnique: {df[col].nunique()}"
            subax.text(0.95, 0.95, info_text, transform=subax.transAxes,
                      va='top', ha='right', fontsize=9,
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    else:
        ax5.text(0.5, 0.5, "No categorical columns to visualize", ha='center', va='center', fontsize=14)
        ax5.axis('off')
    
    # Add overall title
    fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig