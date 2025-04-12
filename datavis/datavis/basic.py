"""
Basic visualization functions.

This module contains fundamental visualization functions for common plot types.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union

from .core import save_figure


def plot_numeric_distribution(df: pd.DataFrame, 
                             columns: Optional[List[str]] = None, 
                             n_cols: int = 3, 
                             figsize: Optional[Tuple[int, int]] = None,
                             bins: int = 30, 
                             kde: bool = True, 
                             color: Optional[str] = None,
                             save_path: Optional[str] = None):
    """
    Plot histograms for numeric columns in the DataFrame with enhanced options.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to visualize
    columns : list of str, optional
        Specific columns to visualize. If None, all numeric columns are used.
    n_cols : int, default=3
        Number of columns in the subplot grid
    figsize : tuple of int, optional
        Figure size (width, height) in inches
    bins : int, default=30
        Number of bins in histograms
    kde : bool, default=True
        Whether to overlay a kernel density estimate
    color : str, optional
        Color for the histograms
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Get numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Validate column names
    valid_columns = []
    for col in columns:
        if col in df.columns:
            valid_columns.append(col)
        else:
            # Try to find a match by converting DataFrame columns to lowercase
            for df_col in df.columns:
                if col.lower() == df_col.lower():
                    valid_columns.append(df_col)
                    break
    
    if len(valid_columns) < len(columns):
        missing_count = len(columns) - len(valid_columns)
        print(f"Warning: {missing_count} column(s) not found in DataFrame. Using {len(valid_columns)} valid columns.")
    
    # Update columns to use only valid ones
    columns = valid_columns
    
    # Skip if no numeric columns
    if not columns:
        print("No numeric columns to visualize")
        return None
    
    # Calculate grid size
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    # Create figure
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Plot each column
    for i, col in enumerate(columns):
        if i < len(axes):
            # Add descriptive statistics
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()
            
            # Create histogram with KDE
            sns.histplot(df[col], kde=kde, bins=bins, ax=axes[i], color=color)
            
            # Add vertical lines for mean and median
            axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                           label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='green', linestyle='-', alpha=0.8, 
                          label=f'Median: {median_val:.2f}')
            
            # Add title and labels with stats
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(f'{col} (σ={std_val:.2f})')
            axes[i].set_ylabel('Frequency')
            axes[i].legend(loc='best')
    
    # Hide unused subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig


def plot_categorical_counts(df: pd.DataFrame, 
                          columns: Optional[List[str]] = None,
                          n_cols: int = 2, 
                          figsize: Optional[Tuple[int, int]] = None,
                          max_categories: int = 20, 
                          orientation: str = 'vertical',
                          palette: str = 'viridis', 
                          show_percentages: bool = True,
                          save_path: Optional[str] = None):
    """
    Plot count plots for categorical columns in the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to visualize
    columns : list of str, optional
        Specific columns to visualize. If None, all categorical and boolean columns are used.
    n_cols : int, default=2
        Number of columns in the subplot grid
    figsize : tuple of int, optional
        Figure size (width, height) in inches
    max_categories : int, default=20
        Maximum number of categories to show in each plot. If exceeded, shows only the most frequent.
    orientation : str, default='vertical'
        Orientation of the bars: 'vertical' or 'horizontal'
    palette : str, default='viridis'
        Color palette to use
    show_percentages : bool, default=True
        Whether to display percentage values on the bars
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Get categorical columns if not specified
    if columns is None:
        # Include object, category, and boolean dtypes
        columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Skip if no categorical columns
    if not columns:
        print("No categorical columns to visualize")
        return None
    
    # Calculate grid size
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    # Create figure
    if figsize is None:
        figsize = (7 * n_cols, 5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Plot each column
    for i, col in enumerate(columns):
        if i < len(axes):
            # Get value counts
            value_counts = df[col].value_counts()
            total_count = value_counts.sum()
            
            # Limit categories if needed
            if len(value_counts) > max_categories:
                print(f"Column '{col}' has {len(value_counts)} categories. Showing top {max_categories}.")
                value_counts = value_counts.head(max_categories)
                
                # Calculate "Other" category if needed
                other_count = total_count - value_counts.sum()
                if other_count > 0:
                    value_counts = pd.concat([value_counts, pd.Series([other_count], index=["Other"])])
            
            # Calculate percentages
            percentages = (value_counts / total_count * 100).round(1)
            
            # Plot horizontally or vertically
            if orientation == 'horizontal':
                # Sort values for better visualization
                value_counts = value_counts.sort_values(ascending=True)
                percentages = percentages[value_counts.index]
                
                bars = sns.barplot(x=value_counts.index, y=value_counts.values, hue=value_counts.index, ax=ax, palette=palette, legend=False)
                
                # Add percentage labels if requested
                if show_percentages:
                    for j, (count, percentage) in enumerate(zip(value_counts, percentages)):
                        axes[i].text(count + (total_count * 0.01), j, 
                                   f'{percentage}%', va='center')
                
                axes[i].set_xlabel('Count')
                axes[i].set_ylabel(col)
                # Add appropriate title with total count
                axes[i].set_title(f'{col} Distribution (n={total_count})')
                
            else:  # vertical
                bars = sns.barplot(x=value_counts.index, y=value_counts.values, hue=value_counts.index,
                                 ax=axes[i], palette=palette, legend=False)
                
                # Add percentage labels if requested
                if show_percentages:
                    for j, (count, percentage) in enumerate(zip(value_counts, percentages)):
                        axes[i].text(j, count + (total_count * 0.01), 
                                   f'{percentage}%', ha='center', va='bottom')
                
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Count')
                # Add appropriate title with total count
                axes[i].set_title(f'{col} Distribution (n={total_count})')
                
                # Rotate x labels if there are multiple categories
                if len(value_counts) > 3:
                    plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    # Hide unused subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig


def plot_boxplots(df: pd.DataFrame, 
                numeric_cols: Optional[List[str]] = None,
                groupby_col: Optional[str] = None, 
                n_cols: int = 2, 
                figsize: Optional[Tuple[int, int]] = None, 
                palette: str = 'Set2',
                save_path: Optional[str] = None):
    """
    Create box plots for numeric columns, optionally grouped by a categorical column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to visualize
    numeric_cols : list of str, optional
        Numeric columns to visualize. If None, all numeric columns are used.
    groupby_col : str, optional
        Categorical column to group by. If None, simple boxplots are created.
    n_cols : int, default=2
        Number of columns in the subplot grid
    figsize : tuple of int, optional
        Figure size (width, height) in inches
    palette : str, default='Set2'
        Color palette for the plots
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Get numeric columns if not specified
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Skip if no numeric columns
    if not numeric_cols:
        print("No numeric columns to visualize")
        return None
    
    # Calculate grid size
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    # Create figure
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Plot each column
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            if groupby_col:
                # Create grouped boxplot
                sns.boxplot(x=groupby_col, y=col, data=df, ax=axes[i], hue=groupby_col, palette=palette, legend=False)
                
                # Rotate x-axis labels if necessary
                if len(df[groupby_col].unique()) > 4:
                    plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
            else:
                # Create simple boxplot
                sns.boxplot(y=df[col], ax=axes[i], 
                          color=sns.color_palette(palette)[i % len(sns.color_palette(palette))])
                axes[i].set_xlabel('')
            
            # Add title
            if groupby_col:
                axes[i].set_title(f'Distribution of {col} by {groupby_col}')
            else:
                axes[i].set_title(f'Distribution of {col}')
                
            # Add statistics text
            stats_text = (
                f'Mean: {df[col].mean():.2f}\n'
                f'Median: {df[col].median():.2f}\n'
                f'Std: {df[col].std():.2f}'
            )
            axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig


def plot_time_series(df: pd.DataFrame, 
                    date_col: str, 
                    value_cols: List[str],
                    groupby: Optional[str] = None, 
                    freq: Optional[str] = None,
                    agg_func: str = 'mean', 
                    figsize: Tuple[int, int] = (12, 6),
                    title: str = 'Time Series Plot', 
                    ylabel: str = 'Value',
                    plot_type: str = 'line', 
                    save_path: Optional[str] = None):
    """
    Plot time series data with flexible options.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to visualize
    date_col : str
        Name of the column containing dates/timestamps
    value_cols : list of str
        Names of columns containing values to plot
    groupby : str, optional
        If provided, group by this column (useful for faceting)
    freq : str, optional
        Frequency for resampling time series: 'D' (daily), 'W' (weekly), 'M' (monthly), etc.
    agg_func : str, default='mean'
        Aggregation function to use when resampling: 'mean', 'sum', 'min', 'max', etc.
    figsize : tuple of int, default=(12, 6)
        Figure size (width, height) in inches
    title : str, default='Time Series Plot'
        Title of the plot
    ylabel : str, default='Value'
        Label for the y-axis
    plot_type : str, default='line'
        Type of plot: 'line', 'area', or 'bar'
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Make a copy to avoid modifying the original dataframe
    plot_df = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(plot_df[date_col]):
        plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    
    # Set date as index for easier resampling
    plot_df.set_index(date_col, inplace=True)
    
    # Resample if frequency is specified
    if freq:
        # Handle groupby with resampling
        if groupby:
            result_dfs = []
            for group_name, group_df in plot_df.groupby(groupby):
                # Resample each group
                resampled = getattr(group_df[value_cols].resample(freq), agg_func)()
                resampled[groupby] = group_name  # Add back the group identifier
                result_dfs.append(resampled)
            
            plot_df = pd.concat(result_dfs)
        else:
            # Simple resampling without groups
            plot_df = getattr(plot_df[value_cols].resample(freq), agg_func)()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data based on plot_type
    if groupby and groupby in plot_df.columns:
        # Plot with groups as different colors
        groups = plot_df[groupby].unique()
        
        for group in groups:
            group_data = plot_df[plot_df[groupby] == group]
            
            for col in value_cols:
                if plot_type == 'line':
                    group_data[col].plot(ax=ax, label=f'{col} - {group}')
                elif plot_type == 'area':
                    group_data[col].plot.area(ax=ax, alpha=0.5, label=f'{col} - {group}')
                elif plot_type == 'bar':
                    group_data[col].plot.bar(ax=ax, alpha=0.7, label=f'{col} - {group}')
    else:
        # Plot without grouping
        for col in value_cols:
            if plot_type == 'line':
                plot_df[col].plot(ax=ax, label=col)
            elif plot_type == 'area':
                plot_df[col].plot.area(ax=ax, alpha=0.5, label=col)
            elif plot_type == 'bar':
                plot_df[col].plot.bar(ax=ax, alpha=0.7, label=col)
    
    # Set labels and title
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    
    # Format x-axis based on plot type
    if plot_type == 'bar':
        plt.xticks(rotation=45)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig


def plot_heatmap(data: Union[pd.DataFrame, np.ndarray], 
                title: str = 'Heatmap',
                figsize: Tuple[int, int] = (10, 8), 
                cmap: str = 'viridis',
                annot: bool = True, 
                fmt: str = '.2f', 
                linewidths: float = 0.5,
                xticklabels: Optional[List[str]] = None, 
                yticklabels: Optional[List[str]] = None,
                vmin: Optional[float] = None, 
                vmax: Optional[float] = None,
                save_path: Optional[str] = None):
    """
    Create a flexible heatmap for any 2D data.
    
    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        The data to visualize
    title : str, default='Heatmap'
        Title of the plot
    figsize : tuple of int, default=(10, 8)
        Figure size (width, height) in inches
    cmap : str, default='viridis'
        Colormap to use
    annot : bool, default=True
        Whether to annotate the heatmap with values
    fmt : str, default='.2f'
        Format string for annotations
    linewidths : float, default=0.5
        Width of the lines that divide cells
    xticklabels : list of str, optional
        Labels for the x-axis
    yticklabels : list of str, optional
        Labels for the y-axis
    vmin : float, optional
        Minimum value for colormap scaling
    vmax : float, optional
        Maximum value for colormap scaling
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(data, annot=annot, fmt=fmt, linewidths=linewidths, cmap=cmap,
                xticklabels=xticklabels, yticklabels=yticklabels,
                vmin=vmin, vmax=vmax, ax=ax)
    
    # Set title
    ax.set_title(title, fontsize=16)
    
    # Rotate x-axis labels if there are many columns
    if data.shape[1] > 8:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig