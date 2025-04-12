"""
Statistical visualization functions.

This module contains functions for visualizing statistical relationships and distributions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union

from .core import save_figure


def plot_correlation_matrix(df: pd.DataFrame, 
                           columns: Optional[List[str]] = None, 
                           figsize: Tuple[int, int] = (12, 10), 
                           method: str = 'pearson',
                           mask_upper: bool = True, 
                           cmap: str = 'coolwarm',
                           save_path: Optional[str] = None):
    """
    Plot a correlation matrix for numeric columns in the DataFrame with enhanced options.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to visualize
    columns : list of str, optional
        Specific columns to include. If None, all numeric columns are used.
    figsize : tuple of int, default=(12, 10)
        Figure size (width, height) in inches
    method : str, default='pearson'
        Method of correlation: 'pearson', 'kendall', 'spearman'
    mask_upper : bool, default=True
        Whether to mask the upper triangle of the correlation matrix
    cmap : str, default='coolwarm'
        Colormap to use for the heatmap
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Get numeric columns if not specified
    if columns is None:
        df_numeric = df.select_dtypes(include=['int64', 'float64'])
    else:
        df_numeric = df[columns]
    
    # Skip if no numeric columns
    if df_numeric.empty:
        print("No numeric columns to visualize")
        return None
    
    # Calculate correlation matrix
    corr = df_numeric.corr(method=method)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for upper triangle if requested
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None
    
    # Create heatmap
    sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, 
                vmin=-1, vmax=1, fmt=".2f", linewidths=0.5,
                annot_kws={"size": 10 if len(corr) > 10 else 12},
                ax=ax)
    
    plt.title(f'Correlation Matrix ({method.capitalize()})', fontsize=16)
    
    # Rotate x-axis labels if there are many features
    if len(corr) > 8:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig


def plot_feature_importance(feature_names: List[str], 
                           importance_values: List[float], 
                           title: str = 'Feature Importance', 
                           figsize: Tuple[int, int] = (12, 8),
                           color: str = '#1f77b4', 
                           top_n: Optional[int] = None,
                           horizontal: bool = True, 
                           show_values: bool = True,
                           save_path: Optional[str] = None):
    """
    Plot feature importance from a machine learning model with enhanced options.
    
    Parameters
    ----------
    feature_names : list of str
        Names of the features
    importance_values : list of float
        Importance values for each feature
    title : str, default='Feature Importance'
        Title of the plot
    figsize : tuple of int, default=(12, 8)
        Figure size (width, height) in inches
    color : str, default='#1f77b4'
        Color for the bars
    top_n : int, optional
        If provided, show only the top N most important features
    horizontal : bool, default=True
        Whether to use horizontal bars (True) or vertical bars (False)
    show_values : bool, default=True
        Whether to display the importance values on the bars
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Create DataFrame for plotting
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Limit to top N features if specified
    if top_n is not None and top_n < len(feature_importance_df):
        feature_importance_df = feature_importance_df.head(top_n)
        title = f'Top {top_n} Feature Importance'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose bar orientation based on horizontal parameter
    if horizontal:
        # Sort for horizontal display (most important at the top)
        feature_importance_df = feature_importance_df.sort_values('Importance')
        
        # Plot bar chart
        bars = ax.barh(y='Feature', width='Importance', data=feature_importance_df, color=color)
        
        # Add value labels if requested
        if show_values:
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width * 1.01  # Slightly to the right of the bar
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                       va='center', fontsize=10)
        
        ax.set_xlabel('Importance', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)
    else:
        # Plot vertical bar chart
        bars = ax.bar(x='Feature', height='Importance', data=feature_importance_df, color=color)
        
        # Add value labels if requested
        if show_values:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height * 1.01, f'{height:.3f}', 
                       ha='center', fontsize=10)
        
        ax.set_xlabel('Feature', fontsize=14)
        ax.set_ylabel('Importance', fontsize=14)
        plt.xticks(rotation=45, ha='right')
    
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig


def plot_scatter_matrix(df: pd.DataFrame, 
                       columns: Optional[List[str]] = None, 
                       hue: Optional[str] = None, 
                       figsize: Optional[Tuple[int, int]] = None,
                       diag_kind: str = "kde", 
                       corner: bool = False,
                       markers: Optional[str] = None, 
                       height: float = 2.5,
                       save_path: Optional[str] = None):
    """
    Plot a scatter matrix for numeric columns in the DataFrame with enhanced options.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to visualize
    columns : list of str, optional
        Specific columns to include. If None, all numeric columns are used (limited to first 5)
    hue : str, optional
        Column to use for coloring points
    figsize : tuple of int, optional
        Figure size (width, height) in inches. If None, calculated based on number of columns.
    diag_kind : str, default="kde"
        Kind of plot to use on the diagonal: 'hist' or 'kde'
    corner : bool, default=False
        If True, plots only the lower triangle of the scatter matrix
    markers : str, optional
        Marker style for the scatter plot points
    height : float, default=2.5
        Height (in inches) of each facet
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    grid : seaborn.axisgrid.PairGrid
        The generated PairGrid object for further customization
    """
    # Get numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Limit to maximum 5 columns to avoid overcrowded plots
        if len(columns) > 5:
            print(f"Limiting scatter matrix to first 5 numeric columns. Original columns: {len(columns)}")
            columns = columns[:5]
    
    # Skip if no numeric columns
    if not columns:
        print("No numeric columns to visualize")
        return None
    
    # Create PairGrid with options
    grid = sns.pairplot(
        df[columns + ([hue] if hue else [])], 
        hue=hue,
        diag_kind=diag_kind,
        corner=corner,
        markers=markers,
        height=height,
        plot_kws={'alpha': 0.7}  # Add some transparency to scatter points
    )
    
    # Add title if not corner plot (title placement can be tricky in corner plots)
    if not corner:
        grid.fig.suptitle('Scatter Plot Matrix', fontsize=16, y=1.02)
    
    # Adjust figure size if provided
    if figsize is not None:
        grid.fig.set_size_inches(figsize)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        grid.fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return grid


def plot_feature_distributions_by_target(df: pd.DataFrame, 
                                       features: List[str], 
                                       target: str, 
                                       bins: int = 30,
                                       figsize: Optional[Tuple[int, int]] = None,
                                       n_cols: int = 3, 
                                       save_path: Optional[str] = None):
    """
    Plot distributions of features grouped by a categorical target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to visualize
    features : list of str
        Numeric columns to visualize
    target : str
        Categorical target variable
    bins : int, default=30
        Number of bins for histograms
    figsize : tuple of int, optional
        Figure size (width, height) in inches
    n_cols : int, default=3
        Number of columns in the subplot grid
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Check if target is categorical
    if df[target].nunique() > 10:
        print(f"Warning: Target variable '{target}' has {df[target].nunique()} unique values. "
              f"Consider using a binary or categorical target.")
    
    # Calculate grid size
    n_rows = (len(features) + n_cols - 1) // n_cols
    
    # Set default figure size if not provided
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Plot each feature
    for i, feature in enumerate(features):
        if i < len(axes):
            # Create histograms grouped by target
            sns.histplot(data=df, x=feature, hue=target, 
                       element="step", kde=True, bins=bins, 
                       common_norm=False, alpha=0.6, ax=axes[i])
            
            axes[i].set_title(f'Distribution of {feature} by {target}')
            
            # Improve legend visibility if there are many classes
            if df[target].nunique() > 5:
                axes[i].legend(fontsize='small', title=target, loc='best')
    
    # Hide unused subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig


def create_correlation_network(df: pd.DataFrame, 
                             threshold: float = 0.5, 
                             figsize: Tuple[int, int] = (10, 10),
                             node_size_factor: float = 3000, 
                             title: str = "Feature Correlation Network",
                             save_path: Optional[str] = None):
    """
    Create a network visualization of correlations between features.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to visualize (numeric columns only)
    threshold : float, default=0.5
        Minimum absolute correlation value to include in the graph
    figsize : tuple of int, default=(10, 10)
        Figure size (width, height) in inches
    node_size_factor : float, default=3000
        Factor to scale node sizes
    title : str, default="Feature Correlation Network"
        Title of the plot
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    try:
        import networkx as nx
    except ImportError:
        print("This function requires networkx. Install with: pip install networkx")
        return None
    
    # Get numeric columns only
    df_numeric = df.select_dtypes(include=['int64', 'float64'])
    
    # Skip if no numeric columns
    if df_numeric.empty:
        print("No numeric columns to visualize")
        return None
    
    # Calculate correlation matrix
    corr = df_numeric.corr()
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for column in corr.columns:
        # Use variance as node size
        var = df_numeric[column].var()
        G.add_node(column, size=var)
    
    # Add edges for correlations above threshold
    for i, col_i in enumerate(corr.columns):
        for j, col_j in enumerate(corr.columns):
            if i < j:  # Only use upper triangle of the correlation matrix
                corr_val = corr.iloc[i, j]
                if abs(corr_val) >= threshold:
                    G.add_edge(col_i, col_j, weight=abs(corr_val), 
                              color='red' if corr_val < 0 else 'green')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up layout
    pos = nx.spring_layout(G, seed=42)
    
    # Get node sizes based on variance
    node_sizes = [node_size_factor * G.nodes[node]['size'] for node in G.nodes]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, alpha=0.8,
                          node_color='lightblue', linewidths=1, edgecolors='black')
    
    # Draw edges with different colors based on positive/negative correlation
    edges_pos = [(u, v) for u, v, d in G.edges(data=True) if d['color'] == 'green']
    edges_neg = [(u, v) for u, v, d in G.edges(data=True) if d['color'] == 'red']
    
    # Get edge weights for width
    edge_weights_pos = [G[u][v]['weight'] * 3 for u, v in edges_pos]
    edge_weights_neg = [G[u][v]['weight'] * 3 for u, v in edges_neg]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges_pos, width=edge_weights_pos,
                          alpha=0.7, edge_color='green')
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges_neg, width=edge_weights_neg,
                          alpha=0.7, edge_color='red', style='dashed')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Positive Correlation'),
        Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Negative Correlation')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set title and turn off axis
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    # Add threshold information
    plt.figtext(0.1, 0.02, f'Correlation threshold: |r| ≥ {threshold}', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig