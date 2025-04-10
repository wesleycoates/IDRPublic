import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple


def set_visualization_style():
    """Set the default visualization style for plots"""
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12


def plot_numeric_distribution(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                              n_cols: int = 3, figsize: Tuple[int, int] = None):
    """
    Plot histograms for numeric columns in the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to visualize
    columns : list of str, optional
        Specific columns to visualize. If None, all numeric columns are used.
    n_cols : int, default=3
        Number of columns in the subplot grid
    figsize : tuple of int, optional
        Figure size (width, height) in inches
    """
    # Get numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Validate column names
    valid_columns = []
    column_mapping = {}
    
    for col in columns:
        if col in df.columns:
            valid_columns.append(col)
        else:
            # Try to find a match by converting DataFrame columns to lowercase with underscores
            for df_col in df.columns:
                if col == df_col.lower().replace(' ', '_'):
                    column_mapping[col] = df_col
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
        return
    
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
            sns.histplot(df[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                           figsize: Tuple[int, int] = (12, 10)):
    """
    Plot a correlation matrix for numeric columns in the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to visualize
    columns : list of str, optional
        Specific columns to include. If None, all numeric columns are used.
    figsize : tuple of int, default=(12, 10)
        Figure size (width, height) in inches
    """
    # Get numeric columns if not specified
    if columns is None:
        df_numeric = df.select_dtypes(include=['int64', 'float64'])
    else:
        df_numeric = df[columns]
    
    # Skip if no numeric columns
    if df_numeric.empty:
        print("No numeric columns to visualize")
        return
    
    # Calculate correlation matrix
    corr = df_numeric.corr()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', 
                vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
    
    plt.title('Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names: List[str], importance_values: List[float], 
                           title: str = 'Feature Importance', figsize: Tuple[int, int] = (12, 8)):
    """
    Plot feature importance from a machine learning model.
    
    Parameters:
    -----------
    feature_names : list of str
        Names of the features
    importance_values : list of float
        Importance values for each feature
    title : str, default='Feature Importance'
        Title of the plot
    figsize : tuple of int, default=(12, 8)
        Figure size (width, height) in inches
    """
    # Create DataFrame for plotting
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot bar chart
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_scatter_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                       hue: Optional[str] = None, figsize: Tuple[int, int] = None):
    """
    Plot a scatter matrix for numeric columns in the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to visualize
    columns : list of str, optional
        Specific columns to include. If None, all numeric columns are used.
    hue : str, optional
        Column to use for coloring points
    figsize : tuple of int, optional
        Figure size (width, height) in inches
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
        return
    
    # Calculate default figure size if not provided
    if figsize is None:
        figsize = (3 * len(columns), 3 * len(columns))
    
    # Create scatter matrix
    sns.set(style="ticks")
    sns.pairplot(df[columns + ([hue] if hue else [])], hue=hue, diag_kind="kde")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # This will execute when you run this script directly
    print("This is a module for data visualization.")
    print("Example usage:")
    print("from visualization import plot_numeric_distribution, plot_correlation_matrix")
    print("plot_numeric_distribution(df)")
    print("plot_correlation_matrix(df)")