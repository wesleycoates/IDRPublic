"""
Core functionality for data visualization.

This module contains core functions and utilities used across the visualization package.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import colorsys


def set_visualization_style(style="whitegrid", context="notebook", 
                           palette="deep", font_scale=1.2):
    """
    Set the default visualization style for plots with customization options.
    
    Parameters
    ----------
    style : str, default="whitegrid"
        The style of the plots. Options include: "darkgrid", "whitegrid", "dark", "white", "ticks"
    context : str, default="notebook"
        The context of the plots. Options include: "paper", "notebook", "talk", "poster"
    palette : str, default="deep"
        Color palette to use. See seaborn's documentation for options.
    font_scale : float, default=1.2
        Scaling factor for font sizes.
        
    Returns
    -------
    dict
        The configured parameters for reference
    """
    sns.set_theme(style=style, context=context, palette=palette, font_scale=font_scale)
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12
    # Improve readability of plot elements
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    # Set default color cycle
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=sns.color_palette(palette))
    
    # Return configured parameters for reference
    return {
        "style": style,
        "context": context,
        "palette": palette,
        "font_scale": font_scale,
        "figure.figsize": plt.rcParams["figure.figsize"]
    }


def adjust_color_brightness(color, index, total):
    """
    Adjust the brightness of a color based on index in a sequence.
    
    Parameters
    ----------
    color : str
        Base color (color name or hex code)
    index : int
        Index in the sequence
    total : int
        Total number of colors needed
    
    Returns
    -------
    str
        Adjusted color in hex format
    """
    # Convert color to RGB
    try:
        rgb = plt.matplotlib.colors.to_rgb(color)
    except:
        # Default to blue if color cannot be converted
        rgb = plt.matplotlib.colors.to_rgb('blue')
    
    # Adjust brightness
    h, l, s = colorsys.rgb_to_hls(*rgb)
    
    # Calculate lighter/darker variations
    brightness_factor = 0.7 + (0.6 * index / (total - 1 if total > 1 else 1))
    
    # Adjust lightness
    l = min(1.0, l * brightness_factor)
    
    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    
    # Return as hex color
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


def save_figure(fig, save_path, dpi=300, bbox_inches='tight'):
    """
    Save a figure to disk if a path is provided.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    save_path : str or None
        Path where the figure should be saved. If None, does nothing.
    dpi : int, default=300
        Resolution of the saved figure
    bbox_inches : str, default='tight'
        Bounding box settings for the saved figure
        
    Returns
    -------
    bool
        True if the figure was saved, False otherwise
    """
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
        return True
    return False


def get_columns_by_type(df, include_numeric=True, include_categorical=True, 
                       include_datetime=True, max_cardinality=None):
    """
    Get columns from a DataFrame based on their data type.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze
    include_numeric : bool, default=True
        Whether to include numeric columns
    include_categorical : bool, default=True
        Whether to include categorical columns
    include_datetime : bool, default=True
        Whether to include datetime columns
    max_cardinality : int, optional
        If provided, categorical columns with more unique values than this will be excluded
        
    Returns
    -------
    dict
        Dictionary with types as keys and lists of column names as values
    """
    import pandas as pd
    
    result = {}
    
    if include_numeric:
        result['numeric'] = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if include_categorical:
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Filter by cardinality if specified
        if max_cardinality is not None:
            cat_cols = [col for col in cat_cols if df[col].nunique() <= max_cardinality]
            
        result['categorical'] = cat_cols
    
    if include_datetime:
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        result['datetime'] = datetime_cols
    
    return result