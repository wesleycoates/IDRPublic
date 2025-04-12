import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union, Any, Callable
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
# And any other libraries your specific functions need

def create_stacked_area_chart(df: pd.DataFrame, 
                            x_col: str, 
                            y_cols: List[str],
                            figsize: Tuple[int, int] = (12, 8),
                            title: str = 'Stacked Area Chart',
                            normalize: bool = False,
                            alpha: float = 0.8,
                            colors: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
    """
    Create a stacked area chart for visualizing composition over time or categories.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to visualize
    x_col : str
        Column name for x-axis values (often a date/time column)
    y_cols : list of str
        Column names for the stacked areas
    figsize : tuple of int, default=(12, 8)
        Figure size (width, height) in inches
    title : str, default='Stacked Area Chart'
        Title of the plot
    normalize : bool, default=False
        Whether to normalize values to show percentages instead of absolute values
    alpha : float, default=0.8
        Transparency of the areas
    colors : list of str, optional
        Colors for each area. If None, uses default color cycle.
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by x-axis if it's a time column
    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
        df = df.sort_values(x_col)
    
    # Prepare data
    x = df[x_col]
    ys = [df[col] for col in y_cols]
    
    # Normalize if requested
    if normalize:
        # Calculate sums for each x value
        sums = np.vstack(ys).sum(axis=0)
        # Normalize each y value
        ys = [y / sums * 100 for y in ys]
    
    # Set colors if not provided
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(y_cols)))
    
    # Create stacked area plot
    ax.stackplot(x, ys, labels=y_cols, colors=colors, alpha=alpha)
    
    # Set labels and title
    ax.set_xlabel(x_col, fontsize=12)
    
    if normalize:
        ax.set_ylabel('Percentage (%)', fontsize=12)
    else:
        ax.set_ylabel('Value', fontsize=12)
    
    ax.set_title(title, fontsize=16)
    
    # Add legend
    ax.legend(loc='best')
    
    # Format x-axis for datetime
    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
        fig.autofmt_xdate()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig


def create_sankey_diagram(nodes: List[str], 
                       source_indices: List[int], 
                       target_indices: List[int],
                       values: List[float], 
                       title: str = 'Sankey Diagram',
                       figsize: Tuple[int, int] = (12, 8), 
                       node_colors: Optional[List[str]] = None,
                       link_colors: Optional[List[str]] = None,
                       save_path: Optional[str] = None):
    """
    Create a Sankey diagram for visualizing flows between nodes.
    Note: Requires plotly to be installed.
    
    Parameters
    ----------
    nodes : list of str
        Names of the nodes
    source_indices : list of int
        Source node indices for each link
    target_indices : list of int
        Target node indices for each link
    values : list of float
        Values for each link (width of the flow)
    title : str, default='Sankey Diagram'
        Title of the plot
    figsize : tuple of int, default=(12, 8)
        Figure size (width, height) in inches
    node_colors : list of str, optional
        Colors for each node. If None, uses default colors.
    link_colors : list of str, optional
        Colors for each link. If None, uses colors based on source node.
    save_path : str, optional
        If provided, save the figure to this path (as PNG)
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The generated figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("This function requires plotly. Install with: pip install plotly")
        return None
    
    # Set default colors if not provided
    if node_colors is None:
        # Generate colors from matplotlib colormap
        cmap = plt.cm.viridis
        node_colors = [f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.8)' 
                       for r, g, b, _ in cmap(np.linspace(0, 1, len(nodes)))]
    
    # Set link colors based on source node if not provided
    if link_colors is None:
        link_colors = [node_colors[src] for src in source_indices]
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors
        )
    )])
    
    # Update layout
    fig.update_layout(
        title_text=title,
        font_size=12,
        width=figsize[0] * 80,  # Convert inches to pixels (approximate)
        height=figsize[1] * 80
    )
    
    # Save figure if path provided
    if save_path:
        fig.write_image(save_path, scale=2)
    
    return fig


def plot_calendar_heatmap(dates: List[Union[str, pd.Timestamp]], 
                        values: List[float],
                        title: str = 'Calendar Heatmap', 
                        year: Optional[int] = None,
                        cmap: str = 'YlGnBu', 
                        figsize: Tuple[int, int] = (16, 8),
                        month_labels: bool = True, 
                        value_label: str = 'Value',
                        save_path: Optional[str] = None):
    """
    Create a calendar heatmap for visualizing daily data across a year.
    
    Parameters
    ----------
    dates : list of str or pd.Timestamp
        Dates for the values
    values : list of float
        Values to plot for each date
    title : str, default='Calendar Heatmap'
        Title of the plot
    year : int, optional
        Year to plot. If None, uses the year of the first date.
    cmap : str, default='YlGnBu'
        Colormap for the heatmap
    figsize : tuple of int, default=(16, 8)
        Figure size (width, height) in inches
    month_labels : bool, default=True
        Whether to show month labels
    value_label : str, default='Value'
        Label for the colorbar
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Convert dates to datetime if they are strings
    if isinstance(dates[0], str):
        dates = pd.to_datetime(dates)
    
    # Create a Series with dates as index and values
    data = pd.Series(values, index=dates)
    
    # Determine year if not provided
    if year is None:
        year = data.index[0].year
    
    # Filter data for the specified year
    data = data[data.index.year == year]
    
    # Create a date range for the entire year
    start_date = pd.Timestamp(f'{year}-01-01')
    end_date = pd.Timestamp(f'{year}-12-31')
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Create a DataFrame with all dates in the year
    df = pd.DataFrame(index=date_range)
    
    # Add day of week (0=Monday, 6=Sunday)
    df['dow'] = df.index.dayofweek
    
    # Add week number and reindex Sunday to appear at the top
    df['week'] = df.index.isocalendar().week
    df.loc[df['dow'] == 6, 'week'] += 0.5  # Offset Sunday to connect with the next week
    
    # Add month for separating the plot
    df['month'] = df.index.month
    
    # Add value from the data
    df['value'] = data
    
    # Handle missing values
    df['value'].fillna(data.min() if len(data) > 0 else 0, inplace=True)
    
    # Define month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap for each day
    for date, row in df.iterrows():
        # Calculate position on the grid
        x = date.dayofyear / 366 * 12  # Spread across 12 months
        y = (6 - row['dow']) % 7  # Invert day of week for a top-to-bottom layout
        
        # Create rectangle
        rect = plt.Rectangle((x, y), 0.95/31, 0.95, 
                            color=plt.cm.get_cmap(cmap)((row['value'] - data.min()) / 
                                                       (data.max() - data.min()) 
                                                       if data.max() > data.min() else 0.5),
                            alpha=0.8, linewidth=0.5, edgecolor='gray')
        ax.add_patch(rect)
        
        # Add month separators
        if date.day == 1:
            ax.axvline(x, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add month labels if requested
    if month_labels:
        for i, month in enumerate(month_names):
            ax.text((i + 0.5) / 12, -0.5, month, ha='center', va='center', fontsize=12)
    
    # Add day of week labels
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, day in enumerate(day_names):
        ax.text(-0.05, 6-i, day, ha='right', va='center', fontsize=10)
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, 
                              norm=plt.Normalize(vmin=data.min(), vmax=data.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.02)
    cbar.set_label(value_label, fontsize=12)
    
    # Set plot limits and remove axes
    ax.set_xlim(-0.1, 12.1)
    ax.set_ylim(-1, 7)
    ax.axis('off')
    
    # Add title
    plt.title(title + f' ({year})', fontsize=16, pad=20)
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig


def plot_sunburst(labels: List[str], 
                parents: List[str], 
                values: Optional[List[float]] = None,
                title: str = 'Sunburst Chart', 
                colorscale: str = 'viridis',
                width: int = 800, 
                height: int = 800, 
                save_path: Optional[str] = None):
    """
    Create a sunburst chart for hierarchical data visualization.
    Note: Requires plotly to be installed.
    
    Parameters
    ----------
    labels : list of str
        Labels for each segment
    parents : list of str
        Parent labels for each segment (empty string for root level)
    values : list of float, optional
        Values for each segment. If None, all segments are equal.
    title : str, default='Sunburst Chart'
        Title of the plot
    colorscale : str, default='viridis'
        Colorscale for the segments
    width : int, default=800
        Width of the plot in pixels
    height : int, default=800
        Height of the plot in pixels
    save_path : str, optional
        If provided, save the figure to this path (as PNG)
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The generated figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("This function requires plotly. Install with: pip install plotly")
        return None
    
    # Create sunburst chart
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total',
        insidetextorientation='radial',
        marker=dict(
            colorscale=colorscale
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        margin=dict(t=30, b=10, l=10, r=10)
    )
    
    # Save figure if path provided
    if save_path:
        fig.write_image(save_path, scale=2)
    
    return fig    # Tight layout might cause issues with 3D plots, so we use a specific rect
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig


def plot_waffle_chart(categories: List[str], 
                    values: List[Union[int, float]],
                    figsize: Tuple[int, int] = (10, 8),
                    title: str = 'Waffle Chart', 
                    rows: int = 10, 
                    columns: int = 10,
                    colors: Optional[List[str]] = None, 
                    value_fmt: str = '{:.1f}%',
                    legend_loc: str = 'upper right', 
                    save_path: Optional[str] = None):
    """
    Create a waffle chart (square pie chart) for visualizing proportions.
    
    Parameters
    ----------
    categories : list of str
        Names of the categories
    values : list of int or float
        Values for each category
    figsize : tuple of int, default=(10, 8)
        Figure size (width, height) in inches
    title : str, default='Waffle Chart'
        Title of the plot
    rows : int, default=10
        Number of rows in the waffle chart
    columns : int, default=10
        Number of columns in the waffle chart
    colors : list of str, optional
        Colors for each category. If None, uses default color cycle.
    value_fmt : str, default='{:.1f}%'
        Format string for percentage values in the legend
    legend_loc : str, default='upper right'
        Location of the legend
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Calculate total value for percentage
    total_value = sum(values)
    
    # Calculate the number of cells for each category
    total_cells = rows * columns
    category_cells = [int(round(value / total_value * total_cells)) for value in values]
    
    # Adjust for rounding errors
    diff = total_cells - sum(category_cells)
    if diff > 0:
        # Add cells to the largest category
        largest_idx = values.index(max(values))
        category_cells[largest_idx] += diff
    elif diff < 0:
        # Remove cells from the smallest category
        smallest_idx = values.index(min(values))
        category_cells[smallest_idx] += diff  # This will subtract because diff is negative
    
    # Create a grid for the waffle chart
    waffle_grid = np.zeros((rows, columns), dtype=int)
    
    # Fill in the grid with category indices
    cell_count = 0
    for category_idx, cell_count_category in enumerate(category_cells):
        for i in range(cell_count_category):
            row_idx = cell_count // columns
            col_idx = cell_count % columns
            
            if row_idx < rows and col_idx < columns:
                waffle_grid[row_idx, col_idx] = category_idx + 1
            
            cell_count += 1
    
    # Set colors if not provided
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
    
    # Add a background color (white or light gray)
    colors = ['#f5f5f5'] + list(colors)
    
    # Create a color map
    cmap = LinearSegmentedColormap.from_list('waffle_cmap', colors, N=len(categories) + 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the waffle chart as an image
    im = ax.imshow(waffle_grid, cmap=cmap, interpolation='none', vmin=0, vmax=len(categories))
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add grid lines
    for i in range(columns + 1):
        ax.axvline(i - 0.5, color='white', linewidth=1.5)
    
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1.5)
    
    # Create legend patches
    legend_elements = []
    for i, (category, value) in enumerate(zip(categories, values)):
        percentage = value / total_value * 100
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=colors[i + 1],
                        label=f'{category} ({value_fmt.format(percentage)})')
        )
    
    # Add legend
    ax.legend(handles=legend_elements, loc=legend_loc)
    
    # Add title
    plt.title(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig


def plot_dendrogram(data: Union[pd.DataFrame, np.ndarray], 
                  labels: Optional[List[str]] = None,
                  method: str = 'ward', 
                  metric: str = 'euclidean',
                  color_threshold: Optional[float] = None,
                  figsize: Tuple[int, int] = (12, 8),
                  title: str = 'Hierarchical Clustering Dendrogram',
                  orientation: str = 'top',
                  save_path: Optional[str] = None):
    """
    Create a dendrogram for hierarchical clustering visualization.
    
    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        The data to cluster. If DataFrame, rows are observations, columns are features.
    labels : list of str, optional
        Labels for observations. If None, indices are used.
    method : str, default='ward'
        Linkage method: 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'
    metric : str, default='euclidean'
        Distance metric for calculating the linkage
    color_threshold : float, optional
        Threshold for coloring the dendrogram branches. If None, automatic threshold is used.
    figsize : tuple of int, default=(12, 8)
        Figure size (width, height) in inches
    title : str, default='Hierarchical Clustering Dendrogram'
        Title of the plot
    orientation : str, default='top'
        Dendrogram orientation: 'top', 'bottom', 'left', 'right'
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    from scipy.cluster import hierarchy
    
    # Convert DataFrame to numpy array if needed
    if isinstance(data, pd.DataFrame):
        if labels is None:
            labels = data.index.tolist()
        data = data.values
    
    # Compute linkage matrix
    Z = hierarchy.linkage(data, method=method, metric=metric)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot dendrogram
    dendrogram = hierarchy.dendrogram(
        Z,
        orientation=orientation,
        labels=labels,
        leaf_rotation=90 if orientation in ['top', 'bottom'] else 0,
        leaf_font_size=10,
        color_threshold=color_threshold,
        ax=ax
    )
    
    # Remove axis frames if orientation is left/right
    if orientation in ['left', 'right']:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Set title and labels
    plt.title(title, fontsize=16)
    
    if orientation in ['top', 'bottom']:
        plt.xlabel('Samples', fontsize=14)
        plt.ylabel('Distance', fontsize=14)
    else:
        plt.xlabel('Distance', fontsize=14)
        plt.ylabel('Samples', fontsize=14)
    
    # Add method information
    plt.figtext(0.01, 0.01, f'Method: {method}, Metric: {metric}', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig


def plot_biplot(df: pd.DataFrame, 
              features: List[str], 
              n_components: int = 2,
              figsize: Tuple[int, int] = (12, 10), 
              title: str = 'PCA Biplot',
              scale_arrows: float = 1.0, 
              samples_alpha: float = 0.7,
              color_by: Optional[str] = None, 
              save_path: Optional[str] = None):
    """
    Create a biplot to visualize PCA results with feature vectors.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to visualize
    features : list of str
        Feature columns to include in the PCA
    n_components : int, default=2
        Number of principal components to compute
    figsize : tuple of int, default=(12, 10)
        Figure size (width, height) in inches
    title : str, default='PCA Biplot'
        Title of the plot
    scale_arrows : float, default=1.0
        Scaling factor for the feature vectors
    samples_alpha : float, default=0.7
        Transparency of the sample points
    color_by : str, optional
        Column name to use for coloring points
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    pca : sklearn.decomposition.PCA
        The fitted PCA model
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Extract the features
    X = df[features].values
    
    # Standardize the features
    X_std = StandardScaler().fit_transform(X)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pc_scores = pca.fit_transform(X_std)
    
    # Calculate the loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the scores
    if color_by is not None and color_by in df.columns:
        scatter = plt.scatter(pc_scores[:, 0], pc_scores[:, 1],
                           c=df[color_by], cmap='viridis',
                           alpha=samples_alpha, edgecolors='w')
        
        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label(color_by)
    else:
        plt.scatter(pc_scores[:, 0], pc_scores[:, 1],
                   alpha=samples_alpha, edgecolors='w')
    
    # Plot the feature vectors
    for i, feature in enumerate(features):
        # Scale the arrows
        scaled_loading = loadings[i, :] * scale_arrows
        
        # Plot the arrow
        plt.arrow(0, 0, scaled_loading[0], scaled_loading[1],
                 color='red', alpha=0.8, head_width=0.05, head_length=0.1)
        
        # Label the arrow
        plt.text(scaled_loading[0] * 1.15, scaled_loading[1] * 1.15,
                feature, color='red', ha='center', va='center', fontsize=10)
    
    # Add a unit circle for reference
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', alpha=0.3)
    ax.add_patch(circle)
    
    # Set labels with explained variance
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    
    # Add a title
    plt.title(title, fontsize=16)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Ensure the aspect ratio is equal
    plt.axis('equal')
    
    # Add explained variance text
    total_var = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
    plt.annotate(f'Total explained variance: {total_var*100:.1f}%',
                xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig, pca


def plot_radar_chart(categories: List[str], 
                   values: List[List[float]], 
                   labels: List[str],
                   figsize: Tuple[int, int] = (10, 10), 
                   title: str = 'Radar Chart',
                   colors: Optional[List[str]] = None, 
                   fill_alpha: float = 0.25,
                   value_range: Optional[Tuple[float, float]] = None,
                   show_legend: bool = True, 
                   save_path: Optional[str] = None):
    """
    Create a radar (spider) chart for comparing multiple groups across different categories.
    
    Parameters
    ----------
    categories : list of str
        Names of the categories (axes)
    values : list of list of float
        Values for each group across categories
        Shape: (n_groups, n_categories)
    labels : list of str
        Names of the groups
    figsize : tuple of int, default=(10, 10)
        Figure size (width, height) in inches
    title : str, default='Radar Chart'
        Title of the plot
    colors : list of str, optional
        Colors for each group. If None, uses default color cycle.
    fill_alpha : float, default=0.25
        Alpha value for the fill color
    value_range : tuple of float, optional
        Range of values for the chart. If None, calculated from data.
    show_legend : bool, default=True
        Whether to show the legend
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Ensure all input lists are the same length
    if len(categories) != len(values[0]):
        raise ValueError("Number of categories must match the number of values per group")
    
    if len(values) != len(labels):
        raise ValueError("Number of value lists must match the number of labels")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Number of categories
    n_categories = len(categories)
    
    # Set number of angles for plot
    angles = np.linspace(0, 2*np.pi, n_categories, endpoint=False).tolist()
    
    # Close the plot by repeating the first angle
    angles += angles[:1]
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Set colors if not provided
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    
    # Determine value range if not provided
    if value_range is None:
        all_values = [val for sublist in values for val in sublist]
        min_val = min(0, min(all_values))  # Include 0 if not the minimum
        max_val = max(all_values) * 1.1    # Add 10% padding
        value_range = (min_val, max_val)
    
    # Set y limits
    ax.set_ylim(value_range)
    
    # Plot each group
    for i, (group_values, label) in enumerate(zip(values, labels)):
        # Close the values by repeating the first value
        group_values_closed = group_values + [group_values[0]]
        
        # Plot the values
        ax.plot(angles, group_values_closed, 'o-', linewidth=2, 
               label=label, color=colors[i])
        
        # Fill the area
        ax.fill(angles, group_values_closed, alpha=fill_alpha, color=colors[i])
    
    # Add legend if requested
    if show_legend:
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    plt.title(title, fontsize=16, y=1.08)
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig


def create_bubble_chart(df: pd.DataFrame, 
                      x_col: str, 
                      y_col: str, 
                      size_col: str,
                      color_col: Optional[str] = None, 
                      label_col: Optional[str] = None,
                      figsize: Tuple[int, int] = (12, 8), 
                      title: str = 'Bubble Chart',
                      size_scale: float = 1000, 
                      alpha: float = 0.7,
                      show_legend: bool = True, 
                      color_map: str = 'viridis',
                      save_path: Optional[str] = None):
    """
    Create a bubble chart with optional labels and color encoding.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to visualize
    x_col : str
        Column name for x-axis values
    y_col : str
        Column name for y-axis values
    size_col : str
        Column name for bubble size
    color_col : str, optional
        Column name for color encoding
    label_col : str, optional
        Column name for bubble labels
    figsize : tuple of int, default=(12, 8)
        Figure size (width, height) in inches
    title : str, default='Bubble Chart'
        Title of the plot
    size_scale : float, default=1000
        Scaling factor for bubble sizes
    alpha : float, default=0.7
        Transparency of the bubbles
    show_legend : bool, default=True
        Whether to show size and color legends
    color_map : str, default='viridis'
        Colormap for color encoding
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate bubble sizes
    size_values = df[size_col].values
    sizes = size_values * size_scale / size_values.max()
    
    # Create scatter plot
    if color_col:
        # Color encoding
        scatter = ax.scatter(df[x_col], df[y_col], s=sizes, c=df[color_col],
                           cmap=color_map, alpha=alpha, edgecolors='w', linewidth=0.5)
        
        # Add colorbar
        if show_legend:
            cbar = plt.colorbar(scatter)
            cbar.set_label(color_col)
    else:
        # No color encoding
        scatter = ax.scatter(df[x_col], df[y_col], s=sizes,
                           alpha=alpha, edgecolors='w', linewidth=0.5)
    
    # Add labels if specified
    if label_col:
        for i, txt in enumerate(df[label_col]):
            ax.annotate(txt, (df[x_col].iloc[i], df[y_col].iloc[i]),
                        fontsize=8, ha='center', va='center')
    
    # Add size legend if requested
    if show_legend:
        # Create dummy scatter points for the size legend
        size_legend_sizes = [min(sizes), (min(sizes) + max(sizes))/2, max(sizes)]
        size_legend_values = [min(size_values), (min(size_values) + max(size_values))/2, max(size_values)]
        
        # Format the values
        if isinstance(size_legend_values[0], (int, float)):
            size_legend_labels = [f'{val:.1f}' for val in size_legend_values]
        else:
            size_legend_labels = [str(val) for val in size_legend_values]
        
        # Create legend handles
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                  markersize=np.sqrt(size/100), label=f'{size_col}: {label}')
            for size, label in zip(size_legend_sizes, size_legend_labels)
        ]
        
        # Add the size legend
        ax.legend(handles=legend_elements, loc='upper right')
    
    # Set axis labels
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    
    # Add title
    ax.set_title(title, fontsize=16)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig


def plot_geographical_data(df: pd.DataFrame, 
                        lat_col: str, 
                        lon_col: str, 
                        color_col: Optional[str] = None,
                        size_col: Optional[str] = None,
                        title: str = 'Geographical Data Visualization',
                        figsize: Tuple[int, int] = (12, 8),
                        cmap: str = 'viridis',
                        marker: str = 'o',
                        alpha: float = 0.7,
                        basemap: bool = True,
                        save_path: Optional[str] = None):
    """
    Create a geographical scatter plot using latitude and longitude data.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to visualize
    lat_col : str
        Column name for latitude values
    lon_col : str
        Column name for longitude values
    color_col : str, optional
        Column name for color encoding
    size_col : str, optional
        Column name for marker size encoding
    title : str, default='Geographical Data Visualization'
        Title of the plot
    figsize : tuple of int, default=(12, 8)
        Figure size (width, height) in inches
    cmap : str, default='viridis'
        Colormap for color encoding
    marker : str, default='o'
        Marker style for points
    alpha : float, default=0.7
        Transparency of the markers
    basemap : bool, default=True
        Whether to use cartopy to add a map background. 
        Requires cartopy to be installed.
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False
        if basemap:
            print("Warning: cartopy is not installed. Using a simple scatter plot instead.")
            print("To install cartopy: pip install cartopy")
            basemap = False
    
    # Create figure with map projection if using cartopy
    if basemap and has_cartopy:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        # Set map extent based on data (with some padding)
        lon_min, lon_max = df[lon_col].min(), df[lon_col].max()
        lat_min, lat_max = df[lat_col].min(), df[lat_col].max()
        
        # Add 10% padding
        lon_padding = (lon_max - lon_min) * 0.1
        lat_padding = (lat_max - lat_min) * 0.1
        
        ax.set_extent([
            lon_min - lon_padding, 
            lon_max + lon_padding,
            lat_min - lat_padding, 
            lat_max + lat_padding
        ], crs=ccrs.PlateCarree())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
    else:
        # Create a regular figure without map background
        fig, ax = plt.subplots(figsize=figsize)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Set axis labels
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
    
    # Determine size of markers
    if size_col:
        # Normalize size values to a reasonable range (20-200)
        sizes = df[size_col].values
        if sizes.min() != sizes.max():  # Avoid division by zero
            sizes = 20 + 180 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
        else:
            sizes = 50  # Default size if all values are the same
    else:
        sizes = 50  # Default size
    
    # Create scatter plot
    if color_col:
        scatter = ax.scatter(df[lon_col], df[lat_col], 
                           c=df[color_col], cmap=cmap, 
                           s=sizes if isinstance(sizes, np.ndarray) else sizes,
                           marker=marker, alpha=alpha, 
                           edgecolors='w', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
        cbar.set_label(color_col, fontsize=12)
    else:
        scatter = ax.scatter(df[lon_col], df[lat_col],
                           s=sizes if isinstance(sizes, np.ndarray) else sizes,
                           marker=marker, alpha=alpha,
                           edgecolors='w', linewidth=0.5)
    
    # Add size legend if needed
    if size_col and isinstance(sizes, np.ndarray) and sizes.min() != sizes.max():
        # Create a separate legend for size
        size_legend_sizes = [sizes.min(), (sizes.min() + sizes.max()) / 2, sizes.max()]
        size_legend_values = [df[size_col].min(), 
                              (df[size_col].min() + df[size_col].max()) / 2, 
                              df[size_col].max()]
        
        # Format the values for the legend
        if isinstance(size_legend_values[0], (int, float)):
            size_legend_labels = [f'{val:.2f}' for val in size_legend_values]
        else:
            size_legend_labels = [str(val) for val in size_legend_values]
        
        # Create handles for the legend
        from matplotlib.lines import Line2D
        size_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                  markersize=np.sqrt(s/10), label=f'{size_col}: {l}')
            for s, l in zip(size_legend_sizes, size_legend_labels)
        ]
        
        # Add the legend
        ax.legend(handles=size_handles, loc='upper right')
    
    # Add title
    plt.title(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig
"""
Advanced visualization functions.

This module contains advanced and specialized visualization functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from typing import Optional, List, Tuple, Dict, Union, Any

from .core import save_figure, adjust_color_brightness


def plot_parallel_coordinates(df: pd.DataFrame, 
                             class_col: str, 
                             features: Optional[List[str]] = None,
                             sample_size: Optional[int] = None, 
                             normalize: bool = True,
                             figsize: Tuple[int, int] = (12, 8),
                             title: str = 'Parallel Coordinates Plot',
                             color_palette: str = 'tab10', 
                             alpha: float = 0.5,
                             save_path: Optional[str] = None):
    """
    Create a parallel coordinates plot to visualize multivariate data.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to visualize
    class_col : str
        Column name to use for coloring lines
    features : list of str, optional
        Specific columns to include. If None, all numeric columns are used.
    sample_size : int, optional
        If provided, random sample of rows to plot (useful for large datasets)
    normalize : bool, default=True
        Whether to normalize the features to [0-1] scale for better visualization
    figsize : tuple of int, default=(12, 8)
        Figure size (width, height) in inches
    title : str, default='Parallel Coordinates Plot'
        Title of the plot
    color_palette : str, default='tab10'
        Color palette to use for class colors
    alpha : float, default=0.5
        Transparency of the lines
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Get numeric columns if not specified
    if features is None:
        features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if class_col in features:
            features.remove(class_col)
    
    # Ensure class column is included
    plot_cols = features + [class_col]
    
    # Create a copy of the dataframe with selected columns
    plot_df = df[plot_cols].copy()
    
    # Sample data if needed
    if sample_size is not None and len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=42)
    
    # Normalize data if requested
    if normalize:
        for col in features:
            plot_df[col] = (plot_df[col] - plot_df[col].min()) / (plot_df[col].max() - plot_df[col].min())
    
    # Get unique classes
    classes = plot_df[class_col].unique()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a color map
    cmap = plt.get_cmap(color_palette)
    colors = {cls: cmap(i % cmap.N) for i, cls in enumerate(classes)}
    
    # Create legend handles
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=colors[cls], lw=2, label=str(cls)) for cls in classes]
    
    # Plot each class
    for cls in classes:
        # Get data for this class
        cls_data = plot_df[plot_df[class_col] == cls]
        
        # Get feature values
        for i, row in cls_data.iterrows():
            # Get y-coordinates (feature values)
            ys = row[features].values
            
            # Plot this instance
            ax.plot(range(len(features)), ys, color=colors[cls], alpha=alpha)
    
    # Set x-axis ticks and labels
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right')
    
    # Set title and labels
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(handles=legend_handles, title=class_col)
    
    plt.tight_layout()
    
    # Save figure if path provided
    save_figure(fig, save_path)
    
    return fig


def plot_3d_surface(x_data: np.ndarray, 
                  y_data: np.ndarray, 
                  z_data: np.ndarray,
                  figsize: Tuple[int, int] = (12, 10),
                  title: str = '3D Surface Plot', 
                  cmap: str = 'viridis',
                  angle: Tuple[int, int] = (30, 45),
                  alpha: float = 0.8,
                  save_path: Optional[str] = None):
    """
    Create a 3D surface plot.
    
    Parameters
    ----------
    x_data : np.ndarray
        Grid of x coordinates
    y_data : np.ndarray
        Grid of y coordinates
    z_data : np.ndarray
        Grid of z values (must have shape matching x_data and y_data)
    figsize : tuple of int, default=(12, 10)
        Figure size (width, height) in inches
    title : str, default='3D Surface Plot'
        Title of the plot
    cmap : str, default='viridis'
        Colormap for the surface
    angle : tuple of int, default=(30, 45)
        Viewing angle (elevation, azimuth) in degrees
    alpha : float, default=0.8
        Transparency of the surface
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(x_data, y_data, z_data, 
                         cmap=cmap, alpha=alpha, 
                         linewidth=0.5, edgecolors='gray',
                         antialiased=True)
    
    # Add a color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=10, pad=0.1)
    cbar.set_label('Z Value', fontsize=12)
    
    # Set viewing angle
    ax.view_init(elev=angle[0], azim=angle[1])
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(title, fontsize=16)
    
    # Tight layout might cause issues with 3D plots, so we use a specific rect
    plt.tight_layout(rect=[0, 0, 1, 0.98])