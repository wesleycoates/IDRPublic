import colorsys
from matplotlib.colors import LinearSegmentedColormap


def create_interactive_dashboard(df: pd.DataFrame, title: str = "Interactive Data Dashboard"):
    """
    Create a comprehensive interactive dashboard with multiple visualization types.
    Note: This function is meant to be used in Jupyter notebooks with ipywidgets installed.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to visualize
    title : str, default="Interactive Data Dashboard"
        Title of the dashboard
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError:
        print("This function requires ipywidgets. Install with: pip install ipywidgets")
        return
    
    # Get column lists by type
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Define visualization options
    viz_types = ['Histogram', 'Scatter Plot', 'Line Chart', 'Bar Chart', 'Box Plot', 
                'Violin Plot', 'Heatmap', 'Pair Plot', 'Correlation Matrix']
    
    # Create output widget for the visualization
    output = widgets.Output()
    
    # Create tab structure
    tab_data = widgets.Output()
    tab_viz = widgets.Output()
    tab_stats = widgets.Output()
    
    tabs = widgets.Tab(children=[tab_data, tab_viz, tab_stats])
    tabs.set_title(0, 'Data Preview')
    tabs.set_title(1, 'Visualizations')
    tabs.set_title(2, 'Statistics')
    
    # DATA TAB
    with tab_data:
        # Create widgets for data filtering
        filter_col = widgets.Dropdown(
            options=['None'] + df.columns.tolist(),
            value='None',
            description='Filter by:',
            style={'description_width': 'initial'}
        )
        
        filter_value = widgets.Text(
            value='',
            placeholder='Filter value',
            description='Value:',
            disabled=True,
            style={'description_width': 'initial'}
        )
        
        n_rows = widgets.IntSlider(
            value=10,
            min=5,
            max=100,
            step=5,
            description='Rows:',
            style={'description_width': 'initial'}
        )
        
        # Function to update filter value widget status
        def update_filter_value_status(*args):
            filter_value.disabled = filter_col.value == 'None'
        
        filter_col.observe(update_filter_value_status, names='value')
        
        # Create data display output
        data_output = widgets.Output()
        
        # Function to update data display
        def update_data_display(*args):
            with data_output:
                clear_output(wait=True)
                
                # Apply filter if selected
                if filter_col.value != 'None' and filter_value.value:
                    try:
                        # Handle different types of filters
                        col = filter_col.value
                        val = filter_value.value
                        
                        if df[col].dtype in ['int64', 'float64']:        # Create dummy scatter points for the legend
        for size, label in zip(size_legend_sizes, size_legend_labels):
            ax.scatter([], [], s=size, c='gray', alpha=0.7, edgecolors='w',
                     linewidth=0.5, label=f'{size_col}: {label}')
        
        # Add the size legend
        plt.legend(loc='lower right')
    
    # Add title
    plt.title(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_dendrogram(data: Union[pd.DataFrame, np.ndarray], labels: Optional[List[str]] = None,
                  method: str = 'ward', metric: str = 'euclidean',
                  color_threshold: Optional[float] = None,
                  figsize: Tuple[int, int] = (12, 8),
                  title: str = 'Hierarchical Clustering Dendrogram',
                  orientation: str = 'top',
                  save_path: Optional[str] = None):
    """
    Create a dendrogram for hierarchical clustering visualization.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_3d_surface(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
                  figsize: Tuple[int, int] = (12, 10),
                  title: str = '3D Surface Plot', 
                  cmap: str = 'viridis',
                  angle: Tuple[int, int] = (30, 45),
                  alpha: float = 0.8,
                  save_path: Optional[str] = None):
    """
    Create a 3D surface plot.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_waffle_chart(categories: List[str], values: List[Union[int, float]],
                    figsize: Tuple[int, int] = (10, 8),
                    title: str = 'Waffle Chart', rows: int = 10, columns: int = 10,
                    colors: Optional[List[str]] = None, value_fmt: str = '{:.1f}%',
                    legend_loc: str = 'upper right', save_path: Optional[str] = None):
    """
    Create a waffle chart (square pie chart) for visualizing proportions.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_interactive_time_series(df: pd.DataFrame, date_col: str, value_cols: List[str],
                               title: str = 'Interactive Time Series Plot',
                               figsize: Tuple[int, int] = (15, 8),
                               date_format: str = '%Y-%m-%d',
                               show_secondary_axis: bool = False):
    """
    Create an interactive time series plot with sliders for date range selection.
    Note: This function is meant to be used in Jupyter notebooks with ipywidgets installed.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to visualize
    date_col : str
        Column name for dates/timestamps
    value_cols : list of str
        Column names for values to plot
    title : str, default='Interactive Time Series Plot'
        Title of the plot
    figsize : tuple of int, default=(15, 8)
        Figure size (width, height) in inches
    date_format : str, default='%Y-%m-%d'
        Format string for displaying dates
    show_secondary_axis : bool, default=False
        Whether to show a secondary y-axis (only used when there are exactly 2 value columns)
    """
    # Try to import ipywidgets
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("This function requires ipywidgets. Install with: pip install ipywidgets")
        return
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by date
    df = df.sort_values(date_col)
    
    # Get date range
    date_min = df[date_col].min().strftime(date_format)
    date_max = df[date_col].max().strftime(date_format)
    
    # Create widgets
    date_range_slider = widgets.SelectionRangeSlider(
        options=[(d.strftime(date_format), d) for d in pd.to_datetime(df[date_col].unique())],
        index=(0, len(df[date_col].unique()) - 1),
        description='Date Range:',
        layout={'width': '800px'}
    )
    
    show_points_checkbox = widgets.Checkbox(
        value=False,
        description='Show Data Points',
        layout={'width': '150px'}
    )
    
    line_styles = widgets.Dropdown(
        options=['solid', 'dashed', 'dotted', 'dashdot'],
        value='solid',
        description='Line Style:',
        layout={'width': '150px'}
    )
    
    # Create output widget for the plot
    output = widgets.Output()
    
    # Function to update the plot
    def update_plot(*args):
        with output:
            # Clear previous output
            output.clear_output(wait=True)
            
            # Get selected date range
            start_date, end_date = date_range_slider.value
            
            # Filter data by date range
            mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
            filtered_df = df.loc[mask]
            
            # Create plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Secondary axis
            if show_secondary_axis and len(value_cols) == 2:
                ax2 = ax.twinx()
                
                # Plot first value on primary axis
                line1 = ax.plot(filtered_df[date_col], filtered_df[value_cols[0]], 
                               label=value_cols[0], linestyle=line_styles.value, 
                               marker='o' if show_points_checkbox.value else None)
                
                # Plot second value on secondary axis
                line2 = ax2.plot(filtered_df[date_col], filtered_df[value_cols[1]], 
                                label=value_cols[1], linestyle=line_styles.value, 
                                color='red', marker='o' if show_points_checkbox.value else None)
                
                ax.set_ylabel(value_cols[0], fontsize=12)
                ax2.set_ylabel(value_cols[1], fontsize=12, color='red')
                ax2.tick_params(axis='y', colors='red')
                
                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc='best')
            else:
                # Plot all values on the same axis
                for col in value_cols:
                    ax.plot(filtered_df[date_col], filtered_df[col], label=col, 
                           linestyle=line_styles.value,
                           marker='o' if show_points_checkbox.value else None)
                
                ax.set_ylabel('Value', fontsize=12)
                ax.legend(loc='best')
            
            # Format x-axis
            ax.set_xlabel('Date', fontsize=12)
            fig.autofmt_xdate()
            
            # Add title with date range
            plt.title(f"{title}\n{start_date} to {end_date}", fontsize=16)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.show()
    
    # Connect the update function to the widgets
    date_range_slider.observe(update_plot, names='value')
    show_points_checkbox.observe(update_plot, names='value')
    line_styles.observe(update_plot, names='value')
    
    # Create layout
    controls = widgets.HBox([show_points_checkbox, line_styles])
    layout = widgets.VBox([
        widgets.HTML(f"<h3>{title}</h3>"),
        date_range_slider,
        controls,
        output
    ])
    
    # Display the dashboard
    display(layout)
    
    # Initialize the plot
    update_plot()


def create_stacked_area_chart(df: pd.DataFrame, x_col: str, y_cols: List[str],
                            figsize: Tuple[int, int] = (12, 8),
                            title: str = 'Stacked Area Chart',
                            normalize: bool = False,
                            alpha: float = 0.8,
                            colors: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
    """
    Create a stacked area chart for visualizing composition over time or categories.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def create_sankey_diagram(nodes: List[str], source_indices: List[int], target_indices: List[int],
                       values: List[float], title: str = 'Sankey Diagram',
                       figsize: Tuple[int, int] = (12, 8), 
                       node_colors: Optional[List[str]] = None,
                       link_colors: Optional[List[str]] = None,
                       save_path: Optional[str] = None):
    """
    Create a Sankey diagram for visualizing flows between nodes.
    Note: Requires plotly to be installed.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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


def plot_calendar_heatmap(dates: List[Union[str, pd.Timestamp]], values: List[float],
                        title: str = 'Calendar Heatmap', year: Optional[int] = None,
                        cmap: str = 'YlGnBu', figsize: Tuple[int, int] = (16, 8),
                        month_labels: bool = True, value_label: str = 'Value',
                        save_path: Optional[str] = None):
    """
    Create a calendar heatmap for visualizing daily data across a year.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def create_animated_chart(df: pd.DataFrame, x_col: str, y_col: str, time_col: str,
                        color_col: Optional[str] = None, size_col: Optional[str] = None,
                        title: str = 'Animated Chart', fps: int = 5,
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None):
    """
    Create an animated scatter plot showing changes over time.
    Note: Requires matplotlib animation support.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to visualize
    x_col : str
        Column name for x-axis values
    y_col : str
        Column name for y-axis values
    time_col : str
        Column name for time values (used for animation frames)
    color_col : str, optional
        Column name for color encoding
    size_col : str, optional
        Column name for size encoding
    title : str, default='Animated Chart'
        Title of the plot
    fps : int, default=5
        Frames per second for the animation
    figsize : tuple of int, default=(10, 6)
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, save the animation to this path (as GIF)
    
    Returns:
    --------
    anim : matplotlib.animation.FuncAnimation
        The animation object
    """
    try:
        import matplotlib.animation as animation
    except ImportError:
        print("This function requires matplotlib animation support.")
        return None
    
    # Get unique time values
    time_values = sorted(df[time_col].unique())
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set axis labels
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    
    # Set title with placeholder for time
    title_with_time = ax.set_title(f"{title}\nTime: {time_values[0]}", fontsize=16)
    
    # Determine x and y limits to keep them fixed during animation
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    
    # Add some padding to the limits
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    # Set fixed limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create a scatter plot with the first time frame
    initial_time = time_values[0]
    initial_data = df[df[time_col] == initial_time]
    
    # Determine scatter plot parameters
    scatter_params = {
        'x': initial_data[x_col],
        'y': initial_data[y_col],
        'alpha': 0.7,
        'edgecolors': 'w',
        'linewidth': 0.5
    }
    
    # Add color encoding if specified
    if color_col is not None:
        scatter_params['c'] = initial_data[color_col]
        scatter_params['cmap'] = 'viridis'
    
    # Add size encoding if specified
    if size_col is not None:
        # Normalize size values to a reasonable range (20-200)
        sizes = initial_data[size_col].values
        if sizes.min() != sizes.max():  # Avoid division by zero
            scatter_params['s'] = 20 + 180 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
        else:
            scatter_params['s'] = 50
    else:
        scatter_params['s'] = 50
    
    # Create scatter plot
    scatter = ax.scatter(**scatter_params)
    
    # Create colorbar if color encoding is used
    if color_col is not None:
        cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
        cbar.set_label(color_col, fontsize=12)
    
    # Animation update function
    def update(frame):
        # Get data for this time frame
        frame_time = time_values[frame]
        frame_data = df[df[time_col] == frame_time]
        
        # Update scatter plot
        scatter.set_offsets(np.c_[frame_data[x_col], frame_data[y_col]])
        
        # Update colors if specified
        if color_col is not None:
            scatter.set_array(frame_data[color_col])
        
        # Update sizes if specified
        if size_col is not None:
            sizes = frame_data[size_col].values
            if sizes.min() != sizes.max():  # Avoid division by zero
                scatter.set_sizes(20 + 180 * (sizes - sizes.min()) / (sizes.max() - sizes.min()))
        
        # Update title with current time
        title_with_time.set_text(f"{title}\nTime: {frame_time}")
        
        return scatter,
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(time_values),
                                  interval=1000 // fps, blit=True)
    
    # Save animation if path provided
    if save_path:
        try:
            from matplotlib.animation import PillowWriter
            anim.save(save_path, writer=PillowWriter(fps=fps))
        except ImportError:
            print("Saving animation requires Pillow. Install with: pip install Pillow")
    
    plt.tight_layout()
    plt.close()  # Close the figure to avoid displaying it twice in notebooks
    
    return anim


def plot_sunburst(labels: List[str], parents: List[str], values: Optional[List[float]] = None,
                title: str = 'Sunburst Chart', colorscale: str = 'viridis',
                width: int = 800, height: int = 800, save_path: Optional[str] = None):
    """
    Create a sunburst chart for hierarchical data visualization.
    Note: Requires plotly to be installed.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    
    return fig


def plot_dual_axis(df: pd.DataFrame, x_col: str, y1_cols: List[str], y2_cols: List[str],
                 figsize: Tuple[int, int] = (12, 6), title: str = 'Dual Axis Chart',
                 y1_label: str = 'Primary Y-Axis', y2_label: str = 'Secondary Y-Axis',
                 y1_color: str = 'blue', y2_color: str = 'red',
                 plot_type: str = 'line', alpha: float = 0.8,
                 save_path: Optional[str] = None):
    """
    Create a dual-axis chart with different variables on primary and secondary y-axes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to visualize
    x_col : str
        Column name for x-axis values
    y1_cols : list of str
        Column names for primary y-axis
    y2_cols : list of str
        Column names for secondary y-axis
    figsize : tuple of int, default=(12, 6)
        Figure size (width, height) in inches
    title : str, default='Dual Axis Chart'
        Title of the plot
    y1_label : str, default='Primary Y-Axis'
        Label for the primary y-axis
    y2_label : str, default='Secondary Y-Axis'
        Label for the secondary y-axis
    y1_color : str, default='blue'
        Color for primary y-axis elements
    y2_color : str, default='red'
        Color for secondary y-axis elements
    plot_type : str, default='line'
        Type of plot: 'line', 'bar', or 'mixed' (line for primary, bar for secondary)
    alpha : float, default=0.8
        Transparency of the plot elements
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Create figure with primary y-axis
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Set x-axis label
    ax1.set_xlabel(x_col, fontsize=12)
    
    # Set primary y-axis label and color
    ax1.set_ylabel(y1_label, fontsize=12, color=y1_color)
    ax1.tick_params(axis='y', labelcolor=y1_color)
    
    # Plot on primary y-axis
    y1_lines = []
    y1_labels = []
    
    for i, col in enumerate(y1_cols):
        # Use different shades of the primary color
        color_val = y1_color if len(y1_cols) == 1 else adjust_color_brightness(y1_color, i, len(y1_cols))
        
        if plot_type in ['line', 'mixed']:
            line, = ax1.plot(df[x_col], df[col], color=color_val, alpha=alpha, 
                           linewidth=2, marker='o', markersize=5)
        else:  # bar
            width = 0.8 / len(y1_cols)
            offset = width * i - width * len(y1_cols) / 2 + width / 2
            line = ax1.bar(df[x_col] + offset, df[col], width=width, color=color_val, alpha=alpha)
        
        y1_lines.append(line)
        y1_labels.append(col)
    
    # Create secondary y-axis
    ax2 = ax1.twinx()
    
    # Set secondary y-axis label and color
    ax2.set_ylabel(y2_label, fontsize=12, color=y2_color)
    ax2.tick_params(axis='y', labelcolor=y2_color)
    
    # Plot on secondary y-axis
    y2_lines = []
    y2_labels = []
    
    for i, col in enumerate(y2_cols):
        # Use different shades of the secondary color
        color_val = y2_color if len(y2_cols) == 1 else adjust_color_brightness(y2_color, i, len(y2_cols))
        
        if plot_type == 'mixed':
            width = 0.8 / len(y2_cols)
            offset = width * i - width * len(y2_cols) / 2 + width / 2
            line = ax2.bar(df[x_col] + offset, df[col], width=width, color=color_val, alpha=alpha)
        elif plot_type == 'bar':
            width = 0.8 / (len(y1_cols) + len(y2_cols))
            offset = width * (i + len(y1_cols)) - width * (len(y1_cols) + len(y2_cols)) / 2 + width / 2
            line = ax2.bar(df[x_col] + offset, df[col], width=width, color=color_val, alpha=alpha)
        else:  # line
            line, = ax2.plot(df[x_col], df[col], color=color_val, alpha=alpha, 
                           linewidth=2, marker='s', markersize=5, linestyle='--')
        
        y2_lines.append(line)
        y2_labels.append(col)
    
    # Format x-axis based on data type
    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
        fig.autofmt_xdate()
    
    # Add grid for primary y-axis
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Add title
    plt.title(title, fontsize=16)
    
    # Combine legends from both axes
    if plot_type == 'line':
        all_lines = y1_lines + y2_lines
        all_labels = y1_labels + y2_labels
        ax1.legend(all_lines, all_labels, loc='best')
    else:
        # For bar charts, we need a different approach
        from matplotlib.lines import Line2D
        
        legend_elements = []
        for i, label in enumerate(y1_labels):
            color_val = y1_color if len(y1_cols) == 1 else adjust_color_brightness(y1_color, i, len(y1_cols))
            legend_elements.append(Line2D([0], [0], color=color_val, lw=4, label=label))
        
        for i, label in enumerate(y2_labels):
            color_val = y2_color if len(y2_cols) == 1 else adjust_color_brightness(y2_color, i, len(y2_cols))
            legend_elements.append(Line2D([0], [0], color=color_val, lw=4, label=label))
        
        ax1.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def adjust_color_brightness(color: str, index: int, total: int) -> str:
    """
    Adjust the brightness of a color based on index in a sequence.
    
    Parameters:
    -----------
    color : str
        Base color (color name or hex code)
    index : int
        Index in the sequence
    total : int
        Total number of colors needed
    
    Returns:
    --------
    str : Adjusted color in hex format
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


def plot_biplot(df: pd.DataFrame, features: List[str], n_components: int = 2,
              figsize: Tuple[int, int] = (12, 10), title: str = 'PCA Biplot',
              scale_arrows: float = 1.0, samples_alpha: float = 0.7,
              color_by: Optional[str] = None, save_path: Optional[str] = None):
    """
    Create a biplot to visualize PCA results with feature vectors.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, pca


def create_bubble_chart(df: pd.DataFrame, x_col: str, y_col: str, size_col: str,
                      color_col: Optional[str] = None, label_col: Optional[str] = None,
                      figsize: Tuple[int, int] = (12, 8), title: str = 'Bubble Chart',
                      size_scale: float = 1000, alpha: float = 0.7,
                      show_legend: bool = True, color_map: str = 'viridis',
                      save_path: Optional[str] = None):
    """
    Create a bubble chart with optional labels and color encoding.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


# Import extra modules needed for advanced functions
    # Handle numeric filters
    if df[col].dtype in ['int64', 'float64']:
        # Try to convert to number and filter
        try:
            if '<' in val:
                num_val = float(val.replace('<', '').strip())
                filtered_df = df[df[col] < num_val]
            elif '>' in val:
                num_val = float(val.replace('>', '').strip())
                filtered_df = df[df[col] > num_val]
            elif '-' in val and not val.startswith('-'):
                lower, upper = map(float, val.split('-'))
                filtered_df = df[(df[col] >= lower) & (df[col] <= upper)]
            else:
                num_val = float(val)
                filtered_df = df[df[col] == num_val]
        except:
            print(f"Invalid numeric filter: {val}")
            filtered_df = df
    else:
        # String filter - case insensitive substring match
        filtered_df = df[df[col].astype(str).str.contains(val, case=False, na=False)]
    except Exception as e:
        print(f"Filter error: {e}")
        filtered_df = df
    else:
        filtered_df = df

# Display data
display(filtered_df.head(n_rows))
print(f"Showing {min(n_rows, len(filtered_df))} of {len(filtered_df)} rows")
        
        # Create data tab layout
        filter_controls = widgets.HBox([filter_col, filter_value, n_rows])
        data_button = widgets.Button(description='Update Data View')
        data_button.on_click(update_data_display)
        
        data_layout = widgets.VBox([
            widgets.HTML(f"<h3>Data Preview</h3>"),
            filter_controls,
            data_button,
            data_output
        ])
        
        display(data_layout)
        update_data_display()  # Initial data display
    
    # VISUALIZATION TAB
    with tab_viz:
        # Create visualization controls
        viz_type = widgets.Dropdown(
            options=viz_types,
            value='Histogram',
            description='Plot Type:',
            style={'description_width': 'initial'}
        )
        
        x_col = widgets.Dropdown(
            options=['None'] + df.columns.tolist(),
            value='None',
            description='X-Axis:',
            style={'description_width': 'initial'}
        )
        
        y_col = widgets.Dropdown(
            options=['None'] + numeric_cols,
            value='None',
            description='Y-Axis:',
            style={'description_width': 'initial'}
        )
        
        hue_col = widgets.Dropdown(
            options=['None'] + categorical_cols,
            value='None',
            description='Color By:',
            style={'description_width': 'initial'}
        )
        
        palette = widgets.Dropdown(
            options=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'tab10', 'Set1', 'Set2', 'Set3'],
            value='viridis',
            description='Color Palette:',
            style={'description_width': 'initial'}
        )
        
        # Function to update visualization
        def update_visualization(*args):
            with output:
                clear_output(wait=True)
                
                plt.figure(figsize=(12, 8))
                
                plot_type = viz_type.value
                x = None if x_col.value == 'None' else x_col.value
                y = None if y_col.value == 'None' else y_col.value
                hue = None if hue_col.value == 'None' else hue_col.value
                
                try:
                    if plot_type == 'Histogram':
                        if x:
                            sns.histplot(data=df, x=x, hue=hue, kde=True, palette=palette.value)
                            plt.title(f'Distribution of {x}', fontsize=16)
                        else:
                            print("Please select an X-axis variable for histogram")
                            return
                    
                    elif plot_type == 'Scatter Plot':
                        if x and y:
                            sns.scatterplot(data=df, x=x, y=y, hue=hue, palette=palette.value)
                            plt.title(f'{y} vs {x}', fontsize=16)
                        else:
                            print("Please select both X and Y axis variables for scatter plot")
                            return
                    
                    elif plot_type == 'Line Chart':
                        if x and y:
                            # Check if x is datetime or numeric
                            if x in datetime_cols or df[x].dtype in ['int64', 'float64']:
                                # Sort by x for better line plots
                                sorted_df = df.sort_values(by=x)
                                sns.lineplot(data=sorted_df, x=x, y=y, hue=hue, palette=palette.value)
                                plt.title(f'{y} vs {x}', fontsize=16)
                                
                                # Rotate x labels if datetime
                                if x in datetime_cols:
                                    plt.xticks(rotation=45)
                            else:
                                print("X-axis variable should be numeric or datetime for line charts")
                                return
                        else:
                            print("Please select both X and Y axis variables for line chart")
                            return
                    
                    elif plot_type == 'Bar Chart':
                        if x:
                            if y:
                                # Aggregated bar chart
                                sns.barplot(data=df, x=x, y=y, hue=hue, palette=palette.value)
                                plt.title(f'Average {y} by {x}', fontsize=16)
                            else:
                                # Count bar chart
                                sns.countplot(data=df, x=x, hue=hue, palette=palette.value)
                                plt.title(f'Count of {x}', fontsize=16)
                            
                            # Rotate x labels if many categories
                            if df[x].nunique() > 5:
                                plt.xticks(rotation=45, ha='right')
                        else:
                            print("Please select an X-axis variable for bar chart")
                            return
                    
                    elif plot_type == 'Box Plot':
                        if y:
                            sns.boxplot(data=df, x=x, y=y, hue=hue, palette=palette.value)
                            plt.title(f'Distribution of {y}' + (f' by {x}' if x else ''), fontsize=16)
                            
                            # Rotate x labels if many categories
                            if x and df[x].nunique() > 5:
                                plt.xticks(rotation=45, ha='right')
                        else:
                            print("Please select a Y-axis variable for box plot")
                            return
                    
                    elif plot_type == 'Violin Plot':
                        if y:
                            sns.violinplot(data=df, x=x, y=y, hue=hue, palette=palette.value, split=hue is not None)
                            plt.title(f'Distribution of {y}' + (f' by {x}' if x else ''), fontsize=16)
                            
                            # Rotate x labels if many categories
                            if x and df[x].nunique() > 5:
                                plt.xticks(rotation=45, ha='right')
                        else:
                            print("Please select a Y-axis variable for violin plot")
                            return
                    
                    elif plot_type == 'Heatmap':
                        if len(numeric_cols) < 2:
                            print("Need at least 2 numeric columns for a heatmap")
                            return
                        
                        # Create correlation matrix
                        corr = df[numeric_cols].corr()
                        
                        # Create mask for upper triangle
                        mask = np.triu(np.ones_like(corr, dtype=bool))
                        
                        # Create heatmap
                        sns.heatmap(corr, annot=True, mask=mask, cmap=palette.value, 
                                    vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
                        plt.title('Correlation Matrix', fontsize=16)
                    
                    elif plot_type == 'Pair Plot':
                        # Limit to 5 variables to avoid overcrowding
                        if len(numeric_cols) < 2:
                            print("Need at least 2 numeric columns for a pair plot")
                            return
                        
                        vars_to_plot = numeric_cols[:5]
                        
                        # Create pair plot
                        g = sns.pairplot(df, vars=vars_to_plot, hue=hue, palette=palette.value, 
                                         diag_kind='kde', plot_kws={'alpha': 0.6})
                        g.fig.suptitle('Pair Plot', y=1.02, fontsize=16)
                        plt.tight_layout()
                    
                    elif plot_type == 'Correlation Matrix':
                        if len(numeric_cols) < 2:
                            print("Need at least 2 numeric columns for a correlation matrix")
                            return
                        
                        # Create correlation matrix
                        plot_correlation_matrix(df, numeric_cols, figsize=(12, 10))
                        return  # Skip the plt.tight_layout() below as it's handled in the function
                    
                    plt.tight_layout()
                    plt.show()
                
                except Exception as e:
                    print(f"Visualization error: {e}")
        
        # Create visualization button
        viz_button = widgets.Button(description='Generate Visualization')
        viz_button.on_click(update_visualization)
        
        # Create visualization tab layout
        viz_controls = widgets.VBox([
            widgets.HTML(f"<h3>Create Visualization</h3>"),
            widgets.HBox([viz_type, palette]),
            widgets.HBox([x_col, y_col, hue_col]),
            viz_button
        ])
        
        # Complete visualization tab layout
        viz_layout = widgets.VBox([viz_controls, output])
        display(viz_layout)
    
    # STATISTICS TAB
    with tab_stats:
        # Create statistics widgets
        stat_type = widgets.Dropdown(
            options=['Summary Statistics', 'Group Statistics', 'Correlation Analysis', 'Distribution Tests'],
            value='Summary Statistics',
            description='Statistics:',
            style={'description_width': 'initial'}
        )
        
        group_col = widgets.Dropdown(
            options=['None'] + categorical_cols,
            value='None',
            description='Group By:',
            disabled=True,
            style={'description_width': 'initial'}
        )
        
        stat_cols = widgets.SelectMultiple(
            options=numeric_cols,
            value=numeric_cols[:min(5, len(numeric_cols))],
            description='Columns:',
            style={'description_width': 'initial'}
        )
        
        # Function to update group_col status
        def update_group_col_status(*args):
            group_col.disabled = stat_type.value != 'Group Statistics'
        
        stat_type.observe(update_group_col_status, names='value')
        
        # Create statistics output
        stats_output = widgets.Output()
        
        # Function to update statistics display
        def update_statistics(*args):
            with stats_output:
                clear_output(wait=True)
                
                selected_cols = list(stat_cols.value)
                if not selected_cols:
                    print("Please select at least one column")
                    return
                
                try:
                    if stat_type.value == 'Summary Statistics':
                        # Display descriptive statistics
                        stats_df = df[selected_cols].describe().T
                        # Add additional statistics
                        stats_df['median'] = df[selected_cols].median()
                        stats_df['skew'] = df[selected_cols].skew()
                        stats_df['kurtosis'] = df[selected_cols].kurtosis()
                        # Reorder columns
                        stats_df = stats_df[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis']]
                        display(stats_df)
                    
                    elif stat_type.value == 'Group Statistics':
                        if group_col.value == 'None':
                            print("Please select a grouping variable")
                            return
                        
                        # Display group statistics
                        group_stats = df.groupby(group_col.value)[selected_cols].agg(['mean', 'median', 'std', 'count'])
                        display(group_stats)
                        
                        # Create group visualization
                        plt.figure(figsize=(14, 8))
                        
                        # Create subplot for each numeric column
                        n_cols = min(3, len(selected_cols))
                        n_rows = (len(selected_cols) + n_cols - 1) // n_cols
                        
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
                        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
                        
                        for i, col in enumerate(selected_cols):
                            if i < len(axes):
                                sns.boxplot(x=group_col.value, y=col, data=df, ax=axes[i])
                                axes[i].set_title(f'{col} by {group_col.value}')
                                axes[i].tick_params(axis='x', rotation=45)
                        
                        # Hide unused subplots
                        for i in range(len(selected_cols), len(axes)):
                            axes[i].set_visible(False)
                        
                        plt.tight_layout()
                        plt.show()
                    
                    elif stat_type.value == 'Correlation Analysis':
                        if len(selected_cols) < 2:
                            print("Please select at least two columns for correlation analysis")
                            return
                        
                        # Calculate correlation matrix
                        corr = df[selected_cols].corr()
                        display(corr)
                        
                        # Create correlation heatmap
                        plt.figure(figsize=(10, 8))
                        mask = np.triu(np.ones_like(corr, dtype=bool))
                        sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', 
                                    vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
                        plt.title('Correlation Matrix', fontsize=16)
                        plt.tight_layout()
                        plt.show()
                        
                        # Display top correlations
                        corr_unstack = corr.unstack()
                        corr_unstack = corr_unstack[corr_unstack < 1]  # Remove self-correlations
                        top_corr = corr_unstack.abs().sort_values(ascending=False).head(10)
                        
                        print("\nTop Correlations:")
                        for (col1, col2), val in top_corr.items():
                            print(f"{col1} — {col2}: {val:.4f}")
                    
                    elif stat_type.value == 'Distribution Tests':
                        from scipy import stats as scipy_stats
                        
                        # Perform normality tests
                        print("Normality Tests (p-value < 0.05 suggests non-normal distribution):")
                        for col in selected_cols:
                            if df[col].nunique() > 5:  # Only test if enough unique values
                                shapiro_stat, shapiro_p = scipy_stats.shapiro(df[col].dropna())
                                ks_stat, ks_p = scipy_stats.kstest(
                                    (df[col].dropna() - df[col].mean()) / df[col].std(), 
                                    'norm'
                                )
                                
                                print(f"\n{col}:")
                                print(f"  Shapiro-Wilk Test: stat={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
                                print(f"  Kolmogorov-Smirnov Test: stat={ks_stat:.4f}, p-value={ks_p:.4f}")
                                
                                # Create Q-Q plot
                                plt.figure(figsize=(10, 4))
                                
                                plt.subplot(1, 2, 1)
                                sns.histplot(df[col].dropna(), kde=True)
                                plt.title(f'Distribution of {col}')
                                
                                plt.subplot(1, 2, 2)
                                scipy_stats.probplot(df[col].dropna(), plot=plt)
                                plt.title(f'Q-Q Plot of {col}')
                                
                                plt.tight_layout()
                                plt.show()
                            else:
                                print(f"\n{col}: Insufficient unique values for normality testing")
                
                except Exception as e:
                    print(f"Statistics error: {e}")
        
        # Create statistics button
        stats_button = widgets.Button(description='Calculate Statistics')
        stats_button.on_click(update_statistics)
        
        # Create statistics tab layout
        stats_controls = widgets.VBox([
            widgets.HTML(f"<h3>Statistical Analysis</h3>"),
            widgets.HBox([stat_type, group_col]),
            stat_cols,
            stats_button
        ])
        
        stats_layout = widgets.VBox([stats_controls, stats_output])
        display(stats_layout)
    
    # Display the main tabs
    display(widgets.HTML(f"<h2>{title}</h2>"))
    display(tabs)


if __name__ == "__main__":
    # This will execute when you run this script directly
    print("Enhanced Data Visualization Module")
    print("This module provides a comprehensive set of visualization functions for data analysis and machine learning.")
    print("Example usage:")
    print("from visualization import plot_numeric_distribution, plot_correlation_matrix")
    print("plot_numeric_distribution(df)")
    print("plot_correlation_matrix(df)")

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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_parallel_coordinates(df: pd.DataFrame, class_col: str, features: Optional[List[str]] = None,
                             sample_size: Optional[int] = None, normalize: bool = True,
                             figsize: Tuple[int, int] = (12, 8),
                             title: str = 'Parallel Coordinates Plot',
                             color_palette: str = 'tab10', alpha: float = 0.5,
                             save_path: Optional[str] = None):
    """
    Create a parallel coordinates plot to visualize multivariate data.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_silhouette_analysis(X: np.ndarray, cluster_labels: np.ndarray,
                           metric: str = 'euclidean', figsize: Tuple[int, int] = (12, 8),
                           title: str = 'Silhouette Analysis', 
                           save_path: Optional[str] = None):
    """
    Create a silhouette plot for evaluating clustering quality.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix used for clustering
    cluster_labels : np.ndarray
        Cluster assignments for each data point
    metric : str, default='euclidean'
        Distance metric for silhouette calculation
    figsize : tuple of int, default=(12, 8)
        Figure size (width, height) in inches
    title : str, default='Silhouette Analysis'
        Title of the plot
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    from sklearn.metrics import silhouette_samples, silhouette_score
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Get silhouette scores
    n_clusters = len(np.unique(cluster_labels))
    
    # The silhouette coefficient can range from -1 to 1
    ax1.set_xlim([-0.1, 1])
    
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
    # Compute the average silhouette score for all samples
    avg_silhouette = silhouette_score(X, cluster_labels, metric=metric)
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels, metric=metric)
    
    # Create color map for clusters
    cmap = plt.cm.get_cmap('viridis', n_clusters)
    
    y_lower = 10
    for i in range(n_clusters):
        # Get samples in this cluster
        samples_in_cluster = sample_silhouette_values[cluster_labels == i]
        samples_in_cluster.sort()
        
        # Get size of cluster
        size_cluster_i = samples_in_cluster.shape[0]
        
        # Create y-axis position range for this cluster
        y_upper = y_lower + size_cluster_i
        
        # Fill the silhouette
        color = cmap(i / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, samples_in_cluster,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}')
        
        # Update y_lower for next plot
        y_lower = y_upper + 10
    
    # Add vertical line for average silhouette score
    ax1.axvline(x=avg_silhouette, color="red", linestyle="--")
    
    # Set labels and title
    ax1.set_title("Silhouette Plot", fontsize=14)
    ax1.set_xlabel("Silhouette Coefficient", fontsize=12)
    ax1.set_ylabel("Cluster", fontsize=12)
    
    # Add text with average silhouette score
    ax1.text(0.7, 0.02, f'Average: {avg_silhouette:.3f}',
             transform=ax1.transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot a 2D projection of the data if more than 2 dimensions
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        
        # Plot the reduced data with cluster colors
        for i in range(n_clusters):
            # Get samples in this cluster
            cluster_samples = X_2d[cluster_labels == i]
            
            # Plot points for this cluster
            ax2.scatter(cluster_samples[:, 0], cluster_samples[:, 1],
                       s=30, color=cmap(i / n_clusters), alpha=0.7,
                       label=f'Cluster {i}')
        
        ax2.set_title("PCA Projection of Clusters", fontsize=14)
        ax2.set_xlabel("Principal Component 1", fontsize=12)
        ax2.set_ylabel("Principal Component 2", fontsize=12)
        ax2.legend(loc='best')
    else:
        # Plot the original 2D data with cluster colors
        for i in range(n_clusters):
            # Get samples in this cluster
            cluster_samples = X[cluster_labels == i]
            
            # Plot points for this cluster
            ax2.scatter(cluster_samples[:, 0], cluster_samples[:, 1],
                       s=30, color=cmap(i / n_clusters), alpha=0.7,
                       label=f'Cluster {i}')
        
        ax2.set_title("Cluster Visualization", fontsize=14)
        ax2.set_xlabel("Feature 1", fontsize=12)
        ax2.set_ylabel("Feature 2", fontsize=12)
        ax2.legend(loc='best')
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def create_violin_swarm_plot(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 6),
                           title: str = None, show_stats: bool = True,
                           palette: str = 'viridis', 
                           save_path: Optional[str] = None):
    """
    Create a combined violin and swarm plot for visualizing distribution and individual data points.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to visualize
    x : str
        Column name for the x-axis (categorical)
    y : str
        Column name for the y-axis (numeric)
    hue : str, optional
        Column name for color grouping
    figsize : tuple of int, default=(12, 6)
        Figure size (width, height) in inches
    title : str, optional
        Title of the plot. If None, creates a default title.
    show_stats : bool, default=True
        Whether to show summary statistics
    palette : str, default='viridis'
        Color palette to use
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create violin plot
    sns.violinplot(x=x, y=y, hue=hue, data=df, inner=None, 
                  palette=palette, ax=ax, alpha=0.5)
    
    # Add swarm plot on top
    sns.swarmplot(x=x, y=y, hue=hue, data=df, 
                 color="black", alpha=0.7, ax=ax, size=4, 
                 dodge=True if hue is not None else False)
    
    # Set title
    if title is None:
        title = f'Distribution of {y} by {x}'
    ax.set_title(title, fontsize=16)
    
    # Add summary statistics if requested
    if show_stats:
        # Get unique x values
        x_values = df[x].unique()
        
        for i, x_val in enumerate(x_values):
            # Filter data for this x value
            if hue is None:
                y_data = df[df[x] == x_val][y]
                
                # Calculate statistics
                mean = y_data.mean()
                median = y_data.median()
                
                # Add statistics text
                stats_text = f'Mean: {mean:.2f}\nMedian: {median:.2f}'
                
                # Position text above the violin
                text_x = i
                text_y = y_data.max() + (y_data.max() - y_data.min()) * 0.05
                
                ax.text(text_x, text_y, stats_text, ha='center', fontsize=9,
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
            else:
                # Handle grouped data if hue is provided
                # This is more complex and might clutter the plot, so we'll skip it
                pass
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust x-tick labels if needed
    if len(df[x].unique()) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_pca_explained_variance(X: np.ndarray, n_components: int = 10,
                              figsize: Tuple[int, int] = (12, 6),
                              title: str = 'PCA Explained Variance',
                              save_path: Optional[str] = None):
    """
    Create a plot showing explained variance for principal components.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    n_components : int, default=10
        Number of components to analyze
    figsize : tuple of int, default=(12, 6)
        Figure size (width, height) in inches
    title : str, default='PCA Explained Variance'
        Title of the plot
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    from sklearn.decomposition import PCA
    
    # Limit n_components to the number of features
    n_components = min(n_components, X.shape[1])
    
    # Fit PCA model
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot explained variance for each component
    components = range(1, n_components + 1)
    ax1.bar(components, pca.explained_variance_ratio_, alpha=0.8)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Explained Variance per Component')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(components, cumulative_variance, 'o-', linewidth=2)
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add threshold lines for common variance targets
    for threshold in [0.7, 0.8, 0.9, 0.95]:
        ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
        
        # Find the number of components that reach this threshold
        n_comp_threshold = np.argmax(cumulative_variance >= threshold) + 1
        
        # Add text annotation
        ax2.text(n_comp_threshold + 0.1, threshold - 0.02, 
                f'{threshold:.0%}: {n_comp_threshold} components', fontsize=9)
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_radar_chart(categories: List[str], values: List[List[float]], labels: List[str],
                   figsize: Tuple[int, int] = (10, 10), title: str = 'Radar Chart',
                   colors: Optional[List[str]] = None, fill_alpha: float = 0.25,
                   value_range: Optional[Tuple[float, float]] = None,
                   show_legend: bool = True, save_path: Optional[str] = None):
    """
    Create a radar (spider) chart for comparing multiple groups across different categories.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_geographical_data(df: pd.DataFrame, 
                        lat_col: str, lon_col: str, 
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
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
        
        # Create dummy scatter points for theimport matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec


def set_visualization_style(style: str = "whitegrid", context: str = "notebook", 
                           palette: str = "deep", font_scale: float = 1.2):
    """
    Set the default visualization style for plots with more customization options.
    
    Parameters:
    -----------
    style : str, default="whitegrid"
        The style of the plots. Options include: "darkgrid", "whitegrid", "dark", "white", "ticks"
    context : str, default="notebook"
        The context of the plots. Options include: "paper", "notebook", "talk", "poster"
    palette : str, default="deep"
        Color palette to use. See seaborn's documentation for options.
    font_scale : float, default=1.2
        Scaling factor for font sizes.
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


def plot_numeric_distribution(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                              n_cols: int = 3, figsize: Tuple[int, int] = None,
                              bins: int = 30, kde: bool = True, color: str = None,
                              save_path: Optional[str] = None):
    """
    Plot histograms for numeric columns in the DataFrame with enhanced options.
    
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
    bins : int, default=30
        Number of bins in histograms
    kde : bool, default=True
        Whether to overlay a kernel density estimate
    color : str, optional
        Color for the histograms
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
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
            axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='green', linestyle='-', alpha=0.8, label=f'Median: {median_val:.2f}')
            
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                           figsize: Tuple[int, int] = (12, 10), method: str = 'pearson',
                           mask_upper: bool = True, cmap: str = 'coolwarm',
                           save_path: Optional[str] = None):
    """
    Plot a correlation matrix for numeric columns in the DataFrame with enhanced options.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_feature_importance(feature_names: List[str], importance_values: List[float], 
                           title: str = 'Feature Importance', figsize: Tuple[int, int] = (12, 8),
                           color: str = '#1f77b4', top_n: Optional[int] = None,
                           horizontal: bool = True, show_values: bool = True,
                           save_path: Optional[str] = None):
    """
    Plot feature importance from a machine learning model with enhanced options.
    
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_scatter_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                       hue: Optional[str] = None, figsize: Tuple[int, int] = None,
                       diag_kind: str = "kde", corner: bool = False,
                       markers: Optional[str] = None, height: float = 2.5,
                       save_path: Optional[str] = None):
    """
    Plot a scatter matrix for numeric columns in the DataFrame with enhanced options.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return grid


def plot_categorical_counts(df: pd.DataFrame, columns: Optional[List[str]] = None,
                           n_cols: int = 2, figsize: Tuple[int, int] = None,
                           max_categories: int = 20, orientation: str = 'vertical',
                           palette: str = 'viridis', show_percentages: bool = True,
                           save_path: Optional[str] = None):
    """
    Plot count plots for categorical columns in the DataFrame.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
                
                bars = sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[i], palette=palette)
                
                # Add percentage labels if requested
                if show_percentages:
                    for j, (count, percentage) in enumerate(zip(value_counts, percentages)):
                        axes[i].text(count + (total_count * 0.01), j, f'{percentage}%', va='center')
                
                axes[i].set_xlabel('Count')
                axes[i].set_ylabel(col)
                # Add appropriate title with total count
                axes[i].set_title(f'{col} Distribution (n={total_count})')
                
            else:  # vertical
                bars = sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i], palette=palette)
                
                # Add percentage labels if requested
                if show_percentages:
                    for j, (count, percentage) in enumerate(zip(value_counts, percentages)):
                        axes[i].text(j, count + (total_count * 0.01), f'{percentage}%', ha='center', va='bottom')
                
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_boxplots(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None,
                 groupby_col: Optional[str] = None, n_cols: int = 2, 
                 figsize: Tuple[int, int] = None, palette: str = 'Set2',
                 save_path: Optional[str] = None):
    """
    Create box plots for numeric columns, optionally grouped by a categorical column.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
                sns.boxplot(x=groupby_col, y=col, data=df, ax=axes[i], palette=palette)
                
                # Rotate x-axis labels if necessary
                if len(df[groupby_col].unique()) > 4:
                    plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
            else:
                # Create simple boxplot
                sns.boxplot(y=df[col], ax=axes[i], color=sns.color_palette(palette)[i % len(sns.color_palette(palette))])
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_time_series(df: pd.DataFrame, date_col: str, value_cols: List[str],
                    groupby: Optional[str] = None, freq: Optional[str] = None,
                    agg_func: str = 'mean', figsize: Tuple[int, int] = (12, 6),
                    title: str = 'Time Series Plot', ylabel: str = 'Value',
                    plot_type: str = 'line', save_path: Optional[str] = None):
    """
    Plot time series data with flexible options.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_heatmap(data: Union[pd.DataFrame, np.ndarray], title: str = 'Heatmap',
               figsize: Tuple[int, int] = (10, 8), cmap: str = 'viridis',
               annot: bool = True, fmt: str = '.2f', linewidths: float = 0.5,
               xticklabels: Optional[List[str]] = None, yticklabels: Optional[List[str]] = None,
               vmin: Optional[float] = None, vmax: Optional[float] = None,
               save_path: Optional[str] = None):
    """
    Create a flexible heatmap for any 2D data.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (10, 8), normalize: bool = False,
                         title: str = 'Confusion Matrix', cmap: str = 'Blues',
                         fmt: str = 'd', save_path: Optional[str] = None):
    """
    Plot a confusion matrix for classification results.
    
    Parameters:
    -----------
    cm : np.ndarray
        Confusion matrix array
    class_names : list of str, optional
        Names of the classes. If None, uses indices.
    figsize : tuple of int, default=(10, 8)
        Figure size (width, height) in inches
    normalize : bool, default=False
        Whether to normalize the confusion matrix
    title : str, default='Confusion Matrix'
        Title of the plot
    cmap : str, default='Blues'
        Colormap to use
    fmt : str, default='d'
        Format string for annotations. Use 'd' for integers, '.2f' for floats.
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Create class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = f'Normalized {title}'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, square=True,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, cbar=True, ax=ax)
    
    # Set labels
    ax.set_xlabel('Predicted label', fontsize=14)
    ax.set_ylabel('True label', fontsize=14)
    ax.set_title(title, fontsize=16)
    
    # Add overall accuracy or other metrics if normalized
    if normalize:
        diag_sum = np.trace(cm)
        n_classes = cm.shape[0]
        ax.text(n_classes - 0.5, -0.5, f'Overall Accuracy: {diag_sum/n_classes:.2f}',
                ha='right', va='center', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_learning_curve(train_sizes: np.ndarray, train_scores: np.ndarray, test_scores: np.ndarray,
                       title: str = 'Learning Curve', figsize: Tuple[int, int] = (10, 6),
                       ylim: Optional[Tuple[float, float]] = None, xlabel: str = 'Training examples',
                       ylabel: str = 'Score', fill_std: bool = True,
                       save_path: Optional[str] = None):
    """
    Plot a learning curve from cross-validation results.
    
    Parameters:
    -----------
    train_sizes : np.ndarray
        Training set sizes
    train_scores : np.ndarray
        Scores on training sets (shape: n_sizes x n_cv_folds)
    test_scores : np.ndarray
        Scores on test sets (shape: n_sizes x n_cv_folds)
    title : str, default='Learning Curve'
        Title of the plot
    figsize : tuple of int, default=(10, 6)
        Figure size (width, height) in inches
    ylim : tuple of float, optional
        Y-axis limits
    xlabel : str, default='Training examples'
        Label for the x-axis
    ylabel : str, default='Score'
        Label for the y-axis
    fill_std : bool, default=True
        Whether to fill the standard deviation area
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate means and standard deviations
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot learning curves
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot training and test scores
    ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    ax.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    
    # Fill standard deviation area if requested
    if fill_std:
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color='r')
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color='g')
    
    # Set plot attributes
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(*ylim)
    
    ax.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_roc_curve(fpr: Union[List[float], np.ndarray], tpr: Union[List[float], np.ndarray], auc: float,
                  title: str = 'ROC Curve', figsize: Tuple[int, int] = (8, 8),
                  label: Optional[str] = None, plot_diagonal: bool = True,
                  save_path: Optional[str] = None):
    """
    Plot a Receiver Operating Characteristic (ROC) curve.
    
    Parameters:
    -----------
    fpr : array-like
        False positive rates
    tpr : array-like
        True positive rates
    auc : float
        Area under the ROC curve
    title : str, default='ROC Curve'
        Title of the plot
    figsize : tuple of int, default=(8, 8)
        Figure size (width, height) in inches
    label : str, optional
        Label for the ROC curve. If None, uses 'ROC curve (AUC = {auc:.2f})'
    plot_diagonal : bool, default=True
        Whether to plot the diagonal line representing random classification
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create label if not provided
    if label is None:
        label = f'ROC curve (AUC = {auc:.2f})'
    
    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=label)
    
    # Plot diagonal line if requested
    if plot_diagonal:
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
    
    # Set plot attributes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_multi_roc_curves(fpr_dict: Dict[str, Union[List[float], np.ndarray]], 
                        tpr_dict: Dict[str, Union[List[float], np.ndarray]],
                        auc_dict: Dict[str, float], title: str = 'ROC Curves',
                        figsize: Tuple[int, int] = (10, 8), plot_diagonal: bool = True,
                        save_path: Optional[str] = None):
    """
    Plot multiple ROC curves for multi-class classification or multiple models.
    
    Parameters:
    -----------
    fpr_dict : dict
        Dictionary with keys as curve names and values as false positive rates
    tpr_dict : dict
        Dictionary with keys as curve names and values as true positive rates
    auc_dict : dict
        Dictionary with keys as curve names and values as AUC scores
    title : str, default='ROC Curves'
        Title of the plot
    figsize : tuple of int, default=(10, 8)
        Figure size (width, height) in inches
    plot_diagonal : bool, default=True
        Whether to plot the diagonal line representing random classification
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each ROC curve
    for name in fpr_dict.keys():
        ax.plot(fpr_dict[name], tpr_dict[name], 
                label=f'{name} (AUC = {auc_dict[name]:.2f})')
    
    # Plot diagonal line if requested
    if plot_diagonal:
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    
    # Set plot attributes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_precision_recall_curve(precision: Union[List[float], np.ndarray], 
                              recall: Union[List[float], np.ndarray],
                              average_precision: float, title: str = 'Precision-Recall Curve',
                              figsize: Tuple[int, int] = (8, 8), label: Optional[str] = None,
                              save_path: Optional[str] = None):
    """
    Plot a precision-recall curve.
    
    Parameters:
    -----------
    precision : array-like
        Precision values
    recall : array-like
        Recall values
    average_precision : float
        Average precision score
    title : str, default='Precision-Recall Curve'
        Title of the plot
    figsize : tuple of int, default=(8, 8)
        Figure size (width, height) in inches
    label : str, optional
        Label for the curve. If None, uses 'Precision-Recall (AP = {average_precision:.2f})'
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create label if not provided
    if label is None:
        label = f'Precision-Recall (AP = {average_precision:.2f})'
    
    # Plot precision-recall curve
    ax.plot(recall, precision, lw=2, label=label)
    
    # Set plot attributes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def create_facet_grid(df: pd.DataFrame, x_var: str, y_var: str, hue: Optional[str] = None,
                     col: Optional[str] = None, row: Optional[str] = None, 
                     kind: str = 'scatter', col_wrap: Optional[int] = None,
                     height: float = 4, aspect: float = 1.2, palette: str = 'deep',
                     title: Optional[str] = None, save_path: Optional[str] = None):
    """
    Create a flexible facet grid for visualizing relationships across multiple variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to visualize
    x_var : str
        Column name for the x-axis
    y_var : str
        Column name for the y-axis
    hue : str, optional
        Column name for color encoding
    col : str, optional
        Column name for creating column facets
    row : str, optional
        Column name for creating row facets
    kind : str, default='scatter'
        Kind of plot: 'scatter', 'line', 'reg' (regression), 'hex', 'kde', 'hist'
    col_wrap : int, optional
        Maximum number of columns in the grid
    height : float, default=4
        Height (in inches) of each facet
    aspect : float, default=1.2
        Aspect ratio of each facet
    palette : str, default='deep'
        Color palette to use
    title : str, optional
        Title for the entire grid
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    grid : seaborn.FacetGrid or seaborn.JointGrid
        The generated grid object for further customization
    """
    # Create appropriate grid based on parameters
    if col is None and row is None:
        if kind == 'scatter':
            # Create a joint plot for single pair of variables
            grid = sns.jointplot(x=x_var, y=y_var, data=df, hue=hue, kind=kind,
                                height=height * 2, ratio=10, palette=palette)
        else:
            # Create simple scatter plot for single pair of variables
            plt.figure(figsize=(height * aspect, height))
            if kind == 'line':
                grid = sns.lineplot(x=x_var, y=y_var, data=df, hue=hue, palette=palette)
            elif kind == 'reg':
                grid = sns.regplot(x=x_var, y=y_var, data=df, scatter_kws={'alpha': 0.7})
            elif kind == 'hex':
                grid = sns.jointplot(x=x_var, y=y_var, data=df, kind='hex', height=height * 2)
            elif kind == 'kde':
                grid = sns.jointplot(x=x_var, y=y_var, data=df, kind='kde', height=height * 2)
            elif kind == 'hist':
                grid = sns.jointplot(x=x_var, y=y_var, data=df, kind='hist', height=height * 2)
            else:
                grid = sns.scatterplot(x=x_var, y=y_var, data=df, hue=hue, palette=palette)
            
            plt.xlabel(x_var, fontsize=14)
            plt.ylabel(y_var, fontsize=14)
            
            if title:
                plt.title(title, fontsize=16)
    else:
        # Create facet grid
        grid = sns.FacetGrid(df, col=col, row=row, hue=hue, height=height, 
                           aspect=aspect, palette=palette, col_wrap=col_wrap)
        
        # Map appropriate plot kind
        if kind == 'scatter':
            grid.map(sns.scatterplot, x_var, y_var, alpha=0.7)
        elif kind == 'line':
            grid.map(sns.lineplot, x_var, y_var)
        elif kind == 'reg':
            grid.map(sns.regplot, x_var, y_var)
        elif kind == 'kde':
            grid.map(sns.kdeplot, x_var, y_var)
        elif kind == 'hist':
            # For histogram, we'll just plot x_var
            grid.map(sns.histplot, x_var)
        else:
            grid.map(sns.scatterplot, x_var, y_var)
        
        # Add legend
        if hue:
            grid.add_legend()
        
        # Set title if provided
        if title:
            grid.fig.suptitle(title, fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return grid


def plot_interactive_components(df: pd.DataFrame,
                              title: str = 'Interactive Visualization Dashboard',
                              figsize: Tuple[int, int] = (18, 12)):
    """
    Create an interactive dashboard with multiple visualization components.
    Note: This function is meant to be used in Jupyter notebooks with ipywidgets installed.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to visualize
    title : str, default='Interactive Visualization Dashboard'
        Title of the dashboard
    figsize : tuple of int, default=(18, 12)
        Figure size (width, height) in inches
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("This function requires ipywidgets. Install with: pip install ipywidgets")
        return
    
    # Get column lists by type
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Define visualization options
    viz_types = ['Histogram', 'Boxplot', 'Scatter Plot', 'Bar Chart', 'Correlation Heatmap']
    
    # Create widgets
    viz_type_widget = widgets.Dropdown(options=viz_types, description='Chart Type:')
    
    x_col_widget = widgets.Dropdown(options=['None'] + numeric_cols + categorical_cols, description='X-axis:')
    y_col_widget = widgets.Dropdown(options=['None'] + numeric_cols, description='Y-axis:')
    hue_col_widget = widgets.Dropdown(options=['None'] + categorical_cols, description='Color by:')
    
    # Create output widget for the plot
    output = widgets.Output()
    
    # Create a function to update the plot
    def update_plot(*args):
        with output:
            # Clear previous plot
            output.clear_output(wait=True)
            
            # Create figure
            plt.figure(figsize=figsize)
            
            # Get selected values
            viz_type = viz_type_widget.value
            x_col = None if x_col_widget.value == 'None' else x_col_widget.value
            y_col = None if y_col_widget.value == 'None' else y_col_widget.value
            hue_col = None if hue_col_widget.value == 'None' else hue_col_widget.value
            
            # Create appropriate plot based on selected type
            try:
                if viz_type == 'Histogram':
                    if x_col:
                        plt.title(f'Distribution of {x_col}', fontsize=16)
                        sns.histplot(data=df, x=x_col, hue=hue_col, kde=True)
                    else:
                        plt.text(0.5, 0.5, "Please select an X-axis variable", ha='center', va='center', fontsize=14)
                
                elif viz_type == 'Boxplot':
                    if x_col and y_col:
                        plt.title(f'Boxplot of {y_col} by {x_col}', fontsize=16)
                        sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col)
                    elif y_col:
                        plt.title(f'Boxplot of {y_col}', fontsize=16)
                        sns.boxplot(data=df, y=y_col)
                    else:
                        plt.text(0.5, 0.5, "Please select a Y-axis variable", ha='center', va='center', fontsize=14)
                
                elif viz_type == 'Scatter Plot':
                    if x_col and y_col:
                        plt.title(f'Scatter Plot of {y_col} vs {x_col}', fontsize=16)
                        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col)
                    else:
                        plt.text(0.5, 0.5, "Please select both X and Y axis variables", ha='center', va='center', fontsize=14)
                
                elif viz_type == 'Bar Chart':
                    if x_col:
                        if y_col:
                            # Grouped bar chart with aggregation
                            plt.title(f'Average {y_col} by {x_col}', fontsize=16)
                            sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col)
                        else:
                            # Count bar chart
                            plt.title(f'Count of {x_col}', fontsize=16)
                            sns.countplot(data=df, x=x_col, hue=hue_col)
                    else:
                        plt.text(0.5, 0.5, "Please select an X-axis variable", ha='center', va='center', fontsize=14)
                
                elif viz_type == 'Correlation Heatmap':
                    if len(numeric_cols) > 1:
                        corr = df[numeric_cols].corr()
                        plt.title('Correlation Matrix', fontsize=16)
                        mask = np.triu(np.ones_like(corr, dtype=bool))
                        sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', 
                                    vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
                    else:
                        plt.text(0.5, 0.5, "Need at least 2 numeric columns for a correlation heatmap", 
                                ha='center', va='center', fontsize=14)
                
                plt.tight_layout()
                plt.show()
            
            except Exception as e:
                plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=14)
                plt.axis('off')
                plt.show()
    
    # Connect the update function to the widgets
    viz_type_widget.observe(update_plot, names='value')
    x_col_widget.observe(update_plot, names='value')
    y_col_widget.observe(update_plot, names='value')
    hue_col_widget.observe(update_plot, names='value')
    
    # Create dashboard layout
    dashboard = widgets.VBox([
        widgets.HTML(f"<h2>{title}</h2>"),
        widgets.HBox([
            widgets.VBox([viz_type_widget, x_col_widget, y_col_widget, hue_col_widget]),
        ]),
        output
    ])
    
    # Display the dashboard
    display(dashboard)
    
    # Trigger initial plot
    update_plot()


def plot_residuals(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray],
                 figsize: Tuple[int, int] = (12, 8), title: str = 'Residual Analysis',
                 save_path: Optional[str] = None):
    """
    Create a comprehensive residual analysis plot for regression models.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    figsize : tuple of int, default=(12, 8)
        Figure size (width, height) in inches
    title : str, default='Residual Analysis'
        Title of the plot
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Calculate residuals
    residuals = np.array(y_true) - np.array(y_pred)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Scatter plot of predicted vs. actual values
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, y_true, alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax1.set_xlabel('Predicted values', fontsize=12)
    ax1.set_ylabel('Actual values', fontsize=12)
    ax1.set_title('Predicted vs. Actual Values', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_pred, residuals, alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--')
    
    ax2.set_xlabel('Predicted values', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residuals vs. Predicted Values', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(residuals, kde=True, ax=ax3)
    
    ax3.set_xlabel('Residual value', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Residuals', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Q-Q plot of residuals
    ax4 = fig.add_subplot(gs[1, 1])
    from scipy import stats
    stats.probplot(residuals, plot=ax4)
    
    ax4.set_title('Q-Q Plot of Residuals', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # Add metrics
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    r2 = 1 - np.sum(residuals ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    metrics_text = (
        f'MSE: {mse:.4f}\n'
        f'RMSE: {rmse:.4f}\n'
        f'MAE: {mae:.4f}\n'
        f'R²: {r2:.4f}'
    )
    
    fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_feature_distributions_by_target(df: pd.DataFrame, features: List[str], 
                                       target: str, bins: int = 30,
                                       figsize: Tuple[int, int] = None,
                                       n_cols: int = 3, save_path: Optional[str] = None):
    """
    Plot distributions of features grouped by a categorical target variable.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def create_correlation_network(df: pd.DataFrame, threshold: float = 0.5, 
                             figsize: Tuple[int, int] = (10, 10),
                             node_size_factor: float = 3000, 
                             title: str = "Feature Correlation Network",
                             save_path: Optional[str] = None):
    """
    Create a network visualization of correlations between features.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_cluster_analysis(df: pd.DataFrame, features: List[str], cluster_labels: np.ndarray,
                        method: str = 'pca', n_components: int = 2,
                        figsize: Tuple[int, int] = (12, 10),
                        title: str = 'Cluster Analysis', 
                        save_path: Optional[str] = None):
    """
    Visualize clusters using dimensionality reduction techniques.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data used for clustering
    features : list of str
        Features used for clustering
    cluster_labels : array-like
        Cluster assignments for each data point
    method : str, default='pca'
        Dimensionality reduction method: 'pca', 'tsne', or 'umap'
    n_components : int, default=2
        Number of components for the dimensionality reduction (2 or 3)
    figsize : tuple of int, default=(12, 10)
        Figure size (width, height) in inches
    title : str, default='Cluster Analysis'
        Title of the plot
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Check valid n_components
    if n_components not in [2, 3]:
        print("n_components must be 2 or 3. Setting to 2.")
        n_components = 2
    
    # Get feature data
    X = df[features].values
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(X)
        method_name = 'PCA'
    elif method.lower() == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(X)
        method_name = 't-SNE'
    elif method.lower() == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            embedding = reducer.fit_transform(X)
            method_name = 'UMAP'
        except ImportError:
            print("UMAP is not installed. Install with: pip install umap-learn")
            print("Falling back to PCA.")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, random_state=42)
            embedding = reducer.fit_transform(X)
            method_name = 'PCA'
    else:
        print(f"Unknown method: {method}. Using PCA instead.")
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(X)
        method_name = 'PCA'
    
    # Create figure based on number of components
    if n_components == 2:
        # 2D plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot with cluster colors
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, 
                           cmap='viridis', s=50, alpha=0.8, edgecolors='w')
        
        # Add legend
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        
        # Add labels
        ax.set_xlabel(f'{method_name} Component 1', fontsize=14)
        ax.set_ylabel(f'{method_name} Component 2', fontsize=14)
        
    else:
        # 3D plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot with cluster colors
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                           c=cluster_labels, cmap='viridis', s=50, alpha=0.8, edgecolors='w')
        
        # Add legend
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        
        # Add labels
        ax.set_xlabel(f'{method_name} Component 1', fontsize=14)
        ax.set_ylabel(f'{method_name} Component 2', fontsize=14)
        ax.set_zlabel(f'{method_name} Component 3', fontsize=14)
    
    # Add title
    plt.title(f'{title} ({method_name} Projection)', fontsize=16)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    return fig


def plot_decision_boundaries(X: np.ndarray, y: np.ndarray, model, feature_names: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (12, 10), title: str = 'Decision Boundaries',
                           save_path: Optional[str] = None):
    """
    Visualize decision boundaries of a classifier in 2D space.
    If X has more than 2 dimensions, PCA is used to reduce it to 2D.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target labels
    model : object
        Trained classifier with predict method
    feature_names : list of str, optional
        Names of the features if X has only 2 dimensions
    figsize : tuple of int, default=(12, 10)
        Figure size (width, height) in inches
    title : str, default='Decision Boundaries'
        Title of the plot
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Check if we need dimensionality reduction
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        feature_1 = "PCA Component 1"
        feature_2 = "PCA Component 2"
        transformed = True
    else:
        X_2d = X
        feature_1 = feature_names[0] if feature_names else "Feature 1"
        feature_2 = feature_names[1] if feature_names else "Feature 2"
        transformed = False
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define mesh grid for decision boundary
    x_min, x_max = X_2d[:, 0].min() - 0.1, X_2d[:, 0].max() + 0.1
    y_min, y_max = X_2d[:, 1].min() - 0.1, X_2d[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Reshape for prediction
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # For PCA transformed data, we need to approximate predictions
    if transformed:
        # This is just an approximation of the decision boundary in PCA space
        # Ideally, we'd need to find the inverse transform, but not all models support this
        try:
            Z = model.predict(mesh_points)
        except:
            # If model can't predict on 2D data (because it was trained on more dimensions)
            # we display a warning
            plt.text(0.5, 0.5, "Cannot display decision boundaries for dimensionality-reduced data with this model",
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', edgecolors='k', alpha=0.8)
            plt.colorbar(label='Class')
            plt.xlabel(feature_1, fontsize=14)
            plt.ylabel(feature_2, fontsize=14)
            plt.title(title, fontsize=16)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            plt.show()
            return fig
    else:
        Z = model.predict(mesh_points)
    
    # Reshape back to mesh shape
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # Plot data points
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', 
                       edgecolors='k', alpha=0.8)
    
    # Add legend if there aren't too many classes
    if len(np.unique(y)) <= 10:
        legend = plt.legend(*scatter.legend_elements(), title="Classes")
        plt.gca().add_artist(legend)
    else:
        plt.colorbar(label='Class')
    
    # Add labels
    plt.xlabel(feature_1, fontsize=14)
    plt.ylabel(feature_2, fontsize=14)
    
    # Add title
    if transformed:
        plt.title(f"{title} (PCA Projection)", fontsize=16)
    else:
        plt.title(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_model_comparison(model_names: List[str], metric_values: List[float], 
                        metric_name: str = 'Accuracy', ci_values: Optional[List[Tuple[float, float]]] = None,
                        figsize: Tuple[int, int] = (10, 6), color: str = '#4472C4',
                        sort_values: bool = True, title: str = 'Model Comparison',
                        annotate_values: bool = True, save_path: Optional[str] = None):
    """
    Create a bar chart comparing performance metrics across multiple models.
    
    Parameters:
    -----------
    model_names : list of str
        Names of the models to compare
    metric_values : list of float
        Performance metric values for each model
    metric_name : str, default='Accuracy'
        Name of the metric being compared
    ci_values : list of tuple, optional
        Confidence intervals for each metric value as (lower, upper) tuples
    figsize : tuple of int, default=(10, 6)
        Figure size (width, height) in inches
    color : str, default='#4472C4'
        Color for the bars
    sort_values : bool, default=True
        Whether to sort bars by metric value (descending)
    title : str, default='Model Comparison'
        Title of the plot
    annotate_values : bool, default=True
        Whether to show metric values on the bars
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object for further customization
    """
    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'Model': model_names,
        metric_name: metric_values
    })
    
    # Sort by metric values if requested
    if sort_values:
        df = df.sort_values(by=metric_name, ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    bars = ax.bar(df['Model'], df[metric_name], color=color, alpha=0.8, width=0.6)
    
    # Add confidence intervals if provided
    if ci_values is not None:
        # Sort CI values if the models were sorted
        if sort_values:
            ci_mapping = {name: ci for name, ci in zip(model_names, ci_values)}
            sorted_ci_values = [ci_mapping[name] for name in df['Model']]
        else:
            sorted_ci_values = ci_values
        
        # Add error bars
        yerr = np.array([(val - ci[0], ci[1] - val) for val, ci in 
                         zip(df[metric_name], sorted_ci_values)]).T
        ax.errorbar(df['Model'], df[metric_name], yerr=yerr, fmt='none', 
                   ecolor='black', capsize=5)
    
    # Add value labels if requested
    if annotate_values:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.set_title(title, fontsize=16)
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust x-axis labels if there are many models
    if len(model_names) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig


def visualize_dataset_summary(df: pd.DataFrame, title: str = "Dataset Summary Visualization",
                            figsize: Tuple[int, int] = (18, 12), max_categories: int = 10,
                            save_path: Optional[str] = None):
    """
    Create a comprehensive visualization dashboard for a dataset, showing key statistics,
    data types, missing values, and distributions.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig