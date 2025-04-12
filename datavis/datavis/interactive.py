"""
Interactive visualization functions.

This module contains functions for creating interactive visualizations and dashboards.
Note: Most functions require ipywidgets in a Jupyter notebook environment.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union, Any

from .core import save_figure


def create_interactive_dashboard(df: pd.DataFrame, 
                               title: str = "Interactive Data Dashboard"):
    """
    Create a simplified interactive dashboard with basic visualization options.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to visualize
    title : str, default="Interactive Data Dashboard"
        Title of the dashboard
        
    Notes
    -----
    This function requires running in a Jupyter notebook with ipywidgets installed.
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
                'Violin Plot', 'Heatmap', 'Pair Plot']
    
    # Create output widget for the visualization
    output = widgets.Output()
    
    # Create tab structure
    tab_data = widgets.Output()
    tab_viz = widgets.Output()
    
    tabs = widgets.Tab(children=[tab_data, tab_viz])
    tabs.set_title(0, 'Data Preview')
    tabs.set_title(1, 'Visualizations')
    
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
        
        # Create data display output
        data_output = widgets.Output()
        
        # Function to update filter value widget status
        def update_filter_value_status(change):
            filter_value.disabled = filter_col.value == 'None'
        
        filter_col.observe(update_filter_value_status, names='value')
        
        # Function to update data display
        def update_data_display(change):
            with data_output:
                clear_output(wait=True)
                
                # Apply filter if selected
                filtered_df = df
                if filter_col.value != 'None' and filter_value.value:
                    try:
                        # Handle different types of filters
                        col = filter_col.value
                        val = filter_value.value
                        
                        if filtered_df[col].dtype in ['int64', 'float64']:
                            # For numeric columns, try to convert the value
                            try:
                                numeric_val = float(val)
                                filtered_df = filtered_df[filtered_df[col] == numeric_val]
                            except ValueError:
                                print(f"Error: '{val}' is not a valid number for column '{col}'")
                        else:
                            # For non-numeric columns, use string comparison
                            filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(val)]
                    except Exception as e:
                        print(f"Error applying filter: {e}")
                
                # Display the data
                display(filtered_df.head(n_rows.value))
        
        # Connect the widgets to the update function
        filter_col.observe(update_data_display, names='value')
        filter_value.observe(update_data_display, names='value')
        n_rows.observe(update_data_display, names='value')
        
        # Layout the widgets
        filter_controls = widgets.HBox([filter_col, filter_value, n_rows])
        display(filter_controls)
        display(data_output)
        
        # Show initial data view
        update_data_display(None)
    
    # VISUALIZATION TAB
    with tab_viz:
        # Create visualization control widgets
        viz_type = widgets.Dropdown(
            options=viz_types,
            value='Histogram',
            description='Chart Type:',
            style={'description_width': 'initial'}
        )
        
        x_axis = widgets.Dropdown(
            options=['None'] + df.columns.tolist(),
            value='None',
            description='X-Axis:',
            style={'description_width': 'initial'}
        )
        
        y_axis = widgets.Dropdown(
            options=['None'] + numeric_cols,
            value='None',
            description='Y-Axis:',
            style={'description_width': 'initial'},
            disabled=False
        )
        
        color_by = widgets.Dropdown(
            options=['None'] + categorical_cols,
            value='None',
            description='Color By:',
            style={'description_width': 'initial'}
        )
        
        # Function to update visualization
        def update_visualization(change):
            with output:
                clear_output(wait=True)
                
                if x_axis.value == 'None':
                    print("Please select an X-axis variable")
                    return
                
                plt.figure(figsize=(10, 6))
                
                try:
                    # Different visualization types
                    if viz_type.value == 'Histogram':
                        if x_axis.value in numeric_cols:
                            sns.histplot(data=df, x=x_axis.value, hue=color_by.value if color_by.value != 'None' else None)
                        else:
                            sns.countplot(data=df, x=x_axis.value, hue=color_by.value if color_by.value != 'None' else None)
                    
                    elif viz_type.value == 'Scatter Plot':
                        if y_axis.value == 'None':
                            print("Please select a Y-axis variable for a scatter plot")
                            return
                        sns.scatterplot(data=df, x=x_axis.value, y=y_axis.value, 
                                      hue=color_by.value if color_by.value != 'None' else None)
                    
                    elif viz_type.value == 'Line Chart':
                        if y_axis.value == 'None':
                            print("Please select a Y-axis variable for a line chart")
                            return
                        sns.lineplot(data=df, x=x_axis.value, y=y_axis.value, 
                                   hue=color_by.value if color_by.value != 'None' else None)
                    
                    elif viz_type.value == 'Bar Chart':
                        if y_axis.value == 'None':
                            sns.countplot(data=df, x=x_axis.value, hue=color_by.value if color_by.value != 'None' else None)
                        else:
                            sns.barplot(data=df, x=x_axis.value, y=y_axis.value, 
                                      hue=color_by.value if color_by.value != 'None' else None)
                    
                    elif viz_type.value == 'Box Plot':
                        if y_axis.value == 'None':
                            print("Please select a Y-axis variable for a box plot")
                            return
                        sns.boxplot(data=df, x=x_axis.value, y=y_axis.value, 
                                   hue=color_by.value if color_by.value != 'None' else None)
                    
                    elif viz_type.value == 'Violin Plot':
                        if y_axis.value == 'None':
                            print("Please select a Y-axis variable for a violin plot")
                            return
                        sns.violinplot(data=df, x=x_axis.value, y=y_axis.value, 
                                     hue=color_by.value if color_by.value != 'None' else None)
                    
                    elif viz_type.value == 'Heatmap':
                        if y_axis.value == 'None':
                            print("Please select a Y-axis variable for a heatmap")
                            return
                        if x_axis.value in categorical_cols and y_axis.value in categorical_cols:
                            # Create a crosstab for categorical variables
                            heatmap_data = pd.crosstab(df[y_axis.value], df[x_axis.value])
                            sns.heatmap(heatmap_data, annot=True, cmap='viridis')
                        elif x_axis.value in numeric_cols and y_axis.value in numeric_cols:
                            # Create a 2D histogram for numeric variables
                            plt.hist2d(df[x_axis.value], df[y_axis.value], bins=20, cmap='viridis')
                            plt.colorbar(label='Count')
                        else:
                            print("Heatmap requires both X and Y to be the same type (both categorical or both numeric)")
                            return
                    
                    elif viz_type.value == 'Pair Plot':
                        if color_by.value != 'None':
                            variables = [col for col in [x_axis.value, y_axis.value] if col != 'None']
                            if len(variables) < 2:
                                print("Please select both X and Y variables for a pair plot")
                                return
                            sns.pairplot(df[variables + [color_by.value]], hue=color_by.value)
                        else:
                            print("Please select a 'Color By' variable for the pair plot")
                            return
                    
                    plt.title(f"{viz_type.value} of {y_axis.value if y_axis.value != 'None' else ''} "
                             f"by {x_axis.value}")
                    plt.tight_layout()
                    plt.show()
                    
                except Exception as e:
                    print(f"Error creating visualization: {e}")
        
        # Connect widgets to the update function
        viz_type.observe(update_visualization, names='value')
        x_axis.observe(update_visualization, names='value')
        y_axis.observe(update_visualization, names='value')
        color_by.observe(update_visualization, names='value')
        
        # Layout the visualization controls
        viz_controls = widgets.VBox([
            widgets.HBox([viz_type]),
            widgets.HBox([x_axis, y_axis]),
            widgets.HBox([color_by])
        ])
        
        display(viz_controls)
        display(output)
        
        # Show initial visualization
        update_visualization(None)
    
    # Display the tabs
    display(widgets.HTML(f"<h2>{title}</h2>"))
    display(tabs)
    
    return tabs


def plot_interactive_time_series(df: pd.DataFrame, 
                               date_col: str, 
                               value_cols: List[str],
                               title: str = 'Interactive Time Series Plot',
                               figsize: Tuple[int, int] = (15, 8),
                               date_format: str = '%Y-%m-%d',
                               show_secondary_axis: bool = False):
    """
    Create an interactive time series plot with sliders for date range selection.
    
    Parameters
    ----------
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
        
    Notes
    -----
    This function requires running in a Jupyter notebook with ipywidgets installed.
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


def create_interactive_components(df: pd.DataFrame,
                                title: str = 'Interactive Visualization Dashboard',
                                figsize: Tuple[int, int] = (18, 12)):
    """
    Create an interactive dashboard with multiple visualization components.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to visualize
    title : str, default='Interactive Visualization Dashboard'
        Title of the dashboard
    figsize : tuple of int, default=(18, 12)
        Figure size (width, height) in inches
        
    Notes
    -----
    This function requires running in a Jupyter notebook with ipywidgets installed.
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


def create_animated_chart(df: pd.DataFrame, 
                        x_col: str, 
                        y_col: str, 
                        time_col: str,
                        color_col: Optional[str] = None, 
                        size_col: Optional[str] = None,
                        title: str = 'Animated Chart', 
                        fps: int = 5,
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None):
    """
    Create an animated scatter plot showing changes over time.
    
    Parameters
    ----------
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
    
    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object
        
    Notes
    -----
    This function requires matplotlib animation support.
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