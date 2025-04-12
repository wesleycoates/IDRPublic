"""
Data Visualization Package

A comprehensive package for data visualization and analysis.
"""

__version__ = '0.1.0'

# Import core functionality
from .core import set_visualization_style, adjust_color_brightness, save_figure, get_columns_by_type

# Import basic visualization functions
from .basic import (
    plot_numeric_distribution,
    plot_categorical_counts,
    plot_boxplots,
    plot_time_series,
    plot_heatmap
)

# Import statistical visualization functions
from .statistical import (
    plot_correlation_matrix,
    plot_feature_distributions_by_target,
    plot_feature_importance,
    plot_scatter_matrix,
    create_correlation_network
)

# Import machine learning visualization functions
from .ml import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_learning_curve,
    plot_decision_boundaries,
    plot_cluster_analysis,
    plot_silhouette_analysis,
    plot_model_comparison,
    plot_residuals
)

# Import advanced visualization functions
from .advanced import (
    plot_parallel_coordinates,
    plot_3d_surface,
    plot_waffle_chart,
    plot_sunburst,
    plot_dendrogram,
    plot_biplot,
    plot_radar_chart,
    create_bubble_chart,
    plot_geographical_data,
    create_stacked_area_chart,
    create_sankey_diagram,
    plot_calendar_heatmap
)

# Import interactive visualization components
from .interactive import (
    create_interactive_dashboard,
    plot_interactive_time_series,
    create_interactive_components,
    create_animated_chart
)

# Import utility functions
from .utils import visualize_dataset_summary