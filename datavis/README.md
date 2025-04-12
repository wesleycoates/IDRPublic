# DataVis - Comprehensive Data Visualization Package

DataVis is a Python package that provides a comprehensive set of data visualization functions for data analysis and machine learning. It builds on top of matplotlib, seaborn, and other visualization libraries to provide a consistent and user-friendly interface for creating high-quality visualizations.

## Features

- **Basic visualizations**: Histograms, bar charts, box plots, etc.
- **Statistical visualizations**: Correlation matrices, feature distributions, etc.
- **Machine learning visualizations**: Model evaluation, clustering analysis, etc.
- **Advanced visualizations**: 3D surfaces, calendar heatmaps, parallel coordinates, etc.
- **Interactive visualizations**: Interactive dashboards, time series explorers, etc.
- **Utilities**: Dataset summaries, correlation networks, etc.

## Installation

```bash
# Install from PyPI
pip install datavis

# For development installation
git clone https://github.com/yourusername/datavis.git
cd datavis
pip install -e .
```

## Dependencies

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- IPython & ipywidgets (for interactive visualizations)
- Scikit-learn (for Sankey diagrams and sunburst charts)
- Cartopy (optional, for geographical visualizations)

## Quick Start

```python
import pandas as pd
import matplotlib.pyplot as plt
from datavis import set_visualization_style, plot_numeric_distribution, visualize_dataset_summary

# Set visualization style
set_visualization_style(style="whitegrid", context="notebook", palette="viridis")

# Load your data
df = pd.read_csv("your_data.csv")

# Create a dataset summary visualization
visualize_dataset_summary(df, title="Dataset Overview")

# Plot distributions of numeric columns
plot_numeric_distribution(df, n_cols=3)

plt.show()
```

## Module Structure

The package is organized into modules by functionality:

- **core**: Style settings and fundamental utilities
- **basic**: Basic plotting functions (histograms, scatter plots, etc.)
- **statistical**: Statistical visualization (correlation, distributions, etc.)
- **ml**: Machine learning visualizations (decision boundaries, clusters, etc.)
- **advanced**: Advanced plots (3D, parallel coordinates, etc.)
- **interactive**: Interactive visualizations and dashboards
- **utils**: Utility functions for visualization

## Usage Examples

### Basic Visualizations

```python
from datavis import plot_numeric_distribution, plot_categorical_counts, plot_boxplots

# Plot distributions of numeric columns
plot_numeric_distribution(df, columns=['numeric1', 'numeric2', 'numeric3'])

# Plot counts of categorical columns
plot_categorical_counts(df, columns=['category1', 'category2'])

# Create boxplots grouped by a categorical variable
plot_boxplots(df, numeric_cols=['numeric1', 'numeric2'], groupby_col='category')
```

### Statistical Visualizations

```python
from datavis import plot_correlation_matrix, plot_feature_importance

# Create a correlation matrix
plot_correlation_matrix(df, mask_upper=True)

# Visualize feature importance from a machine learning model
plot_feature_importance(feature_names=['f1', 'f2', 'f3'], 
                       importance_values=[0.5, 0.3, 0.2], 
                       horizontal=True)
```

### Machine Learning Visualizations

```python
from datavis import plot_confusion_matrix, plot_roc_curve, plot_learning_curve

# Plot a confusion matrix
cm = [[50, 5], [3, 42]]  # Example confusion matrix
plot_confusion_matrix(cm, class_names=['Negative', 'Positive'])

# Plot ROC curve
plot_roc_curve(fpr=[0, 0.2, 0.5, 0.8, 1], tpr=[0, 0.7, 0.8, 0.9, 1], auc=0.85)

# Plot learning curve
plot_learning_curve(train_sizes, train_scores, test_scores)
```

### Advanced Visualizations

```python
from datavis import plot_radar_chart, create_bubble_chart, plot_calendar_heatmap

# Create a radar chart
plot_radar_chart(
    categories=['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E'],
    values=[[4, 3, 5, 2, 4], [3, 4, 3, 5, 2]],
    labels=['Group 1', 'Group 2']
)

# Create a bubble chart
create_bubble_chart(df, x_col='x', y_col='y', size_col='size', color_col='category')

# Create a calendar heatmap
plot_calendar_heatmap(dates, values, year=2023)
```

### Interactive Visualizations

```python
from datavis import create_interactive_dashboard, plot_interactive_time_series

# Create an interactive dashboard
create_interactive_dashboard(df, title="Interactive Data Exploration")

# Create an interactive time series plot
plot_interactive_time_series(df, date_col='date', value_cols=['value1', 'value2'])
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This package builds on top of matplotlib, seaborn, and other visualization libraries.
- Special thanks to all contributors and the open-source community.