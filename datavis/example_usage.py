"""
Example usage of the datavis package.

This script demonstrates the various visualization functions
in the datavis package with a sample dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the datavis package
from datavis import (
    set_visualization_style,
    plot_numeric_distribution,
    plot_categorical_counts,
    plot_boxplots,
    plot_correlation_matrix,
    plot_time_series,
    plot_heatmap,
    visualize_dataset_summary
)

# Sample data generation
def generate_sample_data(n_samples=1000):
    """Generate a sample dataset for demonstration"""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Generate numeric features
    numeric_feature1 = np.random.normal(loc=50, scale=10, size=n_samples)
    numeric_feature2 = np.random.normal(loc=100, scale=20, size=n_samples)
    numeric_feature3 = numeric_feature1 * 0.5 + numeric_feature2 * 0.3 + np.random.normal(loc=0, scale=5, size=n_samples)
    
    # Generate categorical features
    categories = ['A', 'B', 'C', 'D', 'E']
    categorical_feature1 = np.random.choice(categories, size=n_samples)
    categorical_feature2 = np.random.choice(['High', 'Medium', 'Low'], size=n_samples)
    
    # Create time series
    time_series1 = np.sin(np.linspace(0, 10, n_samples)) * 10 + np.random.normal(0, 1, n_samples)
    time_series2 = np.cos(np.linspace(0, 10, n_samples)) * 15 + np.random.normal(0, 1, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'numeric1': numeric_feature1,
        'numeric2': numeric_feature2,
        'numeric3': numeric_feature3,
        'category1': categorical_feature1,
        'category2': categorical_feature2,
        'time_series1': time_series1,
        'time_series2': time_series2
    })
    
    return df

def main():
    """Main function to demonstrate datavis package functionality"""
    # Set the visualization style
    set_visualization_style(style="whitegrid", context="notebook", palette="viridis")
    
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data()
    print(f"Sample data shape: {df.shape}")
    
    # Display dataset summary
    print("\nCreating dataset summary visualization...")
    visualize_dataset_summary(df, title="Sample Dataset Overview")
    
    # Numeric distributions
    print("\nPlotting numeric distributions...")
    plot_numeric_distribution(df, columns=['numeric1', 'numeric2', 'numeric3'], 
                            title="Numeric Features Distribution")
    
    # Categorical counts
    print("\nPlotting categorical counts...")
    plot_categorical_counts(df, columns=['category1', 'category2'], 
                           title="Categorical Features Counts")
    
    # Boxplots
    print("\nPlotting boxplots...")
    plot_boxplots(df, numeric_cols=['numeric1', 'numeric2', 'numeric3'],
                  groupby_col='category1', title="Boxplots by Category")
    
    # Correlation matrix
    print("\nPlotting correlation matrix...")
    plot_correlation_matrix(df, columns=['numeric1', 'numeric2', 'numeric3'], 
                         title="Correlation Matrix")
    
    # Time series plot
    print("\nPlotting time series...")
    plot_time_series(df, date_col='date', value_cols=['time_series1', 'time_series2'],
                     title="Time Series Plot")
    
    # Heatmap
    print("\nPlotting heatmap...")
    # Create a pivot table for the heatmap
    pivot_data = pd.pivot_table(df, values='numeric1', index='category1', 
                            columns='category2', aggfunc='mean')
    plot_heatmap(pivot_data, title="Average numeric1 by categories")
    
    print("\nAll visualizations completed. Close the plot windows to exit.")
    plt.show()

if __name__ == "__main__":
    main()