def plot_cluster_analysis(df: pd.DataFrame, 
                        features: List[str], 
                        cluster_labels: np.ndarray,
                        method: str = 'pca', 
                        n_components: int = 2,
                        figsize: Tuple[int, int] = (12, 10),
                        title: str = 'Cluster Analysis', 
                        save_path: Optional[str] = None):
    """
    Visualize clusters using dimensionality reduction techniques.
    
    Parameters
    ----------
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
    
    Returns
    -------
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
    save_figure(fig, save_path)
    
    plt.tight_layout()
    return fig


def plot_silhouette_analysis(X: np.ndarray, 
                           cluster_labels: np.ndarray,
                           metric: str = 'euclidean', 
                           figsize: Tuple[int, int] = (12, 8),
                           title: str = 'Silhouette Analysis', 
                           save_path: Optional[str] = None):
    """
    Create a silhouette plot for evaluating clustering quality.
    
    Parameters
    ----------
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
    
    Returns
    -------
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
    save_figure(fig, save_path)
    
    return fig


def plot_model_comparison(model_names: List[str], 
                        metric_values: List[float], 
                        metric_name: str = 'Accuracy', 
                        ci_values: Optional[List[Tuple[float, float]]] = None,
                        figsize: Tuple[int, int] = (10, 6), 
                        color: str = '#4472C4',
                        sort_values: bool = True, 
                        title: str = 'Model Comparison',
                        annotate_values: bool = True, 
                        save_path: Optional[str] = None):
    """
    Create a bar chart comparing performance metrics across multiple models.
    
    Parameters
    ----------
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
    
    Returns
    -------
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
    save_figure(fig, save_path)
    
    return fig


def plot_residuals(y_true: Union[List[float], np.ndarray], 
                 y_pred: Union[List[float], np.ndarray],
                 figsize: Tuple[int, int] = (12, 8), 
                 title: str = 'Residual Analysis',
                 save_path: Optional[str] = None):
    """
    Create a comprehensive residual analysis plot for regression models.
    
    Parameters
    ----------
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
    
    Returns
    -------
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
    save_figure(fig, save_path)
    
    return fig
"""
Machine learning visualization functions.

This module contains functions for visualizing machine learning models,
model evaluation, and model comparison.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from typing import Optional, List, Tuple, Dict, Union

from .core import save_figure


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (10, 8), 
                         normalize: bool = False,
                         title: str = 'Confusion Matrix', 
                         cmap: str = 'Blues',
                         fmt: str = 'd', 
                         save_path: Optional[str] = None):
    """
    Plot a confusion matrix for classification results.
    
    Parameters
    ----------
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
    
    Returns
    -------
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
    save_figure(fig, save_path)
    
    return fig


def plot_roc_curve(fpr: Union[List[float], np.ndarray], 
                  tpr: Union[List[float], np.ndarray], 
                  auc: float,
                  title: str = 'ROC Curve', 
                  figsize: Tuple[int, int] = (8, 8),
                  label: Optional[str] = None, 
                  plot_diagonal: bool = True,
                  save_path: Optional[str] = None):
    """
    Plot a Receiver Operating Characteristic (ROC) curve.
    
    Parameters
    ----------
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
    
    Returns
    -------
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
    save_figure(fig, save_path)
    
    return fig


def plot_learning_curve(train_sizes: np.ndarray, 
                       train_scores: np.ndarray, 
                       test_scores: np.ndarray,
                       title: str = 'Learning Curve', 
                       figsize: Tuple[int, int] = (10, 6),
                       ylim: Optional[Tuple[float, float]] = None, 
                       xlabel: str = 'Training examples',
                       ylabel: str = 'Score', 
                       fill_std: bool = True,
                       save_path: Optional[str] = None):
    """
    Plot a learning curve from cross-validation results.
    
    Parameters
    ----------
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
    
    Returns
    -------
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
    save_figure(fig, save_path)
    
    return fig


def plot_decision_boundaries(X: np.ndarray, 
                           y: np.ndarray, 
                           model, 
                           feature_names: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (12, 10), 
                           title: str = 'Decision Boundaries',
                           save_path: Optional[str] = None):
    """
    Visualize decision boundaries of a classifier in 2D space.
    If X has more than 2 dimensions, PCA is used to reduce it to 2D.
    
    Parameters
    ----------
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
    
    Returns
    -------
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
        try:
            Z = model.predict(mesh_points)
        except:
            # If model can't predict on 2D data (because it was trained on more dimensions)
            # display a warning
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
    save_figure(fig, save_path)
    
    return fig