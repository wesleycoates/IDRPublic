"""
Model Evaluation Module

This module contains functions for evaluating trained models on test data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import os

from sklearn.metrics import (
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve, roc_curve,
    # For both
    make_scorer
)

def load_model(model_path):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    
    Returns:
    --------
    object
        The loaded model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {model_path}")
    return model

def evaluate_regression_model(model, X_test, y_test, plot=True, figsize=(15, 10)):
    """
    Evaluate a regression model on test data.
    
    Parameters:
    -----------
    model : object
        Trained model or pipeline
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target values
    plot : bool, default=True
        Whether to create evaluation plots
    figsize : tuple, default=(15, 10)
        Figure size for plots
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)
    
    # Print metrics
    print("Regression Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Explained Variance: {explained_var:.4f}")
    
    # Create plots if requested
    if plot:
        plt.figure(figsize=figsize)
        
        # Scatter plot of actual vs predicted values
        plt.subplot(2, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        
        # Residual plot
        plt.subplot(2, 2, 2)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Histogram of residuals
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Residuals')
        
        # QQ plot of residuals
        plt.subplot(2, 2, 4)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Normal Q-Q Plot of Residuals')
        
        plt.tight_layout()
        plt.show()
    
    # Return metrics
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'explained_variance': explained_var
    }

def evaluate_classification_model(model, X_test, y_test, plot=True, figsize=(15, 12)):
    """
    Evaluate a classification model on test data.
    
    Parameters:
    -----------
    model : object
        Trained model or pipeline
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target values
    plot : bool, default=True
        Whether to create evaluation plots
    figsize : tuple, default=(15, 12)
        Figure size for plots
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Check if this is a binary or multiclass problem
    classes = np.unique(y_test)
    n_classes = len(classes)
    is_binary = n_classes == 2
    
    # For some metrics, we need probability predictions
    try:
        y_pred_proba = model.predict_proba(X_test)
        has_probas = True
    except:
        has_probas = False
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    if is_binary:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        if has_probas:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            roc_auc = None
    else:
        # For multiclass, use weighted average
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        if has_probas:
            # For multiclass, use one-vs-rest ROC AUC
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        else:
            roc_auc = None
    
    # Print metrics
    print("Classification Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"  ROC AUC: {roc_auc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create plots if requested
    if plot:
        plt.figure(figsize=figsize)
        
        # Confusion matrix
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if has_probas and is_binary:
            # ROC curve for binary classification
            plt.subplot(2, 2, 2)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            
            # Precision-Recall curve for binary classification
            plt.subplot(2, 2, 3)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
            plt.plot(recall_curve, precision_curve)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            
            # Histogram of probability predictions
            plt.subplot(2, 2, 4)
            plt.hist(y_pred_proba[:, 1], bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Frequency')
            plt.title('Histogram of Predicted Probabilities')
        
        elif has_probas and not is_binary:
            # For multiclass, show distribution of top class probabilities
            plt.subplot(2, 2, 2)
            top_probs = np.max(y_pred_proba, axis=1)
            plt.hist(top_probs, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Probability of Predicted Class')
            plt.ylabel('Frequency')
            plt.title('Confidence in Predictions')
            
            # For multiclass, show distribution of probability differences
            plt.subplot(2, 2, 3)
            sorted_probs = np.sort(y_pred_proba, axis=1)
            prob_diff = sorted_probs[:, -1] - sorted_probs[:, -2]  # Diff between top and second
            plt.hist(prob_diff, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Probability Difference (Top - Second)')
            plt.ylabel('Frequency')
            plt.title('Prediction Confidence Margin')
        
        plt.tight_layout()
        plt.show()
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    if roc_auc is not None:
        metrics['roc_auc'] = roc_auc
    
    return metrics

def evaluate_model(model, X_test, y_test, problem_type=None, plot=True, figsize=(15, 10)):
    """
    Evaluate a model on test data, automatically detecting the problem type if not specified.
    
    Parameters:
    -----------
    model : object
        Trained model or pipeline
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target values
    problem_type : str, optional
        Problem type ('regression' or 'classification'). If None, it will be auto-detected.
    plot : bool, default=True
        Whether to create evaluation plots
    figsize : tuple, default=(15, 10)
        Figure size for plots
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Auto-detect problem type if not specified
    if problem_type is None:
        # Check if the target is numeric
        if pd.api.types.is_numeric_dtype(y_test) and len(np.unique(y_test)) > 10:
            problem_type = 'regression'
        else:
            problem_type = 'classification'
        
        print(f"Auto-detected problem type: {problem_type}")
    
    # Evaluate based on problem type
    if problem_type == 'regression':
        return evaluate_regression_model(model, X_test, y_test, plot, figsize)
    else:
        return evaluate_classification_model(model, X_test, y_test, plot, figsize)

def compare_models(models, X_test, y_test, problem_type=None, names=None):
    """
    Compare multiple models on the same test data.
    
    Parameters:
    -----------
    models : list
        List of trained models or pipelines
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target values
    problem_type : str, optional
        Problem type ('regression' or 'classification'). If None, it will be auto-detected.
    names : list, optional
        Names for the models. If None, generic names will be used.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame comparing model performance
    """
    # Auto-detect problem type if not specified
    if problem_type is None:
        # Check if the target is numeric
        if pd.api.types.is_numeric_dtype(y_test) and len(np.unique(y_test)) > 10:
            problem_type = 'regression'
        else:
            problem_type = 'classification'
        
        print(f"Auto-detected problem type: {problem_type}")
    
    # Use generic names if not provided
    if names is None:
        names = [f"Model {i+1}" for i in range(len(models))]
    
    # Evaluate each model
    results = []
    
    for i, (name, model) in enumerate(zip(names, models)):
        print(f"\nEvaluating {name}...")
        
        # Evaluate without plots
        metrics = evaluate_model(model, X_test, y_test, problem_type, plot=False)
        
        # Add model name
        metrics['Model'] = name
        
        # Add to results
        results.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns to put Model first
    cols = ['Model'] + [col for col in results_df.columns if col != 'Model']
    results_df = results_df[cols]
    
    # Sort by appropriate metric
    if problem_type == 'regression':
        # For regression, sort by R² (higher is better)
        results_df = results_df.sort_values('r2', ascending=False)
    else:
        # For classification, sort by F1 (higher is better)
        results_df = results_df.sort_values('f1', ascending=False)
    
    # Create comparison plot
    if problem_type == 'regression':
        metrics_to_plot = ['r2', 'explained_variance']
        plt.figure(figsize=(10, 6))
        
        # Plot R² and Explained Variance (higher is better)
        plt.subplot(1, 2, 1)
        sns.barplot(x='Model', y='r2', data=results_df)
        plt.title('R² Score Comparison')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        sns.barplot(x='Model', y='explained_variance', data=results_df)
        plt.title('Explained Variance Comparison')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Plot error metrics (lower is better)
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.barplot(x='Model', y='mse', data=results_df)
        plt.title('MSE Comparison')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 3, 2)
        sns.barplot(x='Model', y='rmse', data=results_df)
        plt.title('RMSE Comparison')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 3, 3)
        sns.barplot(x='Model', y='mae', data=results_df)
        plt.title('MAE Comparison')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    else:
        # For classification, plot accuracy, precision, recall, f1
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        sns.barplot(x='Model', y='accuracy', data=results_df)
        plt.title('Accuracy Comparison')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        plt.subplot(1, 4, 2)
        sns.barplot(x='Model', y='precision', data=results_df)
        plt.title('Precision Comparison')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        plt.subplot(1, 4, 3)
        sns.barplot(x='Model', y='recall', data=results_df)
        plt.title('Recall Comparison')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        plt.subplot(1, 4, 4)
        sns.barplot(x='Model', y='f1', data=results_df)
        plt.title('F1 Score Comparison')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # If ROC AUC is available, plot it separately
        if 'roc_auc' in results_df.columns:
            plt.figure(figsize=(8, 5))
            sns.barplot(x='Model', y='roc_auc', data=results_df)
            plt.title('ROC AUC Comparison')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    return results_df

if __name__ == "__main__":
    print("This is a module for evaluating machine learning models.")
    print("Example usage:")
    print("from model_evaluation import evaluate_model, compare_models")
    print("metrics = evaluate_model(model, X_test, y_test)")