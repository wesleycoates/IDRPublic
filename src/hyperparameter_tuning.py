"""
Hyperparameter Tuning Script

This script performs hyperparameter tuning for various machine learning models.
"""

import pandas as pd
import numpy as np
import argparse
import time
import os
from pathlib import Path

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, accuracy_score, f1_score

# Import custom modules
from data_loader import load_excel_data, clean_data, split_data
from modeling import get_numeric_and_categorical_columns, create_preprocessing_pipeline
from modeling import get_regression_models, get_classification_models, save_model

# Define parameter grids for different models
def get_param_grids():
    """
    Get parameter grids for different models.
    
    Returns:
    --------
    dict
        Dictionary with model names as keys and parameter grids as values
    """
    return {
        # Regression models
        'Linear Regression': {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        },
        'Ridge Regression': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        },
        'Lasso Regression': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'fit_intercept': [True, False],
            'selection': ['cyclic', 'random']
        },
        'Decision Tree': {
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'SVR': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        
        # Classification models
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [100, 200, 300]
        },
        'SVC': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'probability': [True]
        }
    }

def tune_model(X_train, y_train, model_name, model, preprocessor, param_grid, problem_type, 
              cv=5, search_type='grid', n_iter=10, scoring=None, n_jobs=-1, verbose=1):
    """
    Perform hyperparameter tuning for a model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    model_name : str
        Name of the model
    model : estimator
        The model to tune
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    param_grid : dict
        Parameter grid for hyperparameter tuning
    problem_type : str
        Type of problem ('regression' or 'classification')
    cv : int, default=5
        Number of cross-validation folds
    search_type : str, default='grid'
        Type of search ('grid' or 'random')
    n_iter : int, default=10
        Number of iterations for random search
    scoring : str or callable, default=None
        Scoring metric to use
    n_jobs : int, default=-1
        Number of parallel jobs
    verbose : int, default=1
        Verbosity level
        
    Returns:
    --------
    tuple
        Best estimator, best parameters, and CV results
    """
    # Create pipeline with preprocessor and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Create parameter grid for the pipeline
    pipeline_param_grid = {f'model__{param}': values for param, values in param_grid.items()}
    
    # Choose scoring metric based on problem type
    if scoring is None:
        if problem_type == 'regression':
            scoring = 'neg_mean_squared_error'
        else:
            scoring = 'f1_weighted'
    
    # Choose search strategy
    if search_type == 'random':
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=pipeline_param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=42
        )
    else:
        search = GridSearchCV(
            pipeline,
            param_grid=pipeline_param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )
    
    # Perform search
    print(f"\nTuning {model_name}...")
    start_time = time.time()
    search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"Best parameters: {search.best_params_}")
    print(f"Best score: {search.best_score_:.4f}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    # Get and format CV results
    cv_results = pd.DataFrame(search.cv_results_)
    
    # Get only relevant columns and rename them for readability
    cv_columns = [col for col in cv_results.columns if any(x in col for x in ['param_', 'mean_test_score', 'std_test_score', 'rank_test_score'])]
    cv_results = cv_results[cv_columns]
    
    return search.best_estimator_, search.best_params_, cv_results

def tune_models(X_train, y_train, problem_type, models_to_tune, output_dir=None, cv=5, 
               search_type='grid', n_iter=10, scoring=None, n_jobs=-1, verbose=1):
    """
    Perform hyperparameter tuning for multiple models.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    problem_type : str
        Type of problem ('regression' or 'classification')
    models_to_tune : list of str
        Names of models to tune
    output_dir : str, optional
        Directory to save tuned models
    cv : int, default=5
        Number of cross-validation folds
    search_type : str, default='grid'
        Type of search ('grid' or 'random')
    n_iter : int, default=10
        Number of iterations for random search
    scoring : str or callable, default=None
        Scoring metric to use
    n_jobs : int, default=-1
        Number of parallel jobs
    verbose : int, default=1
        Verbosity level
        
    Returns:
    --------
    dict
        Dictionary with model names as keys and tuning results as values
    """
    # Get numeric and categorical columns
    numeric_cols, categorical_cols = get_numeric_and_categorical_columns(X_train)
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numeric_cols, categorical_cols)
    
    # Get models and parameter grids
    if problem_type == 'regression':
        all_models = get_regression_models()
    else:
        all_models = get_classification_models()
    
    param_grids = get_param_grids()
    
    # Filter models to tune
    if not models_to_tune:
        models_to_tune = list(all_models.keys())
    else:
        models_to_tune = [model for model in models_to_tune if model in all_models]
    
    if not models_to_tune:
        print("No valid models to tune.")
        return {}
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Tune each model
    results = {}
    
    for model_name in models_to_tune:
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}. Skipping.")
            continue
        
        model = all_models[model_name]
        param_grid = param_grids[model_name]
        
        # Tune model
        best_model, best_params, cv_results = tune_model(
            X_train, y_train, model_name, model, preprocessor, param_grid,
            problem_type, cv, search_type, n_iter, scoring, n_jobs, verbose
        )
        
        # Save results
        results[model_name] = {
            'best_model': best_model,
            'best_params': best_params,
            'cv_results': cv_results
        }
        
        # Save model if output directory specified
        if output_dir:
            model_filename = f"{model_name.lower().replace(' ', '_')}_tuned.pkl"
            model_path = os.path.join(output_dir, model_filename)
            save_model(best_model, model_path)
    
    return results

def main():
    """Main function to run hyperparameter tuning."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Perform hyperparameter tuning for machine learning models.')
    parser.add_argument('--data', required=True, help='Path to the data file')
    parser.add_argument('--target', required=True, help='Name of the target column')
    parser.add_argument('--problem-type', choices=['regression', 'classification'], required=True,
                        help='Type of problem (regression or classification)')
    parser.add_argument('--models', nargs='+', help='Models to tune (default: all)')
    parser.add_argument('--search-type', choices=['grid', 'random'], default='grid',
                        help='Type of search (grid or random)')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--n-iter', type=int, default=10, help='Number of iterations for random search')
    parser.add_argument('--output-dir', default='models', help='Directory to save tuned models')
    parser.add_argument('--sheet-name', help='Sheet name if input is an Excel file')
    args = parser.parse_args()
    
    # Load and prepare data
    print(f"Loading data from {args.data}...")
    if args.sheet_name:
        df = load_excel_data(args.data, args.sheet_name)
    else:
        df = load_excel_data(args.data)
    
    clean_df = clean_data(df)
    print(f"Data loaded and cleaned. Shape: {clean_df.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(clean_df, args.target)
    print(f"Data split. Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    
    # Tune models
    results = tune_models(
        X_train, y_train,
        problem_type=args.problem_type,
        models_to_tune=args.models,
        output_dir=args.output_dir,
        cv=args.cv,
        search_type=args.search_type,
        n_iter=args.n_iter
    )
    
    # Print summary
    print("\nTuning summary:")
    for model_name, result in results.items():
        print(f"{model_name}: Best score = {result['best_model'].best_score_:.4f}")

if __name__ == "__main__":
    main()