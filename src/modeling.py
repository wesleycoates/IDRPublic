import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import pickle
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,  # Regression metrics
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,  # Classification metrics
    roc_auc_score, roc_curve, precision_recall_curve
)

# Import various models
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def get_numeric_and_categorical_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to analyze
        
    Returns:
    --------
    tuple
        Lists of numeric and categorical column names
    """
    # Identify numeric columns (excluding any obvious target/ID columns)
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Identify categorical columns - including object, category and boolean types
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    return numeric_columns, categorical_columns


def create_preprocessing_pipeline(numeric_columns: List[str], categorical_columns: List[str]) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for numeric and categorical features.
    
    Parameters:
    -----------
    numeric_columns : list of str
        Names of numeric columns
    categorical_columns : list of str
        Names of categorical columns
        
    Returns:
    --------
    ColumnTransformer
        Preprocessing pipeline
    """
    # Define numeric preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define categorical preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ]
    )
    
    return preprocessor


def get_regression_models() -> Dict[str, Any]:
    """
    Get a dictionary of regression models.
    
    Returns:
    --------
    dict
        Dictionary with model names as keys and model instances as values
    """
    return {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor()
    }


def get_classification_models() -> Dict[str, Any]:
    """
    Get a dictionary of classification models.
    
    Returns:
    --------
    dict
        Dictionary with model names as keys and model instances as values
    """
    return {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier()
    }


def evaluate_regression_models(X_train: pd.DataFrame, y_train: pd.Series, 
                              X_test: pd.DataFrame, y_test: pd.Series,
                              preprocessor: ColumnTransformer) -> pd.DataFrame:
    """
    Evaluate multiple regression models and return their performance metrics.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Testing features
    y_test : pd.Series
        Testing target
    preprocessor : ColumnTransformer
        Preprocessing pipeline
        
    Returns:
    --------
    pd.DataFrame
        Performance metrics for each model
    """
    # Get regression models
    models = get_regression_models()
    
    # Results dictionary
    results = []
    
    for name, model in models.items():
        # Create pipeline with preprocessor and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate cross-validation score
        cv_score = np.mean(cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2'))
        
        # Store results
        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'CV R²': cv_score
        })
    
    # Convert to DataFrame and sort by R²
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R²', ascending=False).reset_index(drop=True)
    
    return results_df


def evaluate_classification_models(X_train: pd.DataFrame, y_train: pd.Series, 
                                  X_test: pd.DataFrame, y_test: pd.Series,
                                  preprocessor: ColumnTransformer) -> pd.DataFrame:
    """
    Evaluate multiple classification models and return their performance metrics.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Testing features
    y_test : pd.Series
        Testing target
    preprocessor : ColumnTransformer
        Preprocessing pipeline
        
    Returns:
    --------
    pd.DataFrame
        Performance metrics for each model
    """
    # Get classification models
    models = get_classification_models()
    
    # Results dictionary
    results = []
    
    for name, model in models.items():
        # Create pipeline with preprocessor and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # For ROC-AUC, we need probability predictions
        try:
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = np.nan
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # For multi-class problems, we need to specify an average method
        if len(np.unique(y_train)) > 2:
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
        else:
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
        
        # Calculate cross-validation score
        cv_score = np.mean(cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy'))
        
        # Store results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC-AUC': roc_auc,
            'CV Accuracy': cv_score
        })
    
    # Convert to DataFrame and sort by F1 Score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)
    
    return results_df


def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series,
                        model, param_grid: Dict[str, Any],
                        preprocessor: ColumnTransformer,
                        cv: int = 5, scoring: str = None):
    """
    Tune hyperparameters for a given model using GridSearchCV.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    model : estimator
        The model to tune
    param_grid : dict
        Parameter grid for hyperparameter tuning
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, optional
        Scoring metric to use
        
    Returns:
    --------
    tuple
        Best estimator and a DataFrame with detailed CV results
    """
    # Create pipeline with preprocessor and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Create parameter grid for the pipeline
    pipeline_param_grid = {f'model__{param}': values for param, values in param_grid.items()}
    
    # Create grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid=pipeline_param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Convert CV results to DataFrame
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # Get only relevant columns
    cv_results = cv_results[[col for col in cv_results.columns if 'param_' in col or 'mean_test_score' in col or 'std_test_score' in col]]
    
    # Rename parameter columns
    cv_results.columns = [col.replace('param_model__', '') if 'param_model__' in col else col for col in cv_results.columns]
    
    # Sort by mean test score
    cv_results = cv_results.sort_values('mean_test_score', ascending=False).reset_index(drop=True)
    
    return grid_search.best_estimator_, cv_results


def save_model(model, filepath: str):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : estimator
        Trained model to save
    filepath : str
        Path where the model will be saved
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path where the model is saved
        
    Returns:
    --------
    The loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    # This will execute when you run this script directly
    print("This is a module for building and evaluating predictive models.")
    print("Example usage:")
    print("from modeling import evaluate_regression_models, evaluate_classification_models")
    print("results = evaluate_regression_models(X_train, y_train, X_test, y_test, preprocessor)")