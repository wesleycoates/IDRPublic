#!/usr/bin/env python3

"""
Data Science CLI Tool

A command-line interface for running common data science tasks.
"""

import argparse
import os
import sys
import pandas as pd
import subprocess
from pathlib import Path
import time

# Import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add colors to terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(message):
    """Print a formatted header message."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {message} ==={Colors.END}\n")

def print_success(message):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")

def print_error(message):
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.END}")

def print_info(message):
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.END}")

def run_command(command):
    """Run a shell command and return its output."""
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()
    
    return process.returncode, stdout, stderr

def list_data_files():
    """List all data files in the data directory."""
    print_header("Available Data Files")
    
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print_error(f"Data directory not found: {data_dir}")
        return
    
    # Find all data files
    excel_files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls"))
    csv_files = list(data_dir.glob("*.csv"))
    json_files = list(data_dir.glob("*.json"))
    
    if not excel_files and not csv_files and not json_files:
        print_warning("No data files found in data/raw directory.")
        return
    
    # Print Excel files
    if excel_files:
        print(f"{Colors.BOLD}Excel Files:{Colors.END}")
        for i, file in enumerate(excel_files, 1):
            file_size = file.stat().st_size / 1024  # Size in KB
            print(f"  {i}. {file.name} ({file_size:.1f} KB)")
        print()
    
    # Print CSV files
    if csv_files:
        print(f"{Colors.BOLD}CSV Files:{Colors.END}")
        for i, file in enumerate(csv_files, 1):
            file_size = file.stat().st_size / 1024  # Size in KB
            print(f"  {i}. {file.name} ({file_size:.1f} KB)")
        print()
    
    # Print JSON files
    if json_files:
        print(f"{Colors.BOLD}JSON Files:{Colors.END}")
        for i, file in enumerate(json_files, 1):
            file_size = file.stat().st_size / 1024  # Size in KB
            print(f"  {i}. {file.name} ({file_size:.1f} KB)")
        print()

def list_models():
    """List all saved models."""
    print_header("Available Models")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print_error(f"Models directory not found: {models_dir}")
        return
    
    # Find all model files
    model_files = list(models_dir.glob("*.pkl"))
    
    if not model_files:
        print_warning("No model files found in models directory.")
        return
    
    print(f"{Colors.BOLD}Saved Models:{Colors.END}")
    for i, file in enumerate(model_files, 1):
        file_size = file.stat().st_size / 1024  # Size in KB
        modified_time = time.ctime(file.stat().st_mtime)
        print(f"  {i}. {file.name} ({file_size:.1f} KB, last modified: {modified_time})")
    print()

def explore_data(file_path):
    """Perform basic data exploration for a given file."""
    print_header(f"Exploring Data: {file_path}")
    
    # Check if file exists
    file = Path(file_path)
    if not file.exists():
        print_error(f"File not found: {file_path}")
        return
    
    # Determine file type and load accordingly
    file_extension = file.suffix.lower()
    
    try:
        if file_extension in ['.xlsx', '.xls']:
            print_info("Loading Excel file...")
            df = pd.read_excel(file_path)
        elif file_extension == '.csv':
            print_info("Loading CSV file...")
            df = pd.read_csv(file_path)
        elif file_extension == '.json':
            print_info("Loading JSON file...")
            df = pd.read_json(file_path)
        else:
            print_error(f"Unsupported file format: {file_extension}")
            return
        
        # Basic information
        print(f"\n{Colors.BOLD}Basic Information:{Colors.END}")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")
        
        # Column information
        print(f"\n{Colors.BOLD}Column Information:{Colors.END}")
        for col in df.columns:
            non_null = df[col].count()
            total = len(df)
            null_percentage = (1 - non_null / total) * 100 if total > 0 else 0
            
            print(f"  {col}")
            print(f"    - Type: {df[col].dtype}")
            print(f"    - Missing: {total - non_null} ({null_percentage:.1f}%)")
            
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f"    - Min: {df[col].min()}")
                print(f"    - Max: {df[col].max()}")
                print(f"    - Mean: {df[col].mean()}")
            elif pd.api.types.is_string_dtype(df[col]):
                unique_values = df[col].nunique()
                print(f"    - Unique values: {unique_values}")
                
                # Show some sample values if there aren't too many
                if unique_values <= 10:
                    print(f"    - Values: {', '.join(str(x) for x in df[col].unique()[:10])}")
            
            print()
        
        # Show first few rows
        print(f"\n{Colors.BOLD}First 5 Rows:{Colors.END}")
        print(df.head().to_string())
        
        print_success("Data exploration complete!")
    
    except Exception as e:
        print_error(f"Error exploring data: {e}")

def run_notebook(notebook_path):
    """Run a Jupyter notebook and display its output."""
    print_header(f"Running Notebook: {notebook_path}")
    
    # Check if file exists
    notebook = Path(notebook_path)
    if not notebook.exists():
        print_error(f"Notebook not found: {notebook_path}")
        return
    
    # Check if nbconvert is available
    print_info("Checking if nbconvert is available...")
    return_code, _, _ = run_command("jupyter nbconvert --version")
    
    if return_code != 0:
        print_error("jupyter nbconvert not found. Make sure Jupyter is installed.")
        print_info("Try running: pip install jupyter")
        return
    
    # Run the notebook
    print_info("Running notebook...")
    return_code, stdout, stderr = run_command(
        f"jupyter nbconvert --to notebook --execute --inplace {notebook_path}"
    )
    
    if return_code != 0:
        print_error(f"Error running notebook: {stderr}")
        return
    
    print_success(f"Notebook executed successfully: {notebook_path}")

def train_model(data_file, target_column, problem_type, model_name=None):
    """Train a model on the given data file."""
    print_header(f"Training Model on {data_file}")
    
    # Check if file exists
    data_path = Path(data_file)
    if not data_path.exists():
        print_error(f"Data file not found: {data_file}")
        return
    
    # Import required modules
    try:
        from src.data_loader import load_excel_data, clean_data, split_data
        from src.modeling import (
            get_numeric_and_categorical_columns, create_preprocessing_pipeline,
            evaluate_regression_models, evaluate_classification_models,
            save_model
        )
    except ImportError as e:
        print_error(f"Error importing required modules: {e}")
        print_info("Make sure you're running this script from the project root directory.")
        return
    
    try:
        # Determine file type and load accordingly
        file_extension = data_path.suffix.lower()
        
        if file_extension in ['.xlsx', '.xls']:
            print_info("Loading Excel file...")
            df = load_excel_data(data_file)
        elif file_extension == '.csv':
            print_info("Loading CSV file...")
            df = pd.read_csv(data_file)
        elif file_extension == '.json':
            print_info("Loading JSON file...")
            df = pd.read_json(data_file)
        else:
            print_error(f"Unsupported file format: {file_extension}")
            return
        
        # Clean data
        print_info("Cleaning data...")
        clean_df = clean_data(df)
        
        # Check if target column exists
        if target_column not in clean_df.columns:
            print_error(f"Target column not found: {target_column}")
            print_info(f"Available columns: {', '.join(clean_df.columns)}")
            return
        
        # Split data
        print_info(f"Splitting data with target column: {target_column}...")
        X_train, X_test, y_train, y_test = split_data(clean_df, target_column)
        
        # Create preprocessing pipeline
        numeric_cols, categorical_cols = get_numeric_and_categorical_columns(X_train)
        preprocessor = create_preprocessing_pipeline(numeric_cols, categorical_cols)
        
        # Train models
        if problem_type == 'regression':
            print_info("Training regression models...")
            results = evaluate_regression_models(X_train, y_train, X_test, y_test, preprocessor)
        elif problem_type == 'classification':
            print_info("Training classification models...")
            results = evaluate_classification_models(X_train, y_train, X_test, y_test, preprocessor)
        else:
            print_error(f"Unsupported problem type: {problem_type}")
            print_info("Use 'regression' or 'classification'")
            return
        
        # Display results
        print(f"\n{Colors.BOLD}Model Performance:{Colors.END}")
        print(results.to_string())
        
        # Determine best model
        best_model_name = results.iloc[0]['Model']
        print_success(f"Best model: {best_model_name}")
        
        # Save model if model_name provided
        if model_name:
            # Create pipeline for the best model
            from sklearn.pipeline import Pipeline
            
            if problem_type == 'regression':
                from src.modeling import get_regression_models
                models = get_regression_models()
            else:
                from src.modeling import get_classification_models
                models = get_classification_models()
            
            # Get the best model
            best_model = models[best_model_name]
            
            # Create pipeline
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
            
            # Fit
            pipeline.fit(X_train, y_train)
            
            # Save
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True, parents=True)
            
            model_path = model_dir / f"{model_name}.pkl"
            save_model(pipeline, str(model_path))
            
            print_success(f"Model saved to: {model_path}")
        
    except Exception as e:
        print_error(f"Error training model: {e}")

def make_predictions(model_file, data_file, output_file=None):
    """Make predictions using a trained model."""
    print_header(f"Making Predictions with {model_file}")
    
    # Check if files exist
    model_path = Path(model_file)
    data_path = Path(data_file)
    
    if not model_path.exists():
        print_error(f"Model file not found: {model_file}")
        return
    
    if not data_path.exists():
        print_error(f"Data file not found: {data_file}")
        return
    
    # Generate output filename if not provided
    if output_file is None:
        predictions_dir = Path("predictions")
        predictions_dir.mkdir(exist_ok=True, parents=True)
        
        output_file = predictions_dir / f"{data_path.stem}_predictions.csv"
    
    # Run prediction script
    try:
        from src.predict import load_model, make_predictions, save_predictions
        from src.data_loader import load_excel_data, clean_data
        
        # Load model
        print_info(f"Loading model from {model_file}...")
        model = load_model(model_file)
        
        # Load data
        file_extension = data_path.suffix.lower()
        
        if file_extension in ['.xlsx', '.xls']:
            print_info("Loading Excel file...")
            df = load_excel_data(data_file)
        elif file_extension == '.csv':
            print_info("Loading CSV file...")
            df = pd.read_csv(data_file)
        else:
            print_error(f"Unsupported file format: {file_extension}")
            return
        
        # Clean data
        print_info("Cleaning data...")
        clean_df = clean_data(df)
        
        # Make predictions
        print_info("Making predictions...")
        predictions = make_predictions(model, clean_df)
        
        # Save predictions
        print_info(f"Saving predictions to {output_file}...")
        save_predictions(predictions, data_file, os.path.dirname(output_file))
        
        print_success(f"Predictions saved to: {output_file}")
        
    except Exception as e:
        print_error(f"Error making predictions: {e}")

def main():
    """Main function to parse arguments and run commands."""
    parser = argparse.ArgumentParser(
        description='Data Science CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available data files
  python src/data_science_cli.py list-data
  
  # Explore a data file
  python src/data_science_cli.py explore data/raw/my_data.xlsx
  
  # Train a model
  python src/data_science_cli.py train data/raw/my_data.xlsx target_column regression my_model
  
  # Make predictions
  python src/data_science_cli.py predict models/my_model.pkl data/raw/new_data.xlsx
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List data command
    list_data_parser = subparsers.add_parser('list-data', help='List available data files')
    
    # List models command
    list_models_parser = subparsers.add_parser('list-models', help='List available models')
    
    # Explore data command
    explore_parser = subparsers.add_parser('explore', help='Explore a data file')
    explore_parser.add_argument('file', help='Path to the data file')
    
    # Run notebook command
    notebook_parser = subparsers.add_parser('notebook', help='Run a Jupyter notebook')
    notebook_parser.add_argument('notebook', help='Path to the notebook')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('data', help='Path to the data file')
    train_parser.add_argument('target', help='Target column name')
    train_parser.add_argument('type', choices=['regression', 'classification'], 
                             help='Problem type (regression or classification)')
    train_parser.add_argument('name', nargs='?', help='Name to save the model (optional)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('model', help='Path to the model file')
    predict_parser.add_argument('data', help='Path to the data file')
    predict_parser.add_argument('output', nargs='?', help='Path to save predictions (optional)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run appropriate command
    if args.command == 'list-data':
        list_data_files()
    elif args.command == 'list-models':
        list_models()
    elif args.command == 'explore':
        explore_data(args.file)
    elif args.command == 'notebook':
        run_notebook(args.notebook)
    elif args.command == 'train':
        train_model(args.data, args.target, args.type, args.name)
    elif args.command == 'predict':
        make_predictions(args.model, args.data, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()