"""
Prediction Script

This script loads a trained model and uses it to make predictions on new data.
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Import custom modules
from data_loader import load_excel_data, clean_data

def load_model(model_path):
    """Load a trained model from disk."""
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")
    return model

def make_predictions(model, data):
    """Make predictions using the trained model."""
    try:
        predictions = model.predict(data)
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

def save_predictions(predictions, input_file, output_dir):
    """Save predictions to a CSV file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename based on input filename
    input_filename = Path(input_file).stem
    output_file = os.path.join(output_dir, f"{input_filename}_predictions.csv")
    
    # Create DataFrame with predictions
    df_predictions = pd.DataFrame({'predictions': predictions})
    
    # Save to CSV
    df_predictions.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def main():
    """Main function to run the prediction process."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Make predictions using a trained model.')
    parser.add_argument('--model', required=True, help='Path to the trained model file')
    parser.add_argument('--input', required=True, help='Path to the input data file')
    parser.add_argument('--output-dir', default='predictions', help='Directory to save predictions')
    parser.add_argument('--sheet-name', help='Sheet name if input is an Excel file')
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model)
    
    # Load and clean the data
    print(f"Loading data from {args.input}...")
    if args.sheet_name:
        df = load_excel_data(args.input, args.sheet_name)
    else:
        df = load_excel_data(args.input)
    
    clean_df = clean_data(df)
    print(f"Data loaded and cleaned. Shape: {clean_df.shape}")
    
    # Make predictions
    print("Making predictions...")
    predictions = make_predictions(model, clean_df)
    
    if predictions is not None:
        # Save predictions
        save_predictions(predictions, args.input, args.output_dir)

if __name__ == "__main__":
    main()