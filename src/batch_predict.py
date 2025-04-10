"""
Batch Prediction Script

This script loads a trained model and processes multiple files for prediction.
"""

import os
import pandas as pd
import numpy as np
import argparse
import glob
from pathlib import Path
import time

# Import custom modules
from data_loader import load_excel_data, clean_data

def load_model(model_path):
    """Load a trained model from disk."""
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")
    return model

def process_file(model, file_path, output_dir, sheet_name=None):
    """Process a single file for prediction."""
    try:
        print(f"\nProcessing file: {file_path}")
        
        # Load and clean the data
        if sheet_name:
            df = load_excel_data(file_path, sheet_name)
        else:
            df = load_excel_data(file_path)
        
        clean_df = clean_data(df)
        print(f"Data loaded and cleaned. Shape: {clean_df.shape}")
        
        # Make predictions
        predictions = model.predict(clean_df)
        
        # Create output filename
        input_filename = Path(file_path).stem
        output_file = os.path.join(output_dir, f"{input_filename}_predictions.csv")
        
        # Combine original data with predictions
        df_result = clean_df.copy()
        df_result['prediction'] = predictions
        
        # Save to CSV
        df_result.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        return True
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return False

def main():
    """Main function to run the batch prediction process."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Make batch predictions using a trained model.')
    parser.add_argument('--model', required=True, help='Path to the trained model file')
    parser.add_argument('--input-dir', required=True, help='Directory containing input files')
    parser.add_argument('--pattern', default='*.xlsx', help='File pattern to match (e.g., *.xlsx, *.csv)')
    parser.add_argument('--output-dir', default='predictions', help='Directory to save predictions')
    parser.add_argument('--sheet-name', help='Sheet name if inputs are Excel files')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    model = load_model(args.model)
    
    # Get list of files matching the pattern
    input_pattern = os.path.join(args.input_dir, args.pattern)
    files = glob.glob(input_pattern)
    
    if not files:
        print(f"No files found matching the pattern {input_pattern}")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    start_time = time.time()
    successful = 0
    
    for file_path in files:
        if process_file(model, file_path, args.output_dir, args.sheet_name):
            successful += 1
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print summary
    print(f"\nBatch processing complete.")
    print(f"Successfully processed {successful} out of {len(files)} files.")
    print(f"Total time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()