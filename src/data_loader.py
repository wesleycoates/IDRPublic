import os
import pandas as pd
from typing import Optional, Tuple


def load_excel_data(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from an Excel file into a pandas DataFrame.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file
    sheet_name : str, optional
        Name of the sheet to load. If None, the first sheet is loaded.
        
    Returns:
    --------
    pd.DataFrame
        The loaded data
    """
    try:
        # Try to read the Excel file
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
        
        print(f"Successfully loaded data from {file_path}")
        print(f"Data shape: {df.shape}")
        
        return df
    
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning operations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to clean
        
    Returns:
    --------
    pd.DataFrame
        The cleaned data
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Drop rows with all NaN values
    cleaned_df.dropna(how='all', inplace=True)
    
    # Drop duplicated rows
    cleaned_df.drop_duplicates(inplace=True)
    
    # Convert column names to lowercase and replace spaces with underscores
    cleaned_df.columns = [col.lower().replace(' ', '_') for col in cleaned_df.columns]
    
    # Print cleaning summary
    print(f"Cleaning summary:")
    print(f"  Original shape: {df.shape}")
    print(f"  Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df

# Filter the data to the target dataset in scope
# Filter the main dataframe to include only 'Single' cases
# First, let's check the unique values in the "Type of Dispute" column
print("Unique values in 'Type of Dispute' column:")
print(df["Type of Dispute"].unique())

# Now filter for only Single cases
df_single = df[df["Type of Dispute"] == "Single"]
print(f"\nFiltered for Single cases only")
print(f"Original dataframe shape: {df.shape}")
print(f"Filtered dataframe shape: {df_single.shape}")
print(f"Percentage of Single cases: {df_single.shape[0]/df.shape[0]*100:.2f}%")

# Display the first few rows of the filtered data
print("\nFirst 5 rows of filtered data:")
df_single.head()


def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The data to split
    target_column : str
        The name of the target column
    test_size : float, default=0.2
        The proportion of the data to include in the test split
    random_state : int, default=42
        Controls the shuffling applied to the data before splitting
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Data split summary:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # This will execute when you run this script directly
    # Example usage:
    print("This is a module for loading and preprocessing data.")
    print("Here's an example of how to use it:")
    print("from data_loader import load_excel_data, clean_data, split_data")
    print("df = load_excel_data('path/to/your/excel_file.xlsx')")
    print("clean_df = clean_data(df)")
    print("X_train, X_test, y_train, y_test = split_data(clean_df, 'target_column')")