#!/usr/bin/env python3

"""
Project Initialization Script

This script sets up the data science project structure and installs required dependencies.
Run this script when you first initialize your GitHub CodeSpace.
"""

import os
import subprocess
import sys
from pathlib import Path
import shutil

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def run_command(command, desc=None):
    """Run a shell command and print its output."""
    if desc:
        print(f"\n>> {desc}")
    
    print(f"$ {command}")
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print(f"Error: {result.stderr}")
    
    return result.returncode == 0

def create_directory_structure():
    """Create the project directory structure."""
    print_section("Creating Directory Structure")
    
    directories = [
        "data/raw",
        "notebooks",
        "src",
        "models",
        "predictions",
        "reports",
        "figures"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create empty __init__.py files for src directory
    Path("src/__init__.py").touch(exist_ok=True)
    print("Created src/__init__.py")

def copy_source_files():
    """Copy source files to appropriate locations."""
    print_section("Setting Up Source Files")
    
    # Check if this script is being run from project root
    if not Path("src").exists():
        print("Error: Please run this script from the project root directory.")
        return False
    
    # Check if required source files exist in the same directory as this script
    required_files = [
        "src/data_loader.py",
        "src/visualization.py", 
        "src/modeling.py",
        "src/feature_engineering.py",
        "src/predict.py",
        "src/batch_predict.py",
        "src/hyperparameter_tuning.py",
        "src/model_evaluation.py",
        "notebooks/exploration.ipynb"
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"Error: Required file not found: {file_path}")
            all_files_exist = False
    
    if not all_files_exist:
        print("\nSome required files are missing. Please make sure all project files are in place.")
        return False
    
    print("All source files are in place.")
    return True

def install_dependencies():
    """Install required Python packages."""
    print_section("Installing Dependencies")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("Error: requirements.txt not found.")
        return False
    
    # Install requirements
    return run_command("pip install -r requirements.txt", "Installing required packages")

def check_file_permissions():
    """Make scripts executable."""
    print_section("Setting File Permissions")
    
    scripts = [
        "src/predict.py",
        "src/batch_predict.py",
        "src/hyperparameter_tuning.py"
    ]
    
    for script in scripts:
        if Path(script).exists():
            run_command(f"chmod +x {script}", f"Making {script} executable")

def find_excel_files():
    """Find Excel files in the project directory."""
    print_section("Finding Excel Files")
    
    # Find all Excel files
    excel_files = list(Path(".").glob("**/*.xlsx")) + list(Path(".").glob("**/*.xls"))
    
    if not excel_files:
        print("No Excel files found in the project directory.")
        return
    
    print("Found Excel files:")
    for file in excel_files:
        file_size = file.stat().st_size / 1024  # Size in KB
        print(f"  - {file} ({file_size:.1f} KB)")
    
    # Suggest moving Excel files to data/raw if they're not already there
    files_to_move = [f for f in excel_files if not str(f).startswith("data/raw")]
    
    if files_to_move:
        print("\nWould you like to move these Excel files to the data/raw directory? (y/n)")
        choice = input().strip().lower()
        
        if choice == 'y':
            for file in files_to_move:
                target = Path("data/raw") / file.name
                shutil.copy2(file, target)
                print(f"Copied {file} to {target}")
            
            print("\nExcel files have been copied to data/raw directory.")
            print("Make sure to update any file paths in your notebooks to point to these files.")

def update_notebook_placeholders():
    """Update placeholders in the exploration notebook."""
    print_section("Updating Notebook Placeholders")
    
    notebook_path = Path("notebooks/exploration.ipynb")
    if not notebook_path.exists():
        print("Error: Exploration notebook not found.")
        return False
    
    # Find Excel files in data/raw
    excel_files = list(Path("data/raw").glob("*.xlsx")) + list(Path("data/raw").glob("*.xls"))
    
    if not excel_files:
        print("No Excel files found in data/raw directory. Skipping notebook update.")
        return True
    
    # Use the first Excel file
    first_excel = excel_files[0]
    relative_path = f"../data/raw/{first_excel.name}"
    
    print(f"Found Excel file: {first_excel}")
    print(f"Using relative path: {relative_path}")
    
    # Try to update the notebook
    try:
        # Read the notebook
        notebook_content = notebook_path.read_text()
        
        # Replace the placeholder path
        updated_content = notebook_content.replace(
            'file_path = "../data/raw/your_excel_file.xlsx"',
            f'file_path = "{relative_path}"'
        )
        
        # Write back
        notebook_path.write_text(updated_content)
        
        print("Updated Excel file path in exploration notebook.")
        return True
    except Exception as e:
        print(f"Error updating notebook: {e}")
        return False

def main():
    """Main function to run the initialization process."""
    print_section("Data Science Project Initialization")
    print("This script will set up your data science project in GitHub CodeSpace.")
    
    # Create directories
    create_directory_structure()
    
    # Check and copy source files
    if not copy_source_files():
        print("\nError: Failed to set up source files.")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\nError: Failed to install dependencies.")
        return
    
    # Set file permissions
    check_file_permissions()
    
    # Find and process Excel files
    find_excel_files()
    
    # Update notebook placeholders
    update_notebook_placeholders()
    
    print_section("Initialization Complete")
    print("""
Your data science project has been set up successfully!

Next steps:
1. Upload your data to the 'data/raw' directory if you haven't already
2. Open 'notebooks/exploration.ipynb' to start exploring your data
3. Follow the guided process to build and evaluate predictive models
4. Use the various scripts in the 'src' directory for more advanced tasks

Happy data science!
    """)

if __name__ == "__main__":
    main()