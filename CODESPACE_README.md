# Data Science Setup in GitHub CodeSpace

This guide will help you get started with your data science workflow in GitHub CodeSpace.

## 🚀 Quick Start

After opening your CodeSpace, run:

```bash
python init_project.py
```

This script will set up your environment and install all required dependencies.

## 📂 Project Structure

```
predictive-analytics/
│
├── data/                # Store your data files here
│   └── raw/             # Place your Excel file here
│
├── notebooks/           # Jupyter notebooks for exploration
│   └── exploration.ipynb
│
├── src/                 # Source code
│   ├── data_loader.py   # Functions to load and preprocess data
│   ├── visualization.py # Visualization utilities
│   ├── modeling.py      # Model building and evaluation
│   ├── feature_engineering.py # Feature creation functions
│   ├── predict.py       # Make predictions with a trained model
│   ├── batch_predict.py # Process multiple files for prediction
│   ├── hyperparameter_tuning.py # Tune model hyperparameters
│   ├── model_evaluation.py      # Evaluate model performance
│   └── data_science_cli.py      # Command-line interface
│
├── models/              # Saved models will be stored here
│
├── requirements.txt     # Project dependencies
│
└── README.md            # Project documentation
```

## 🛠️ Using the CLI Tool

For convenience, you can use the command-line interface to perform common tasks:

### List all data files:
```bash
python src/data_science_cli.py list-data
```

### Explore a data file:
```bash
python src/data_science_cli.py explore data/raw/your_file.xlsx
```

### Train a model:
```bash
python src/data_science_cli.py train data/raw/your_file.xlsx target_column regression model_name
```

### Make predictions:
```bash
python src/data_science_cli.py predict models/model_name.pkl data/raw/new_data.xlsx
```

### List all available commands:
```bash
python src/data_science_cli.py --help
```

## 📊 Using Jupyter Notebooks

To launch Jupyter Notebook in your CodeSpace:

1. Click on the "Ports" tab in the lower part of the CodeSpace window
2. Click "Forward a Port"
3. Enter port 8888
4. Run this command in the terminal:
   ```bash
   jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
   ```
5. Click the "Open in Browser" button that appears in the Ports tab

## 💡 Common Workflows

### Basic Exploratory Data Analysis
1. Upload your Excel file to `data/raw/`
2. Open and run `notebooks/exploration.ipynb`
3. Follow the guided analysis process

### Building and Evaluating Models
1. Use the exploration notebook to identify the best model
2. For more advanced tuning, use the hyperparameter tuning script:
   ```bash
   python src/hyperparameter_tuning.py --data data/raw/your_file.xlsx --target target_column --problem-type regression
   ```

### Making Predictions on New Data
1. Use the predict script with your trained model:
   ```bash
   python src/predict.py --model models/your_model.pkl --input data/raw/new_data.xlsx
   ```
2. Find predictions in the `predictions/` directory

## 🔍 Troubleshooting

- **ImportError**: Make sure you're running scripts from the project root directory
- **FileNotFoundError**: Ensure that data files are in the `data/raw/` directory
- **Package not found**: Run `pip install -r requirements.txt` to install dependencies

For more help or to report issues, please contact the project administrator.