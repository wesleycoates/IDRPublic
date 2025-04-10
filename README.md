# IDR Predictive Analytics project for Testing purposes

This repository contains a basic framework for data science and predictive analytics, set up specifically for use in GitHub CodeSpace. The topic is independent dispute resolution and is based on publicly available public use files (PUFs). 

## Project Structure

```
predictive-analytics/
│
├── data/                # Store data files here
│   └── raw/             # Place raw Excel file here
│
├── notebooks/           # Jupyter notebooks for exploration
│   └── exploration.ipynb
│
├── src/                 # Source code
│   ├── __init__.py
│   ├── data_loader.py   # Functions to load and preprocess data
│   ├── visualization.py # Visualization utilities
│   └── modeling.py      # Model building and evaluation
│
├── models/              # Saved models will be stored here
│
├── requirements.txt     # Project dependencies
│
└── README.md            # Project documentation
```

## Getting Started

### 1. Set Up the environment

When you the repository in GitHub CodeSpace, follow these steps to set up the environment:

```bash
# Install required packages
pip install -r requirements.txt

# Create any missing directories
mkdir -p data/raw models
```

### 2. Prepare Data

1. Upload PUF Excel file to the `data/raw/` directory
2. Update the file path in the notebook to point to the Excel file

### 3. Explore the Data

1. Open the `notebooks/exploration.ipynb` notebook
2. Follow the guided process to:
   - Load and clean the data
   - Visualize features and relationships
   - Select a target variable
   - Build and evaluate predictive models
   - Examine feature importance
   - Save the best model

### 4. Working with the Models

After a model has been built and saved, use it to make predictions on new data:

```python
from modeling import load_model

# Load the saved model
model = load_model('models/best_model.pkl')

# Make predictions on new data
predictions = model.predict(new_data)
```

## Key Features

This project provides:

- **Data Loading**: Easily load and clean data from Excel files
- **Visualization**: Create common visualizations for data exploration
- **Model Building**: Automatically evaluate multiple models to find the best performer
- **Feature Importance**: Understand which features have the most impact
- **Persistence**: Save and load models for future use

## Customizing for specific Needs

To adapt this framework for this specific project:

1. **Add more data sources**: Extend `data_loader.py` to handle different file formats
2. **Create custom visualizations**: Add the visualization functions to `visualization.py`
3. **Add new models**: Extend the model dictionaries in `modeling.py`
4. **Feature engineering**: Add feature engineering functions to `data_loader.py`

## Next Steps for Development

After mastering the basics:

1. Create a prediction script for batch processing
2. Implement more advanced feature engineering
3. Add hyperparameter tuning for better model performance
4. Implement cross-validation for more robust evaluation
5. Export models for deployment in production environments

## Requirements

This project uses the following Python packages:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter
- openpyxl (for Excel file support)