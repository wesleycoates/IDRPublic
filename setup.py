from setuptools import setup, find_packages

setup(
    name="predictive_analytics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        # Add other dependencies
    ],
)