"""
Setup script for datavis package.
"""

from setuptools import setup, find_packages

setup(
    name="datavis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Comprehensive data visualization package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/datavis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "interactive": ["ipywidgets>=7.6.0", "IPython>=7.0.0"],
        "advanced": ["plotly>=5.0.0", "networkx>=2.5.0"],
        "geo": ["cartopy>=0.18.0"],
        "all": [
            "ipywidgets>=7.6.0", 
            "IPython>=7.0.0", 
            "plotly>=5.0.0", 
            "networkx>=2.5.0",
            "cartopy>=0.18.0"
        ],
    },
)