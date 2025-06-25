# World Bank Data Analysis

## Introduction

This project provides a streamlined pipeline for processing and evaluating World Bank country-level indicators. The goal is to clean the dataset, handle missing values, and apply dimensionality reduction techniques to uncover key patterns and insights across different countries.

## Technologies

- **Python 3.9**
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: scipy, scikit-learn (PCA)

## Installation

1. Clone the repository:
   ```bash
   git clone https://your-repo-url/world_bank-master.git
   cd world_bank-master
   ```
2. Create and activate a virtual environment:
   ```bash
   conda create myenv
   conda actiavate myenv
   ```
3. Install dependencies:
   ```bash
   conda install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
   ```

## Usage

- **Jupyter Notebook**: Launch `sample.ipynb` to follow a step-by-step walkthrough of data loading, cleaning, and PCA analysis:
- **Modules**: The `worldBank` package contains two main modules:
  - `data_processing.py`: Functions to inspect and clean missing data
  - `evaluation.py`: Functions to apply PCA and evaluate components

## Project Structure

```
world_bank-master/           # Root folder
├── data/                    # Raw data files
│   └── CountryWorldBank.csv # World Bank indicators
├── worldBank/               # Source code package
│   ├── data_processing.py   # Data cleaning and missing-value utilities
│   └── evaluation.py        # PCA and evaluation routines
├── sample.ipynb             # Notebook demonstrating the workflow
└── README.md                # Project overview (this file)
```
