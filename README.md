# F1 Prediction Project

This project contains machine learning models and data analysis tools for Formula 1 race predictions.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository** (if not already done):

   ```bash
   git clone <repository-url>
   cd "F1 project"
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   # Using venv (built into Python)
   python -m venv venv

   # Activate the virtual environment
   # On macOS/Linux:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

The following packages will be installed:

- **numpy** (≥1.21.0): Numerical computing library
- **pandas** (≥1.3.0): Data manipulation and analysis
- **matplotlib** (≥3.5.0): Plotting and visualization
- **scikit-learn** (≥1.0.0): Machine learning library
- **imbalanced-learn** (≥0.8.0): Tools for handling imbalanced datasets
- **xgboost** (≥1.5.0): Gradient boosting library
- **joblib** (≥1.1.0): Parallel computing and model persistence
- **urllib3** (≥1.26.0): HTTP client library

## Usage

After installation, you can run the various Python scripts and Jupyter notebooks in the project:

- `data_scraping.py` - Data collection scripts
- `model_training.py` - Model training pipeline
- `prediction.py` - Make predictions with trained models
- `preprocessing.py` - Data preprocessing utilities

## Troubleshooting

If you encounter any issues during installation:

1. **Update pip**:

   ```bash
   pip install --upgrade pip
   ```

2. **Install dependencies individually** if the requirements.txt fails:

   ```bash
   pip install numpy pandas matplotlib scikit-learn imbalanced-learn xgboost joblib urllib3
   ```

3. **Check Python version**:

   ```bash
   python --version
   ```

4. **Verify installation**:
   ```bash
   python -c "import numpy, pandas, matplotlib, sklearn, imblearn, xgboost, joblib, urllib3; print('All packages installed successfully!')"
   ```

## Project Structure

- `data_scraping.py` - Data collection from F1 APIs
- `model_training.py` - Training machine learning models
- `prediction.py` - Making predictions with trained models
- `preprocessing.py` - Data preprocessing utilities
- `*.ipynb` - Jupyter notebooks for exploration and analysis
- `*.csv` - Data files
- `*.joblib` - Trained model files
- `*.png` - Generated visualizations and plots
- `results_report.md` - Project results and findings
