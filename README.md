# Position Salaries — Polynomial Regression

Predicts salary from position level using Polynomial Regression (degree 3), including data understanding, preprocessing, model training, and evaluation.

## Dataset

- **Source:** [Position_Salaries dataset on Kaggle](https://www.kaggle.com/datasets/akram24/position-salaries)
- Download `Position_Salaries.csv` and place it in the project root (same directory as the script).
- **Input feature:** `Level`
- **Target variable:** `Salary`

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

```bash
python polynomial_regression_salary.py
```

## What the script does

1. **Data Understanding** — loads the CSV, previews records, and prints dataset info/summary statistics.
2. **Data Preprocessing** — checks for missing values, selects `Level` (X) and `Salary` (y), and splits the data 80/20 into train/test sets.
3. **Model Development** — transforms `Level` into polynomial features (degree 3) and fits a `LinearRegression` model on the transformed features.
4. **Model Evaluation** — computes MAE, MSE, and R² on the test set, and saves a scatter plot with the fitted curve as `polynomial_regression_curve.png`.

## Output

- Console output with dataset summary, predictions, and evaluation metrics.
- `polynomial_regression_curve.png` — visualization of the original data points vs. the fitted polynomial curve.

## Key Findings

Salary increases non-linearly with position level, growing sharply at senior levels. Polynomial Regression captures this curve far better than a straight-line (Linear Regression) fit, giving much more accurate predictions for higher-level positions.
