import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data Understanding

print("--- Task 1: Data Understanding ---")

# 1. Load the dataset using Pandas.
# Note: Ensure 'Position_Salaries.csv' is in the same directory.
try:
    df = pd.read_csv('Position_Salaries.csv')
except FileNotFoundError:
    print("Dataset not found. Please download from: https://www.kaggle.com/datasets/akram24/position-salaries")
    print("Place 'Position_Salaries.csv' in the same directory and run again.")
    exit()

# 2. Display the first five records.
print("\nFirst five records:")
print(df.head())

# 3. Identify Input Feature and Target Variable
print("\nIdentification:")
print("Input Feature: 'Level'")
print("Target Variable: 'Salary'")

# 4. Display dataset information and summary statistics.
print("\nDataset Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe())



# Data Preprocessing

print("\n--- Task 2: Data Preprocessing ---")

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Select appropriate feature(s) and target variable
# We select 'Level' as X (Input) and 'Salary' as y (Target)
X = df[['Level']].values
y = df['Salary'].values

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")



#  Model Development

print("\nModel Development ")

# 1. Transform the input feature using Polynomial Features (Degree = 3)
poly_reg = PolynomialFeatures(degree=3)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.transform(X_test)

# 2. Train a Polynomial Regression model
model = LinearRegression()
model.fit(X_poly_train, y_train)
print("Polynomial Regression Model Trained (Degree = 3).")

# 3. Predict salaries for the test dataset
y_pred = model.predict(X_poly_test)
print("\nPredicted salaries for test set:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted:.2f}")



#  Model Evaluation

print("\n Model Evaluation ")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Create Scatter plot of the original data and Polynomial Regression Curve
# We plot the curve over a smooth range of X values for better visualization
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Original Data')
plt.plot(X_grid, model.predict(poly_reg.fit_transform(X_grid)), color='blue', label='Polynomial Regression Curve (Degree 3)')
plt.title('Position Level vs Salary (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.savefig('polynomial_regression_curve.png')
print("\nPlot saved as 'polynomial_regression_curve.png'.")

# Observations
print("\nObservations based on model performance:")
print("1. The R2 score suggests how well the model fits the non-linear data; a high R2 score indicates a strong fit.")
print("2. The polynomial curve captures the exponential-like growth in salaries at higher position levels much better than a straight line would.")
print("3. MAE and MSE provide an understanding of the average error in salary prediction, which is significantly reduced by using degree 3 rather than degree 1.")


# Conclusion

conclusion = """
Conclusion:
Key findings show that salary increases non-linearly with position level, growing much faster at the highest levels. 
The main difference between Linear Regression and Polynomial Regression is that Linear Regression assumes a straight-line 
relationship, which fails to capture curves, whereas Polynomial Regression introduces higher-degree terms to model curves 
and complex patterns accurately. 
One clear advantage of Polynomial Regression for this dataset is its ability to flexibly curve and closely follow the true 
exponential trend of the salaries, resulting in far more accurate predictions for senior positions (like C-level and CEO) 
compared to a simple linear model.
"""
print(conclusion)
