# Data-science-projects

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv('sales_data.csv')  # Replace with your dataset path

# Quick look at data
print(data.head())
print(data.info())

# Check for missing values
print(data.isnull().sum())

# EDA - Visualize sales trend
plt.figure(figsize=(10,6))
sns.lineplot(x='Month', y='Sales', data=data)
plt.title('Monthly Sales Trend')
plt.show()

# Prepare Data for Modeling
# Assuming 'Month' is numeric and 'Sales' is target variable
X = data[['Month']]  # Features
y = data['Sales']    # Target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Plot predictions vs actual
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()
