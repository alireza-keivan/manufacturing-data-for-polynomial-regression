# Manufacturing Data Analysis and Polynomial Regression

This project demonstrates a comprehensive approach to analyzing manufacturing data and building a polynomial regression model. The goal is to predict the **Quality Rating** of products based on various input features using both data preprocessing and machine learning techniques.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Code Explanation](#code-explanation)
  - [1. Importing Libraries](#1-importing-libraries)
  - [2. Loading the Dataset](#2-loading-the-dataset)
  - [3. Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
  - [4. Data Preprocessing](#4-data-preprocessing)
  - [5. Polynomial Feature Transformation](#5-polynomial-feature-transformation)
  - [6. Model Training and Evaluation](#6-model-training-and-evaluation)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [License](#license)

---

## Overview

This project uses a dataset from a manufacturing process to:
- Perform **exploratory data analysis (EDA)** to better understand the data distribution and relationships.
- Preprocess the data using **standardization** and generate **polynomial features** for regression.
- Train a **linear regression model** to predict the target variable (**Quality Rating**).
- Evaluate the model performance using metrics such as **R² Score** and **Root Mean Squared Error (RMSE)**.

The project is implemented in Python and relies on popular libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.

---

## Dataset

The dataset contains multiple features related to manufacturing processes. The target variable is **Quality Rating**, which is a numerical score representing the quality of a manufactured product. Each feature represents a different aspect of the manufacturing process.

**Key Information:**
- The dataset is stored as a CSV file.
- The path to the dataset can be specified in the code using the `dataset_path` variable.

---

## Project Workflow

The project follows this step-by-step workflow:
1. Import required libraries.
2. Load the dataset and inspect its structure.
3. Perform exploratory data analysis (EDA) to understand feature distributions and relationships.
4. Preprocess the data by scaling features and generating polynomial features.
5. Split the dataset into training and testing subsets.
6. Train a **Linear Regression model**.
7. Evaluate the model's performance using metrics like **R² Score** and **RMSE**.
8. Visualize the model's predictions against actual values.

---

## Code Explanation

### 1. Importing Libraries

```python
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation and CSV file handling
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For advanced data visualization
import warnings  # To suppress warnings
```

Here, we import essential libraries for data analysis, visualization, and machine learning. Warnings are suppressed to ensure clean output during runtime.

---

### 2. Loading the Dataset

```python
dataset_path = r'C:/Users/win10/OneDrive/اسناد/git-repo/manufacturing-data-for-polynomial-regression/manufacturing.csv'
data = pd.read_csv(dataset_path)

print("Preview of the dataset:")
print(data.head())
```

The dataset is loaded using the `pandas` library. The first few rows of the dataset are displayed to verify its structure.

---

### 3. Exploratory Data Analysis (EDA)

#### Visualizing Feature Distributions
```python
for column in data.columns:
    sns.histplot(data[column], kde=True, color='blue', bins=20)
    plt.title(f"Distribution of {column}")
```
![image](https://github.com/alireza-keivan/manufacturing-data-for-polynomial-regression/blob/main/src/features.png)
- Histograms with KDE overlays are used to visualize the distribution of each feature.
- This helps identify patterns, skewness, or outliers in the data.

#### Scatter Plots for Relationships
```python
for column in data.columns:
    sns.scatterplot(x=data[column], y=data['Quality Rating'], color='green')
    plt.title(f"{column} vs Quality Rating")
```
![image](https://github.com/alireza-keivan/manufacturing-data-for-polynomial-regression/blob/main/src/features%202.png)
- Scatter plots are used to visualize relationships between each feature and the target variable (`Quality Rating`).

#### Boxplot for Outlier Detection
```python
data.boxplot(patch_artist=True, boxprops=dict(facecolor='lightblue'))
```
- Boxplots help identify potential outliers in the dataset.
![image](https://github.com/alireza-keivan/manufacturing-data-for-polynomial-regression/blob/main/src/boxplot.png)
---

### 4. Data Preprocessing

#### Standardizing Features
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
```
- Features are scaled to have zero mean and unit variance using `StandardScaler`, which is essential for polynomial regression.

---

### 5. Polynomial Feature Transformation

```python
from sklearn.preprocessing import PolynomialFeatures

poly_transformer = PolynomialFeatures(degree=2)
X_poly = poly_transformer.fit_transform(X_scaled)
```
- Polynomial features are generated to capture non-linear relationships in the data.

---

### 6. Model Training and Evaluation

#### Splitting the Data
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
```
- The dataset is split into training and testing subsets (70%-30%).

#### Training a Linear Regression Model
```python
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
```
- A linear regression model is trained on the polynomial features.

#### Evaluating the Model
```python
from sklearn.metrics import r2_score, mean_squared_error

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
```
- The model's performance is evaluated using:
  - **R² Score**: Measures the proportion of variance explained by the model.
  - **RMSE**: Measures the average error of predictions.

#### Visualizing Predictions
```python
sns.scatterplot(x=y_test, y=y_test_pred, color='orange')
plt.title("Testing Data: Actual vs Predicted")
```
- Scatter plots compare the actual and predicted values to assess the model's accuracy.
![image](https://github.com/alireza-keivan/manufacturing-data-for-polynomial-regression/blob/main/src/scatter%201.png)
---

## Results

- **Training Data R² Score**: `0.927`
- **Testing Data R² Score**: `0.923`
- **Training Data RMSE**: `3.374`
- **Testing Data RMSE**: `3.915`


---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this code, provided proper attribution is given.
