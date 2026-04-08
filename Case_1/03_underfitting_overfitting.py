# Stage 03: Underfitting and Overfitting (All Features)
# This script demonstrates model complexity using all features from the dataset.
# It includes handling of missing values, encoding of categorical features, 
# and polynomial feature expansion to show the transition from underfitting to overfitting.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Load Data
data_path = "resources/salary_prediction/job_salary_prediction_dataset.csv"
print("Loading dataset...")
df = pd.read_csv(data_path)

# Sample the data for faster execution while still being representative
# Using 10,000 samples is enough to see the patterns without overwhelming memory
sample_df = df.sample(10000, random_state=42)

# Separate features (X) and target (y)
X = sample_df.drop('salary', axis=1)
y = sample_df['salary']

# 2. Preprocessing Logic (Referencing salary-prediction.ipynb)

# Define column types
num_cols = ['experience_years', 'skills_count', 'certifications']
ord_cols = ['education_level', 'company_size']
nom_cols = ['job_title', 'industry', 'location', 'remote_work']

# Define ordinal mappings
education_order = ['High School', 'Diploma', 'Bachelor', 'Master', 'PhD']
company_order = ['Startup', 'Small', 'Medium', 'Large', 'Enterprise']

# Preprocessing Pipeline:
# - SimpleImputer handles missing values (using 'mean' for numeric, 'most_frequent' for categorical)
# - OrdinalEncoder/OneHotEncoder for categorical data
# - StandardScaler for all features (crucial before polynomial expansion)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[education_order, company_order])),
    ('scaler', StandardScaler())
])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('ord', ordinal_transformer, ord_cols),
        ('nom', nominal_transformer, nom_cols)
    ])

# 3. Model Training and Error Analysis
# We will test degrees 1, 2, and 3. 
# Degree 1: Underfitting (Linear model on all features)
# Degree 2: Balanced/Complex (Captures interactions)
# Degree 3: Overfitting (Very high dimensionality on 10k samples)
degrees = [1, 2, 3]

train_errors = []
test_errors = []

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nStarting model evaluation across different degrees of complexity...")

for degree in degrees:
    # Create a full pipeline: Preprocessing -> Polynomial Features -> Linear Regression
    # Note: PolynomialFeatures is applied AFTER scaling the preprocessed features
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=degree)),
        ('regressor', LinearRegression())
    ])
    
    # Train the model
    print(f"Training model with Polynomial Degree {degree}...")
    model_pipeline.fit(X_train, y_train)
    
    # Predict
    y_train_pred = model_pipeline.predict(X_train)
    y_test_pred = model_pipeline.predict(X_test)
    
    # Calculate error (Root Mean Squared Error)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    
    print(f"  - Degree {degree}: Train RMSE = {train_rmse:.2f}, Test RMSE = {test_rmse:.2f}")

# 4. Visualization of the Error Curve
# This clearly shows where the model starts to overfit.
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, label='Training Error (RMSE)', marker='o', color='blue', linewidth=2)
plt.plot(degrees, test_errors, label='Testing Error (RMSE)', marker='s', color='red', linewidth=2)

# Highlighting underfitting and overfitting areas
plt.annotate('Underfitting (Simple)', xy=(1, test_errors[0]), xytext=(1.2, test_errors[0] + 5000),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Potential Overfitting (Too Complex)', xy=(3, test_errors[2]), xytext=(2.2, test_errors[2] + 5000),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('Bias-Variance Tradeoff: Error vs Model Complexity', fontsize=14)
plt.xlabel('Polynomial Degree (Complexity)', fontsize=12)
plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=12)
plt.xticks(degrees)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('error_curve_all_features.png')
print("\nSaved error curve plot as 'error_curve_all_features.png'")

# 5. Summary Analysis
print("\n--- Final Analysis ---")
print(f"Lowest Training Error: Degree {degrees[np.argmin(train_errors)]}")
print(f"Lowest Testing Error (Best Generalization): Degree {degrees[np.argmin(test_errors)]}")

if test_errors[-1] > test_errors[-2] * 1.1:
    print("Observation: Testing error significantly increased at high degree, indicating OVERFITTING.")
elif train_errors[0] > train_errors[1] * 1.1:
    print("Observation: High training error at degree 1 indicates potential UNDERFITTING.")

print("\nModel complexity analysis with all features completed.")
