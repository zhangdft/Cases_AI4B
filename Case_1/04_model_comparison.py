# Stage 04: Model Comparison
# This script compares multiple machine learning models using all features 
# and K-fold Cross-Validation for robust performance evaluation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Set visual style
sns.set_theme(style="whitegrid")

# 1. Load Data
data_path = "resources/salary_prediction/job_salary_prediction_dataset.csv"
print("Loading dataset...")
df = pd.read_csv(data_path)

# Using 10,000 samples for cross-validation efficiency
sample_df = df.sample(10000, random_state=42)

# Separate features (X) and target (y)
X = sample_df.drop('salary', axis=1)
y = sample_df['salary']

# 2. Preprocessing Logic
num_cols = ['experience_years', 'skills_count', 'certifications']
ord_cols = ['education_level', 'company_size']
nom_cols = ['job_title', 'industry', 'location', 'remote_work']

education_order = ['High School', 'Diploma', 'Bachelor', 'Master', 'PhD']
company_order = ['Startup', 'Small', 'Medium', 'Large', 'Enterprise']

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

# 3. Define Models
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# 4. K-fold Cross-Validation (K=5)
print("\nPerforming 5-fold Cross-Validation for multiple models...")
cv_results = {}
for name, model in models.items():
    # Create a pipeline that includes preprocessing and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Calculate R2 score with CV
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
    cv_results[name] = scores
    print(f"  - {name}: Mean R2 = {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# 5. Visualizing Model Performance
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(cv_results))
plt.title('Model Comparison (R2 Score Distribution)', fontsize=14)
plt.ylabel('R2 Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('model_comparison_r2_all_features.png')
print("\nSaved model comparison boxplot as 'model_comparison_r2_all_features.png'")

# 6. Final Evaluation of Best Model (e.g., XGBoost)
best_model_name = 'XGBoost'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', models[best_model_name])
])

print(f"\nFinal training on 80% split using {best_model_name}...")
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

print(f"\n--- Best Model ({best_model_name}) Final Metrics ---")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f} USD")

# Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'Actual vs Predicted Salaries ({best_model_name})', fontsize=14)
plt.xlabel('Actual Salary (USD)', fontsize=12)
plt.ylabel('Predicted Salary (USD)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('actual_vs_predicted_all_features.png')
print("Saved actual vs predicted plot as 'actual_vs_predicted_all_features.png'")

print("\nModel comparison and selection completed.")
