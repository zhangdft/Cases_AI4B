# Stage 05: Model Interpretation
# This script interprets the trained model by analyzing feature importance.
# It uses consistent preprocessing and the best performing model (XGBoost).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

# Set visual style
sns.set_theme(style="whitegrid")

# 1. Load Data
data_path = "resources/salary_prediction/job_salary_prediction_dataset.csv"
print("Loading dataset...")
df = pd.read_csv(data_path)

# Using 10,000 samples for interpretation
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

# 3. Train Best Model (XGBoost)
print("\nTraining XGBoost model for feature importance analysis...")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

pipeline.fit(X, y)

# 4. Extract Feature Names from Pipeline
# We need to retrieve the names of one-hot encoded features
feature_names_out = []

# Numeric features
feature_names_out.extend(num_cols)

# Ordinal features
feature_names_out.extend(ord_cols)

# Nominal features (One-Hot Encoded)
nominal_onehot_names = list(pipeline.named_steps['preprocessor'].named_transformers_['nom'].named_steps['onehot'].get_feature_names_out(nom_cols))
feature_names_out.extend(nominal_onehot_names)

# 5. Extract and Visualize Feature Importance
importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({'feature': feature_names_out, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False).head(20)

print("\nTop 20 Important Features (detailed):")
print(importance_df)

plt.figure(figsize=(12, 10))
sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
plt.title('Top 20 Detailed Feature Importances (XGBoost)', fontsize=14)
plt.xlabel('Importance Score (Gain)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('feature_importance_all_features.png')
print("\nSaved feature importance plot as 'feature_importance_all_features.png'")

# 6. Aggregated Importance by Original Category
aggregated_importance = {}

# Sum importance for one-hot encoded columns back into their original feature categories
for original in (num_cols + ord_cols + nom_cols):
    total = importance_df[importance_df['feature'].str.startswith(original)]['importance'].sum()
    aggregated_importance[original] = total

agg_df = pd.DataFrame(list(aggregated_importance.items()), columns=['feature', 'importance'])
agg_df = agg_df.sort_values(by='importance', ascending=False)

print("\n--- Aggregated Feature Importance ---")
print(agg_df)

plt.figure(figsize=(10, 6))
sns.barplot(data=agg_df, x='importance', y='feature', palette='coolwarm')
plt.title('Aggregated Importance of Original Features', fontsize=14)
plt.xlabel('Aggregated Importance Score', fontsize=12)
plt.ylabel('Original Feature', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('aggregated_feature_importance.png')
print("Saved aggregated feature importance plot as 'aggregated_feature_importance.png'")

print("\nModel interpretation completed.")
