# Stage 02: Feature Analysis
# This script analyzes the correlation between features to identify potential redundancies.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set visual style
sns.set_theme(style="whitegrid")

# Define file path
data_path = "resources/salary_prediction/job_salary_prediction_dataset.csv"

# Load data
print("Loading dataset...")
df = pd.read_csv(data_path)

# 1. Label Encoding Categorical Variables for Correlation Analysis
# We create a copy for analysis to keep the original data intact.
df_encoded = df.copy()
le = LabelEncoder()

categorical_cols = df_encoded.select_dtypes(include=['object']).columns
print("\nEncoding categorical columns:", list(categorical_cols))

for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# 2. Correlation Matrix
print("\n--- Correlation Matrix ---")
corr_matrix = df_encoded.corr()
print(corr_matrix['salary'].sort_values(ascending=False))

# 3. Heatmap of Correlations
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of All Features')
plt.savefig('correlation_heatmap.png')
print("\nSaved correlation heatmap as 'correlation_heatmap.png'")

# 4. Boxplot for Education Level vs Salary
# Education levels are often ordinal. Let's see the distribution.
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='education_level', y='salary', 
            order=['High School', 'Diploma', 'Bachelor', 'Master', 'PhD'])
plt.title('Education Level vs Salary Distribution')
plt.xlabel('Education Level')
plt.ylabel('Salary (USD)')
plt.savefig('education_vs_salary.png')
print("Saved education vs salary boxplot as 'education_vs_salary.png'")

# 5. Boxplot for Industry vs Salary
plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='industry', y='salary')
plt.xticks(rotation=45)
plt.title('Industry vs Salary Distribution')
plt.xlabel('Industry')
plt.ylabel('Salary (USD)')
plt.savefig('industry_vs_salary.png')
print("Saved industry vs salary boxplot as 'industry_vs_salary.png'")

print("\nFeature analysis completed.")
