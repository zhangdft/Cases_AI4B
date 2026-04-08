# Stage 01: Data Exploration
# This script loads the dataset and performs initial exploratory data analysis.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visual style
sns.set_theme(style="whitegrid")

# Define file path
data_path = "resources/salary_prediction/job_salary_prediction_dataset.csv"

# Load data
print("Loading dataset...")
df = pd.read_csv(data_path)

# 1. Basic Information
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Summary Statistics ---")
print(df.describe())

# 2. Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 3. Basic Visualization of the Target Variable (Salary)
plt.figure(figsize=(10, 6))
sns.histplot(df['salary'], kde=True, color='skyblue')
plt.title('Distribution of Salary')
plt.xlabel('Salary (USD)')
plt.ylabel('Frequency')
plt.savefig('salary_distribution.png')
print("\nSaved salary distribution plot as 'salary_distribution.png'")

# 4. Experience vs Salary (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df.sample(1000), x='experience_years', y='salary', alpha=0.5)
plt.title('Experience Years vs Salary (Sample of 1000)')
plt.xlabel('Experience Years')
plt.ylabel('Salary (USD)')
plt.savefig('experience_vs_salary.png')
print("Saved experience vs salary plot as 'experience_vs_salary.png'")

print("\nData exploration completed.")
