import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matminer.datasets import load_dataset
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Set style for plotting
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def explore_features(df, target_col):
    """Visualizes basic feature distributions and correlations."""
    print("\n--- Step 2.1: Feature Exploration ---")
    # Plot target distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df[target_col], kde=True)
    plt.title(f"Distribution of {target_col}")
    plt.savefig("target_distribution.png")
    
    # Correlation with target for top features
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
    print("Top 10 features correlated with target:")
    print(correlations.head(11)) # including target itself

def remove_redundant_features(df, target_col, threshold=0.95):
    """Removes features that are highly correlated with each other."""
    print(f"\n--- Step 2.2: Removing Redundant Features (threshold={threshold}) ---")
    X = df.drop(columns=["formula", "composition", target_col])
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Dropping {len(to_drop)} redundant features.")
    X_new = X.drop(columns=to_drop)
    print(f"Remaining features: {X_new.shape[1]}")
    
    return X_new, df[target_col]

def compare_models(X, y):
    """Compares multiple ML models using cross-validation."""
    print("\n--- Step 3: Model Comparison ---")
    
    # Scale features for SVM and Linear Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "SVM (RBF)": SVR(kernel='rbf')
    }
    
    results = []
    names = []
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        # Use scaled data for Linear and SVM
        data = X_scaled if name in ["Linear Regression", "SVM (RBF)"] else X
        cv_scores = cross_val_score(model, data, y, cv=cv, scoring='r2')
        results.append(cv_scores)
        names.append(name)
        print(f"{name:20}: R2 = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Boxplot for comparison
    plt.figure(figsize=(10, 6))
    plt.boxplot(results, labels=names)
    plt.title("Model Comparison (R2 Score)")
    plt.ylabel("R2 Score")
    plt.savefig("model_comparison.png")
    
    return names[np.argmax([r.mean() for r in results])]

def main():
    print("--- Step 1: Loading Dataset ---")
    df = load_dataset("elastic_tensor_2015")
    target_col = "G_VRH"
    df = df[["formula", target_col]]
    
    print("\n--- Step 2: Featurization ---")
    df = StrToComposition().featurize_dataframe(df, "formula")
    ep_featurizer = ElementProperty.from_preset(preset_name="magpie")
    df = ep_featurizer.featurize_dataframe(df, col_id="composition")
    df = df.dropna()
    
    # 2.1 Feature Exploration
    explore_features(df, target_col)
    
    # 2.2 Remove redundancy
    X_clean, y = remove_redundant_features(df, target_col)
    
    # 3. Model Comparison
    best_model_name = compare_models(X_clean, y)
    print(f"\nBest performing model: {best_model_name}")

    # 4. Final Model Training & Visualization (using best model - e.g., XGBoost/RF)
    print(f"\n--- Step 4: Final Training with {best_model_name} ---")
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)
    
    if best_model_name == "XGBoost":
        model = XGBRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Parity plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel(f"Actual {target_col} (GPa)")
    plt.ylabel(f"Predicted {target_col} (GPa)")
    plt.title(f"Final Model: {best_model_name}")
    plt.savefig("final_model_parity_plot.png")

if __name__ == "__main__":
    main()
