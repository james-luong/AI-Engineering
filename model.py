### Import
# EDA
import pandas as pd
import os
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt


### EDA ###
df = pd.read_csv(os.path.join('data', 'Tetuan City power consumption.csv'))

# Data types of all columns
print("Data Types of Each Column")
print(df.dtypes)
print("\n")

# Basic statistical summary of numeric columns
print("Statistical Summary of Numeric Columns")
print(df.describe())
print("\n")

# Check for missing values
print("Missing Values in Each Column")
print(df.isnull().sum())


# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Remove duplicates
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Shape after removing duplicates: {df.shape}")

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.savefig('results/missing_values_heatmap.png')  # Save for report
plt.show()

# Create boxplots for numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns

# Plot multiple boxplots
fig, axes = plt.subplots(nrows=len(numerical_cols)//3 + 1, ncols=3, figsize=(15, 10))
axes = axes.ravel()

for i, col in enumerate(numerical_cols):
    if i < len(axes):
        axes[i].boxplot(df[col].dropna(), showfliers=False)
        axes[i].set_title(col)

# Hide empty subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('results/boxplots_numerical.png')  # Save for report
plt.show()

# Remove outliers using IQR method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply to numerical columns (be careful with target variable)
for col in numerical_cols:
    if col != 'target_column':  # Replace with actual target name
        original_shape = df.shape
        df = remove_outliers_iqr(df, col)
        print(f"Removed {original_shape[0] - df.shape[0]} outliers from {col}")

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns
target_col = 'Zone 1 Power Consumption'

# Identify predictors
exclude_cols = ['DateTime', 'Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']
predictors = [col for col in df.columns if col not in exclude_cols]

print(f"List of predictor columns: {predictors}")

# Summary stats table
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(df[predictors].describe())

# Distribution plots for key predictors
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.ravel()

for i, col in enumerate(predictors[:6]):  # Plot first 6 predictors
    axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('results/univariate_distributions.png')  # Save for report
plt.show()

# Identify numerical and categorical columns
target = 'Zone 1 Power Consumption'

# Correlation of predictors with target
corr_matrix = df[predictors + [target]].corr()
print("Correlation Matrix:\n", corr_matrix[target].sort_values(ascending=False))

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig('results/correlation_heatmap.png')  # Save for report
plt.show()

# Correlation matrix for Power Consumption columns only
power_cols = ['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']
power_corr = df[power_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(power_corr, annot=True, cmap='YlGnBu', center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Among Power Consumption Zones', fontsize=16)
plt.tight_layout()
plt.savefig('results/power_consumption_correlation.png')  # Save for report
plt.show()

# Top correlations with target
target_correlations = corr_matrix[target].sort_values(ascending=False)
print("\nTop features correlated with target:")
print(target_correlations)

top_features = target_correlations.index[1:4]  # Top 3 features excluding the target itself
if len(top_features) > 0:
    sns.pairplot(df[list(top_features) + [target]], diag_kind='kde')
    plt.suptitle('Pairplot of Top Features', y=1.02)
    plt.savefig('results/pairplot_top_features.png')  # Save for report
    plt.show()

# Visualizing the relationship between the strongest predictor and the Target
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df.sample(5000), x=top_features[0], y=target, alpha=0.4)
plt.title(f'Relationship: {top_features[0]} vs. {target} (Sampled)', fontsize=14)
plt.xlabel(top_features[0])
plt.ylabel(target)
plt.savefig('results/scatter_top_feature.png')  # Save for report
plt.show()

# Determine problem type
if df[target].dtype in ['float64', 'int64']:
    print("Problem Type: Regression (Numerical Target)")

    # Equal-frequency binning into 3 classes
    df["Target_Class"], bin_edges = pd.qcut(df[target], q=3, labels=['Low', 'Medium', 'High'], retbins=True)
    
    print("Binned target variable into 3 classes:")
    print(df["Target_Class"].value_counts())
    target_to_plot = "Target_Class"
else:
    print("Problem Type: Classification (Categorical Target)")
    target_to_plot = target

# Visualise target distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x=target_to_plot, palette='magma', order=['Low', 'Medium', 'High'])
plt.title('Distribution of Zone 1 Power Consumption Classes', fontsize=14)
plt.xlabel('Power Consumption Category', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('results/target_distribution.png')  # Save for report
plt.show()

# Summary statistics of target classes
print("\nSummary Statistics of Target Classes")
class_counts = df[target_to_plot].value_counts().sort_index()
class_percentages = df[target_to_plot].value_counts(normalize=True).sort_index() * 100

summary_df = pd.DataFrame({
    'Count': class_counts,
    'Percentage (%)': class_percentages.map('{:.2f}%'.format)
})
print(summary_df)

# Imbalance Check 
max_pct = class_percentages.max()
if max_pct > 80:
    print(f"\nALERT: Severe imbalance detected ({max_pct:.2f}%).")
else:
    print(f"\nBalance Check: No severe imbalance. Maximum class density is {max_pct:.2f}%.")

# Show the numerical ranges for each bin
print("\nBin Boundaries (Quantile-based)")
for i in range(len(bin_edges)-1):
    label = ['Low', 'Medium', 'High'][i]
    print(f"Class {label}: ({bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}]")



### FEATURE ENGINEERING ###
# Target Discretization (Equal Frequency Binning)
# df['Target_Class'] = pd.qcut(df['Zone 1 Power Consumption'], q=3, labels=['Low', 'Medium', 'High'])

# Feature Normalisation & Transformation
guassian_features = ['Temperature', 'Humidity']
outlier_features = ['Wind Speed', 'general diffuse flows', 'diffuse flows']

# Create new features
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek
df['Month'] = df['DateTime'].dt.month

df_scaled = df.copy()

# Apply StandardScaler to Gaussian features
scaler_gaussian = StandardScaler()
df_scaled[guassian_features] = scaler_gaussian.fit_transform(df[guassian_features])

# Apply RobustScaler to outlier features
scaler_outliers = RobustScaler()
df_scaled[outlier_features] = scaler_outliers.fit_transform(df[outlier_features])

# Encode categorical variables

### Sets Definition
# Set 1: Original features
X_set1 = df[['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']]

# Set 2: Top correlated features with target
correlations = df_scaled.drop(columns=['DateTime', 'Target_Class'])
correlations = correlations.corr()['Zone 1 Power Consumption'].sort_values(ascending=False)
top_features = correlations.index[1:4] # Top 3 features excluding target
X_set2 = df_scaled[top_features]

# Set 3: Scaled + engineered features
X_set3 = df_scaled.drop(columns=['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption', 'DateTime', 'Target_Class'])

# Set 4: PCA Dimensionality Reduction
pca = PCA(n_components=3)
X_set4 = pd.DataFrame(pca.fit_transform(X_set3))

# Set 5: Domain knowledge features
domain_features = ['Temperature', 'Hour', 'Month', 'DayOfWeek']
X_set5 = df_scaled[domain_features]

feature_sets = {
    "Set 1: Original Features": X_set1,
    "Set 2: Top Correlated Features": X_set2,
    "Set 3: Scaled + Engineered Features": X_set3,
    "Set 4: PCA Features": X_set4,
    "Set 5: Domain Knowledge Features": X_set5
}

y = df['Target_Class']

sets_info = {
    "Sets": [],
    "Decision Tree Accuracy": [],
    "Baseline SVM CV Accuracy": [],
    "Baseline SVM CV Std Dev": [],
    "Tuned SVM CV Accuracy": [],
    "Tuned SVM CV Std Dev": [],
    "Parameters Tuning Results": [],
    "SVM with K-Best Accuracy": [],
    "SVM with PCA Accuracy": []
}

### Task 4: Baseline Model Development
for name, X_features in feature_sets.items():
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.3, random_state=42)
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    y_pred_dt = dt_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred_dt)
    print(f"{name} \n- Decision Tree Baseline Accuracy: {acc:.4f}")
    sets_info["Sets"].append(name)
    sets_info["Decision Tree Accuracy"].append(acc)

    ### Task 6 + 7: SVC Model Improvement and Tuning
    sample_size = 5000
    X_train_svc = X_train[:sample_size]  # Sample 5000 rows for tuning
    y_train_svc = y_train[:sample_size]
    X_test_svc = X_test[:sample_size]
    y_test_svc = y_test[:sample_size]

    # Task 6: Baseline SVM with 5-fold CV
    svm_baseline = SVC()
    cv_scores = cross_val_score(svm_baseline, X_train_svc, y_train_svc, cv=5, error_score='raise')
    print(f"- Baseline SVM CV Average Accuracy: {cv_scores.mean():.4f} \n- Standard Deviation: {cv_scores.std():.4f}")
    sets_info["Baseline SVM CV Accuracy"].append(cv_scores.mean())
    sets_info["Baseline SVM CV Std Dev"].append(cv_scores.std())

    # Task 7: Hyperparameter Tuning with GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
    grid.fit(X_test_svc, y_test_svc)

    print("- Best Parameters found:", grid.best_params_)
    sets_info["Parameters Tuning Results"].append(grid.best_params_)

    # Tuned SVM
    best_svm = SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])

    # 5-fold cross validation for tuned SVM
    cv_scores = cross_val_score(best_svm, X_test_svc, y_test_svc, cv=5)
    print(f"- Tuned SVM CV Average Accuracy: {cv_scores.mean():.4f} \n- Standard Deviation: {cv_scores.std():.4f}")
    sets_info["Tuned SVM CV Accuracy"].append(cv_scores.mean())
    sets_info["Tuned SVM CV Std Dev"].append(cv_scores.std())

    ### Task 8 + 9: K-Best Feature Selection and PCA Comparison
    # Task 8: K-Best Feature Selection
    selector = SelectKBest(score_func=f_classif, k=3)
    X_train_kbest = selector.fit_transform(X_train_svc, y_train_svc)
    X_test_kbest = selector.transform(X_test_svc)

    svc_kbest = SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
    svc_kbest.fit(X_train_kbest, y_train_svc)
    kbest_acc = accuracy_score(y_test_svc, svc_kbest.predict(X_test_kbest))
    print(f"- SVM with K-Best Features Accuracy: {kbest_acc:.4f}")
    sets_info["SVM with K-Best Accuracy"].append(kbest_acc)

    # Apply PCA - reducing to top 3 components
    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train_svc)
    X_test_pca = pca.transform(X_test_svc)

    # Train SVM on PCA features
    svm_pca = SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
    svm_pca.fit(X_train_pca, y_train_svc)
    pca_acc = svm_pca.score(X_test_pca, y_test_svc)
    print(f"- SVM with PCA Accuracy: {pca_acc:.4f}\n")
    sets_info["SVM with PCA Accuracy"].append(pca_acc)
    
    
with open(os.path.join('results', 'Training Accuracy.txt'), 'w') as f:
    for i in range(len(sets_info["Sets"])):
        f.write(f"{sets_info['Sets'][i]}:\n")
        f.write(f"  - Decision Tree Accuracy: {sets_info['Decision Tree Accuracy'][i]:.4f}\n")
        f.write(f"  - Baseline SVM CV Accuracy: {sets_info['Baseline SVM CV Accuracy'][i]:.4f}\n")
        f.write(f"  - Baseline SVM CV Std Dev: {sets_info['Baseline SVM CV Std Dev'][i]:.4f}\n")
        f.write(f"  - Tuned SVM CV Accuracy: {sets_info['Tuned SVM CV Accuracy'][i]:.4f}\n")
        f.write(f"  - Tuned SVM CV Std Dev: {sets_info['Tuned SVM CV Std Dev'][i]:.4f}\n")
        f.write(f"  - Parameters Tuning Results: {sets_info['Parameters Tuning Results'][i]}\n")
        f.write(f"  - SVM with K-Best Accuracy: {sets_info['SVM with K-Best Accuracy'][i]:.4f}\n")
        f.write(f"  - SVM with PCA Accuracy: {sets_info['SVM with PCA Accuracy'][i]:.4f}\n\n")

sets_df = pd.DataFrame(sets_info)
sets_df.to_csv(os.path.join('data', 'sets_info.csv'), index=False)
