import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import joblib
import numpy as np

# Load and Inspect Data
df = pd.read_csv('D:\\microsoft\\GUIDE_Train.csv')
print("Dataset Shape:", df.shape)
print(df.info())
print("Summary statistics for numerical columns:")
print(df.describe())

# Distribution of categorical variables
print("Distribution of categorical variables:")
print(df['Category'].value_counts())
print(df['IncidentGrade'].value_counts())
print(df['SuspicionLevel'].value_counts())

# EDA Section
# 1. Histograms for numerical features
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_features].hist(figsize=(15, 10), bins=30, edgecolor='black')
plt.suptitle('Histograms of Numerical Features', fontsize=16)
plt.show()

# 2. Box plots for numerical features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(len(numerical_features) // 3 + 1, 3, i + 1)
    sns.boxplot(x='IncidentGrade', y=feature, data=df)
    plt.title(f'Box Plot of {feature}')
plt.tight_layout()
plt.show()

# 3. Distribution of categorical features
plt.figure(figsize=(15, 5))
categorical_features = df.select_dtypes(include=['object']).columns
for i, feature in enumerate(categorical_features):
    plt.subplot(len(categorical_features) // 3 + 1, 3, i + 1)
    sns.countplot(y=feature, data=df)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# 4. Correlation matrix for numerical features
plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# 5. Pair plots for feature relationships
subset_features = numerical_features[:5]  # Adjust as needed
sns.pairplot(df[subset_features + ['IncidentGrade']], hue='IncidentGrade')
plt.suptitle('Pair Plot of Selected Features', y=1.02)
plt.show()

# 6. Class balance in target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='IncidentGrade', data=df)
plt.title('Distribution of Target Variable')
plt.show()

# 7. Relationship between categorical features and target variable
plt.figure(figsize=(15, 10))
for i, feature in enumerate(categorical_features):
    plt.subplot(len(categorical_features) // 3 + 1, 3, i + 1)
    sns.countplot(y=feature, hue='IncidentGrade', data=df)
    plt.title(f'{feature} vs IncidentGrade')
plt.tight_layout()
plt.show()

# Data Preprocessing
# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)  # Impute numerical missing values with median
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna('missing', inplace=True)  # Impute categorical missing values

# Feature and target variable separation
X = df.drop(columns=['IncidentGrade'])
y = df['IncidentGrade']

# Encoding categorical features
categorical_features = X.select_dtypes(include=['object']).columns
X[categorical_features] = X[categorical_features].astype('category')
encoder = LabelEncoder()
for col in categorical_features:
    X[col] = encoder.fit_transform(X[col])

# Data Splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model Selection and Training
# Preprocessing pipeline
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['category']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ]
)

# Define the Random Forest model and parameter grid
model = RandomForestClassifier(random_state=42)
param_dist = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__bootstrap': [True]
}

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=8,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)
print("Best parameters found:", random_search.best_params_)

# Model Evaluation
y_pred = random_search.best_estimator_.predict(X_val)
print("Validation Set Evaluation:")
print(f'Accuracy: {accuracy_score(y_val, y_pred):.2f}')
print(classification_report(y_val, y_pred))

# Cross-Validation
cv_scores = cross_val_score(random_search.best_estimator_, X, y, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Mean Cross-Validation Accuracy: {cv_scores.mean():.2f}')

# Final Evaluation on Test Set
df_test = pd.read_csv('D:\\microsoft\\GUIDE_Test.csv')
df_test.fillna(df_test.median(numeric_only=True), inplace=True)  # Impute numerical missing values
for col in df_test.select_dtypes(include=['object']).columns:
    df_test[col].fillna('missing', inplace=True)  # Impute categorical missing values

X_test = df_test.drop(columns=['IncidentGrade'])
y_test = df_test['IncidentGrade']

# Encoding for test set
for col in X_test.select_dtypes(include=['object']).columns:
    X_test[col] = encoder.transform(X_test[col])

y_test_pred = random_search.best_estimator_.predict(X_test)
print("Test Set Evaluation:")
print(f'Macro-F1 Score: {f1_score(y_test, y_test_pred, average="macro"):.2f}')
print(f'Precision: {precision_score(y_test, y_test_pred, average="macro"):.2f}')
print(f'Recall: {recall_score(y_test, y_test_pred, average="macro"):.2f}')

# Save the best model
joblib.dump(random_search.best_estimator_, 'random_forest_pipeline.pkl')

# Clean up memory
import gc
del df, X, y, X_train, X_val, y_train, y_val, X_test, y_test
gc.collect()
