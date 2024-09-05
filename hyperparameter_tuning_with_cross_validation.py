import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib
import os
import time
import gc

# Define file paths
train_path = 'D:\\microsoft\\GUIDE_Train.csv'
test_path = 'D:\\microsoft\\GUIDE_Test.csv'
output_dir = 'D:\\microsoft\\GUIDE_Train_Split'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to load and optimize data in chunks
def load_and_optimize_csv(file_path, chunk_size=100000):
    chunks = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)
    df_list = []

    for chunk in chunks:
        for col in chunk.select_dtypes(include=['float64']).columns:
            chunk[col] = chunk[col].astype('float32')
        for col in chunk.select_dtypes(include=['int64']).columns:
            chunk[col] = chunk[col].astype('int32')
        for col in chunk.select_dtypes(include=['object']).columns:
            chunk[col] = chunk[col].astype('category')
        
        numerical_cols = chunk.select_dtypes(include=['float32', 'int32']).columns
        chunk[numerical_cols] = chunk[numerical_cols].fillna(chunk[numerical_cols].median())
        
        categorical_cols = chunk.select_dtypes(include=['category']).columns
        for col in categorical_cols:
            chunk[col] = chunk[col].cat.add_categories(['missing'])
            chunk[col] = chunk[col].fillna('missing')
        
        df_list.append(chunk)
    
    df_combined = pd.concat(df_list, ignore_index=True)
    return df_combined

print("Starting data loading and processing...")
start_time = time.time()
df_train = load_and_optimize_csv(train_path)
df_test = load_and_optimize_csv(test_path)
end_time = time.time()
print(f"Data loading and processing time: {end_time - start_time:.2f} seconds")

# Define the target variable and features for the training dataset
X = df_train.drop(columns=['IncidentGrade'])
y = df_train['IncidentGrade']

# Handle missing values in the target variable
y.fillna(y.mode()[0], inplace=True)

if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    raise ValueError("There are still missing values in the dataset.")

numerical_features = X.select_dtypes(include=['float32', 'int32']).columns
categorical_features = X.select_dtypes(include=['category']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define the Random Forest model
model = RandomForestClassifier()

# Define a pipeline with preprocessing and the Random Forest model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Define parameter distributions for RandomizedSearchCV
param_dist = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

# Use RandomizedSearchCV with cross-validation
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=16,  # Number of parameter combinations to sample
    cv=5,  # 5-fold cross-validation
    scoring='f1_macro',  # Use macro-F1 score for scoring
    n_jobs=2,  # Use fewer CPUs to avoid memory issues
    verbose=2,
    random_state=42
)

print("Starting model training with Random Search and Cross-Validation...")
start_time = time.time()
random_search.fit(X, y)  # Use entire dataset for RandomizedSearchCV
end_time = time.time()
print(f"Model training time: {end_time - start_time:.2f} seconds")

print("Best parameters found: ", random_search.best_params_)

print("Making predictions...")
start_time = time.time()
y_pred = random_search.best_estimator_.predict(X)
end_time = time.time()
print(f"Prediction time: {end_time - start_time:.2f} seconds")

# Evaluate performance metrics
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred, average='macro')
precision = precision_score(y, y_pred, average='macro')
recall = recall_score(y, y_pred, average='macro')

print(f'Overall Accuracy: {accuracy:.2f}')
print(f'Macro F1 Score: {f1:.2f}')
print(f'Macro Precision: {precision:.2f}')
print(f'Macro Recall: {recall:.2f}')

# Detailed classification report
print("\nClassification Report:\n", classification_report(y, y_pred))

# Save the best model
joblib.dump(random_search.best_estimator_, 'random_forest_pipeline.pkl')

# Additional cross-validation to check consistency
print("Performing additional cross-validation...")
cv_scores = cross_val_score(random_search.best_estimator_, X, y, cv=5, scoring='f1_macro')
print(f'Cross-Validation F1 Scores: {cv_scores}')
print(f'Mean Cross-Validation F1 Score: {cv_scores.mean():.2f}')

# Clean up memory
del df_train, df_test, X, y, random_search
gc.collect()
