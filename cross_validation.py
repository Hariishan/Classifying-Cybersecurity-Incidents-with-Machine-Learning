import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

# Sample a portion of the dataset to avoid memory issues
X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.8, random_state=42)

numerical_features = X_sample.select_dtypes(include=['float32', 'int32']).columns
categorical_features = X_sample.select_dtypes(include=['category']).columns

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
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__bootstrap': [True]
}

# Use RandomizedSearchCV with cross-validation
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=8,  # Reduced number of iterations
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=1,  # Use a single CPU to save memory
    verbose=2,
    random_state=42
)

print("Starting model training with Random Search and Cross-Validation...")
start_time = time.time()
gc.collect()  # Force garbage collection
random_search.fit(X_sample, y_sample)  # Fit using the sampled dataset
end_time = time.time()
print(f"Model training time: {end_time - start_time:.2f} seconds")

print("Best parameters found: ", random_search.best_params_)

print("Making predictions...")
start_time = time.time()
y_pred = random_search.best_estimator_.predict(X_sample)
end_time = time.time()
print(f"Prediction time: {end_time - start_time:.2f} seconds")

accuracy = accuracy_score(y_sample, y_pred)
print(f'Overall Accuracy: {accuracy:.2f}')

# Save the best model
joblib.dump(random_search.best_estimator_, 'random_forest_pipeline.pkl')

# Additional cross-validation to check consistency
print("Performing additional cross-validation...")
cv_scores = cross_val_score(random_search.best_estimator_, X_sample, y_sample, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Mean Cross-Validation Accuracy: {cv_scores.mean():.2f}')

# Clean up memory
del df_train, df_test, X, y, random_search
gc.collect()
