import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
import time

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

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=500, solver='liblinear'))
])

print("Starting model training...")
start_time = time.time()
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
pipeline.fit(X_train, y_train)
end_time = time.time()
print(f"Model training time: {end_time - start_time:.2f} seconds")

print("Making predictions...")
start_time = time.time()
y_pred = pipeline.predict(X_val)
end_time = time.time()
print(f"Prediction time: {end_time - start_time:.2f} seconds")

accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.2f}')

joblib.dump(pipeline, 'logistic_regression_pipeline.pkl')
