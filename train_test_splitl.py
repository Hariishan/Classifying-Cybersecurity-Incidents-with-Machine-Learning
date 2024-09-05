import pandas as pd
from sklearn.model_selection import train_test_split
import os

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
        # Reduce memory usage by optimizing data types
        for col in chunk.select_dtypes(include=['float64']).columns:
            chunk[col] = chunk[col].astype('float32')
        for col in chunk.select_dtypes(include=['int64']).columns:
            chunk[col] = chunk[col].astype('int32')
        for col in chunk.select_dtypes(include=['object']).columns:
            chunk[col] = chunk[col].astype('category')
        
        # Handle missing values in numerical columns
        numerical_cols = chunk.select_dtypes(include=['float32', 'int32']).columns
        chunk[numerical_cols] = chunk[numerical_cols].fillna(chunk[numerical_cols].median())
        
        # Handle missing values in categorical columns
        categorical_cols = chunk.select_dtypes(include=['category']).columns
        for col in categorical_cols:
            chunk[col] = chunk[col].cat.add_categories(['missing'])
            chunk[col] = chunk[col].fillna('missing')
        
        df_list.append(chunk)
    
    # Combine chunks into a single DataFrame
    df_combined = pd.concat(df_list, ignore_index=True)
    
    return df_combined

# Load and process the datasets
df_train = load_and_optimize_csv(train_path)
df_test = load_and_optimize_csv(test_path)

# Define the target variable and features for the training dataset
X = df_train.drop(columns=['IncidentGrade'])
y = df_train['IncidentGrade']

# Handle missing values in the target variable
y.fillna(y.mode()[0], inplace=True)

# Verify there are no missing values remaining
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    raise ValueError("There are still missing values in the dataset.")

# Split the data into training and validation sets with stratification
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Save the splits into CSV files
X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
X_val.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
y_val.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)

print("Data splitting completed and saved.")
