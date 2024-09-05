import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('D:\\microsoft\\GUIDE_Train.csv')

# a. Handling Missing Data
# Identify missing values
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Strategy for handling missing data
# For numerical columns: imputation with the median value
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# For categorical columns: imputation with the most frequent value
cat_cols = df.select_dtypes(include=['object']).columns
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# b. Feature Engineering
# Example: Create new features from the 'Timestamp' column if it exists
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df.drop('Timestamp', axis=1, inplace=True)  # Drop original timestamp column

# Normalize numerical columns (optional, depending on the model requirements)
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# c. Encoding Categorical Variables
# Example: One-hot encode categorical columns
one_hot_encoder = OneHotEncoder(drop='first', sparse=False)
cat_encoded = one_hot_encoder.fit_transform(df[cat_cols])

# Create a DataFrame for encoded categorical variables
cat_encoded_df = pd.DataFrame(cat_encoded, columns=one_hot_encoder.get_feature_names_out(cat_cols))

# Concatenate the original DataFrame with the encoded columns and drop original categorical columns
df = pd.concat([df.drop(cat_cols, axis=1), cat_encoded_df], axis=1)

# Example: Label encode the target variable 'IncidentGrade'
if 'IncidentGrade' in df.columns:
    label_encoder = LabelEncoder()
    df['IncidentGrade'] = label_encoder.fit_transform(df['IncidentGrade'])

# Final DataFrame shape after preprocessing
print("\nDataFrame shape after preprocessing:", df.shape)

# Save the preprocessed data (optional)
df.to_csv('D:\\microsoft\\GUIDE_Train_Preprocessed.csv', index=False)
