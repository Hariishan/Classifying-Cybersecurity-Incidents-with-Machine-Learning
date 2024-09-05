import pandas as pd

# Load the dataset
df = pd.read_csv('D:\microsoft\GUIDE_Train.csv')
print("Dataset Shape:", df.shape)
print(df.info())
# Summary statistics for numerical columns
print("Summary statistics for numerical columns")
print(df.describe())

# Distribution of categorical variables
print("Distribution of categorical variables")
print(df['Category'].value_counts())
print(df['IncidentGrade'].value_counts())
print(df['SuspicionLevel'].value_counts())


# Checking the distribution of the target variable
print("Checking the distribution of the target variable")
print(df['IncidentGrade'].value_counts())

 #this is my intital inspection code provide me a eda code