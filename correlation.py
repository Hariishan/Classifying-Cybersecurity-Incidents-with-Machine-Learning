import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('D:\\microsoft\\GUIDE_Train.csv')

# Compute and Plot the Correlation Matrix for Numerical Columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

if not numerical_columns.empty:
    corr_matrix = df[numerical_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

# Distribution of the Target Variable 'IncidentGrade'
if 'IncidentGrade' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x='IncidentGrade', data=df)
    plt.title('Class Distribution of IncidentGrade')
    plt.show()
