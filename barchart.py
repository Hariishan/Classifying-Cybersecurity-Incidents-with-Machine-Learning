import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('D:\\microsoft\\GUIDE_Train.csv')

# Bar Charts for Categorical Variables
categorical_columns = ['Category', 'IncidentGrade', 'SuspicionLevel']
for column in categorical_columns:
    if column in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=column, data=df)
        plt.title(f'Distribution of {column}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
