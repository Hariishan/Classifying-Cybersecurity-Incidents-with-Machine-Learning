import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('D:\\microsoft\\GUIDE_Train.csv')
# Compute and Plot the Correlation Matrix for Numerical Columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns # Pairplot to Observe Relationships Between Numerical Features and 'IncidentGrade'
if 'IncidentGrade' in df.columns and not numerical_columns.empty:
    try:
        sns.pairplot(df, hue='IncidentGrade', palette='Set1', diag_kind='kde')
        plt.suptitle('Pairplot of Numerical Features Colored by IncidentGrade', y=1.02)
        plt.show()
    except ValueError as e:
        print(f"Error generating pairplot: {e}")
else:
    print("Column 'IncidentGrade' not found in the dataset or no numerical columns available.")