import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('D:\\microsoft\\GUIDE_Train.csv')


# Visualize Histograms for Numerical Columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(12, 8))
df[numerical_columns].hist(bins=20, edgecolor='black', figsize=(12, 8))
plt.suptitle('Histograms of Numerical Columns')
plt.show()






# Check for Missing Values
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)
