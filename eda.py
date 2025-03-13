import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
df = pd.read_csv('clean_data.csv')

# Univariate Analysis

# i. Summary statistics for numerical columns
print("Summary Statistics for Numerical Columns:")
print(df.describe())

# ii. Skewness and Variance
print("\nSkewness and Variance for Marks:")
print("Skewness for Marks: ", df['Marks'].skew())
print("Variance for Marks: ", df['Marks'].var())

# iii. Frequency distribution for categorical variables
print("\nFrequency Distribution for 'Status' column:")
print(df['Status'].value_counts())

print("\nFrequency Distribution for 'First Name' column:")
print(df['First Name'].value_counts())

# iv. Visualize Distributions using Histograms and Box Plots
# Histogram for 'Marks'
plt.figure(figsize=(10, 6))
sns.histplot(df['Marks'], kde=True)
plt.title('Distribution of Marks')
plt.show()

# Boxplot for 'Marks'
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Marks'])
plt.title('Boxplot for Marks')
plt.show()

# Histogram for 'Attendance (%)'
plt.figure(figsize=(10, 6))
sns.histplot(df['Attendance (%)'], kde=True)
plt.title('Distribution of Attendance (%)')
plt.show()

# Boxplot for 'Attendance (%)'
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Attendance (%)'])
plt.title('Boxplot for Attendance (%)')
plt.show()

# Bivariate Analysis

# i. Correlation Matrix for Numerical Variables
corr_matrix = df[['Marks', 'Attendance (%)']].corr()
print("\nCorrelation Matrix for Marks and Attendance (%):")
print(corr_matrix)

# Visualizing correlation matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix between Marks and Attendance')
plt.show()

# ii. Scatter plot for Marks vs Attendance (%)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Marks'], y=df['Attendance (%)'])
plt.title('Scatter Plot of Marks vs Attendance (%)')
plt.xlabel('Marks')
plt.ylabel('Attendance (%)')
plt.show()

# iii. Bar plots for categorical vs numerical variables
# Bar plot for Marks by Status
plt.figure(figsize=(10, 6))
sns.barplot(x='Status', y='Marks', data=df)
plt.title('Average Marks by Status')
plt.show()

# Box plot for Attendance (%) by Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='Status', y='Attendance (%)', data=df)
plt.title('Attendance Distribution by Status')
plt.show()

# Violin plot for Marks by Status
plt.figure(figsize=(10, 6))
sns.violinplot(x='Status', y='Marks', data=df)
plt.title('Distribution of Marks by Status')
plt.show()
 # Multivariate Analysis

# i. Pairplot for multiple variables
sns.pairplot(df[['Marks', 'Attendance (%)']])
plt.show()

# ii. Heatmap to visualize correlations among multiple variables
# Correlation matrix for all numerical columns
corr_matrix_all = df[['Attendance (%)', 'Marks']].corr()

# Heatmap of the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix_all, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for All Numerical Variables')
plt.show()

# iii. Grouped comparisons based on 'Status'
grouped_data = df.groupby('Status').agg({'Marks': 'mean', 'Attendance (%)': 'mean'}).reset_index()
print("\nGrouped Comparisons by Status:")
print(grouped_data)

# Bar plot for grouped comparisons
plt.figure(figsize=(10, 6))
sns.barplot(x='Status', y='Marks', data=grouped_data)
plt.title('Average Marks by Status')
plt.show()

# Combined Data Analysis

# i. Relationship between 'Marks' and 'Attendance (%)' by 'Status'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Marks', y='Attendance (%)', hue='Status', data=df)
plt.title('Marks vs Attendance (%) by Status')
plt.show()

# ii. Violin plot to visualize the distribution of Marks and Attendance by Status
plt.figure(figsize=(10, 6))
sns.violinplot(x='Status', y='Marks', data=df, inner='quart')
plt.title('Distribution of Marks by Status')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Status', y='Attendance (%)', data=df, inner='quart')
plt.title('Distribution of Attendance by Status')
plt.show()

# iii. Pairplot with 'Status' as the hue to observe combined relationships
sns.pairplot(df[['Marks', 'Attendance (%)', 'Status']], hue='Status')
plt.show()
