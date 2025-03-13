import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('origional_data.csv')

# Step 1: Data Inspection
print("Initial Data:")
print(df.head())
print("\nData Info:")
print(df.info())

# Step 2: Handle Missing Values
# Impute missing numerical values with the mean
df['Marks'] = df['Marks'].fillna(df['Marks'].mean())
df['Attendance (%)'] = df['Attendance (%)'].fillna(df['Attendance (%)'].mean())

# Fill missing categorical values with a placeholder
df['First Name'] = df['First Name'].fillna('Unknown')
df['Last Name'] = df['Last Name'].fillna('Unknown')
df['Status'] = df['Status'].fillna('Unknown')

# Check for remaining missing values
print("\nMissing Values after Imputation:")
print(df.isnull().sum())

# Step 3: Handle Duplicates
df = df.drop_duplicates()

# Check for remaining duplicates
print("\nNumber of Duplicates after Removal:")
print(df.duplicated().sum())

# Step 4: Detect and Treat Outliers
# Boxplot for Marks to detect outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Marks'])
plt.title('Boxplot for Marks')
plt.show()

# Calculate IQR for Marks to remove outliers
Q1_marks = df['Marks'].quantile(0.25)
Q3_marks = df['Marks'].quantile(0.75)
IQR_marks = Q3_marks - Q1_marks
lower_bound_marks = Q1_marks - 1.5 * IQR_marks
upper_bound_marks = Q3_marks + 1.5 * IQR_marks

# Remove outliers in 'Marks'
df = df[(df['Marks'] >= lower_bound_marks) & (df['Marks'] <= upper_bound_marks)]

# Boxplot for Attendance to detect outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Attendance (%)'])
plt.title('Boxplot for Attendance')
plt.show()

# Calculate IQR for Attendance to remove outliers
Q1_attendance = df['Attendance (%)'].quantile(0.25)
Q3_attendance = df['Attendance (%)'].quantile(0.75)
IQR_attendance = Q3_attendance - Q1_attendance
lower_bound_attendance = Q1_attendance - 1.5 * IQR_attendance
upper_bound_attendance = Q3_attendance + 1.5 * IQR_attendance

# Remove outliers in 'Attendance (%)'
df = df[(df['Attendance (%)'] >= lower_bound_attendance) & (df['Attendance (%)'] <= upper_bound_attendance)]

# Step 5: Standardize Categorical Columns (e.g., fix typos or formatting inconsistencies)
# Standardize 'Status' column to have consistent format
df['Status'] = df['Status'].str.strip().str.capitalize()

# Check if there are any unexpected values in 'Status'
print("\nUnique values in 'Status' column after standardization:")
print(df['Status'].unique())

# Final Cleaned Data Inspection
print("\nCleaned Data (First 5 rows):")
print(df.head())

# Save the cleaned data to a new CSV file
df.to_csv('clean_data.csv', index=False)
