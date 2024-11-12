
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('diabetes.csv')

print("Initial data inspection:")
print(dataset.head())
print(dataset.info())
print(dataset.describe())


dataset.loc[2, 'Glucose'] = np.nan
dataset.loc[5, 'BloodPressure'] = np.nan

dataset.loc[10, 'BMI'] = -5
dataset.loc[15, 'Age'] = -20

print("\nMissing values per column before cleaning:")
print(dataset.isnull().sum())

dataset.fillna(dataset.median(), inplace=True)

print("\nMissing values after filling:")
print(dataset.isnull().sum())


dataset['BMI'] = dataset['BMI'].apply(lambda x: dataset['BMI'].median() if x < 0 else x)
dataset['Age'] = dataset['Age'].apply(lambda x: dataset['Age'].median() if x < 0 else x)

print("\nData after fixing invalid values:")
print(dataset[['BMI', 'Age']].describe())

for col in dataset.columns:
    if dataset[col].dtype == 'object':
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

print("\nData types after conversion:")
print(dataset.dtypes)

print("\nFinal Data Summary:")
print(dataset.describe())
print("\nSample of cleaned data:")
print(dataset.head())

statistics = {
    'Mean': dataset.mean(),
    'Median': dataset.median(),
    'Mode': dataset.mode().iloc[0],
    'Standard Deviation': dataset.std(),
    'Variance': dataset.var(),
}

print("\nStatistics of dataset:")
print(pd.DataFrame(statistics))

correlation_matrix = dataset.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)


sns.set(style="whitegrid")
dataset.hist(bins=15, figsize=(15, 10), edgecolor='black')
plt.suptitle('Histograms of Different Features', fontsize=16)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
sns.boxplot(data=dataset, width=0.5, palette="Set3")
plt.title('Boxplots of Different Features', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sns.pairplot(dataset, hue='Outcome', palette="coolwarm")
plt.suptitle('Pairplot of Features', fontsize=16)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.show()

sns.pairplot(dataset)
plt.suptitle('Pairwise Scatterplot of Features', fontsize=16)
plt.tight_layout()
plt.show()
