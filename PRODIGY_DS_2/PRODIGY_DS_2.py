import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_file = r"D:\Program File\Prodigy Info Tech\PRODIGY_DS_2\train.csv"  
df = pd.read_csv(data_file)



print(df.info())

print("\nMissing Values:\n", df.isnull().sum())


df['Age'].fillna(df['Age'].median(), inplace=True) 
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  
df.drop(columns=['Cabin'], inplace=True)  

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

print("\nSummary Statistics:\n", df.describe())

plt.figure(figsize=(8,5))
sns.countplot(x='Survived', data=df, palette='coolwarm')
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True, color='blue')
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Pclass', y='Age', data=df, palette='Set2')
plt.title("Age Distribution Across Passenger Classes")
plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
