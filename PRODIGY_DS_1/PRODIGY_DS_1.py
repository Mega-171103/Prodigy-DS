import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_file = r"D:\Program File\Prodigy Info Tech\PRODIGY_DS_1\age_gender.csv\age_gender.csv" 
df = pd.read_csv(data_file)

print(df.head(10))

plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(8, 5))
gender_counts = df['gender'].value_counts()
plt.bar(gender_counts.index.astype(str), gender_counts.values, color=['blue', 'pink'])
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])  # Assuming 0 = Male, 1 = Female
plt.show()
