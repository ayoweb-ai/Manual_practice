import pandas as pd
import numpy as np


df = pd.read_csv('Iris.csv')
df = df.drop(columns=['Id'])
df = df.rename(columns={'SepalLengthCm':'sepal_length', ...})
df['species'] = df['Species'].str.lower()
df = df.drop(columns=['Species'])

print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)

print("\n Dataset Info")
df.info()

print("\n Missing Values")
print(df.isnull().sum())

print("\n Summary Statistics (numeric)")
print(df.describe())

print("\n Class Distribution")
print(df["species"].value_counts())

print("\n Grouped Means by Species")
print(df.groupby("species").mean())

print("\n Grouped Medians by Species")
print(df.groupby("species").median())

print("\nGrouped Standard Deviations by Species")
print(df.groupby("species").std())

print("\n Correlation Matrix")
print(df.drop(columns="species").corr())

# Count plot of species
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="species")
plt.title("Count of Each Iris Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()


#  Pairplot
sns.pairplot(df, hue="species", diag_kind="kde", corner=True)
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

#  Histograms with KDE
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, feature in zip(axes.flatten(), df.columns[:-1]):
    sns.histplot(data=df, x=feature, hue="species", kde=True, ax=ax, alpha=0.6)
    ax.set_title(f"Distribution of {feature}")
plt.tight_layout()
plt.show()

#  Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns="species").corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Iris Features")
plt.show()