import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('Titanic-Dataset.csv')
print("\n DataTypes")
print(df.dtypes)
print("Missing values per column:\n", df.isnull().sum())
print("\nHandle Missing Values using mean/median/mode:\n")
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode(), inplace=True)
df.drop(columns=['Cabin'], inplace=True)

print(df.isnull().sum())
print("\n Convert categorical features to numerical features:\n")
le=LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
print(df.head())
print("\nNormalize the numerical features:\n")
scaler = StandardScaler()
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(df[numeric_cols].head())
print("\nVisualize Outliers with Boxplots")
plt.figure(figsize = (12,8))
for i,col in enumerate(numeric_cols, 1):
    plt.subplot(2,2,i)
    sns.boxenplot(data=df, x=col)
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

print("\nRemoving Outliers")
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    iqr = Q3 - Q1
    lower_bound = Q1 - 1.5 * iqr
    upper_bound = Q3 + 1.5 * iqr
    return data[(data[column] >= lower_bound)& (data[column] <= upper_bound)]
for col in numeric_cols:
    df=remove_outliers_iqr(df, col)
print("Remaining rows after outlier removal:", df.shape[0])
