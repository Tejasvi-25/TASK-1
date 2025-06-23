🧹 Data Cleaning & Preprocessing Project
📌 Objective
To learn how to clean and prepare raw data for machine learning using Python. The dataset used is the Titanic Dataset, but the process can be applied to any structured dataset.

🛠 Tools Used
Python

Pandas – for data manipulation

NumPy – for numerical operations

Matplotlib / Seaborn – for data visualization

scikit-learn – for normalization and preprocessing

🔄 Data Cleaning & Preprocessing Steps
1. 📥 Import the Dataset & Explore Basic Info
Load the dataset using pandas.read_csv().

Use .info(), .describe(), and .head() to understand the structure.

Check for:

Number of nulls

Data types

General distribution of features

2. 🧩 Handle Missing Values
Identify missing values with .isnull().sum().

Strategies:

Fill numerical features with mean or median.

Fill categorical features with mode.

Drop columns with too many missing values if appropriate (e.g., Cabin in Titanic dataset).

3. 🔢 Convert Categorical Features into Numerical
Use one-hot encoding for categorical variables (pd.get_dummies()).

Use label encoding for binary variables (optional).

4. 📊 Normalize/Standardize Numerical Features
Normalize or standardize features like Age and Fare using StandardScaler from sklearn.preprocessing.

This helps improve the performance of many ML algorithms.

5. 🧪 Visualize Outliers with Boxplots and Remove Them
Use seaborn.boxplot() to visualize outliers in numerical columns.

Use the IQR method (Interquartile Range) to detect and remove outliers.

📁 Files Included
data_cleaning.py – Python script with all preprocessing steps

titanic.csv – Sample dataset (downloaded from Kaggle or direct CSV link)

✅ Outcome
A cleaned dataset ready for:

Exploratory Data Analysis (EDA)

Machine Learning model training and evaluation
