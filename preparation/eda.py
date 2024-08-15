import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/StudentsPerformance.csv")
print(df.head())
print(df.shape)

#check for missing values
print(df.isna().sum())

#check for duplicates
print(df.duplicated().sum())

#check null and Dtypes
print(df.info())

#how many unique values for each column
print(df.nunique())

#overall information
print(df.describe())


#looking for unique values in independent columns
print(df['gender'].unique())
print(df['race_ethnicity'].unique())
print(df['parental level of education'].unique())
print(df['lunch'].unique())
print(df['test preparation course'].unique())


#separating numeric and categorical features
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']


#adding columns for total score and average score
df['total score'] = df['math score'] + df['reading_score'] + df['writing_score']
df['average score'] = df['total score']/3



