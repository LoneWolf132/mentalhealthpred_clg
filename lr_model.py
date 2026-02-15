#AI Model Engine
import numpy as np
import pandas as pd
#from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
import os
os.system('cls')
cwd=os.path.dirname(os.path.abspath(__file__))
dataset_path= os.path.join(cwd, "depression_dataset.csv")

#pandas dataframe
df = pd.read_csv(dataset_path)

df_cleaned = df[df["Sleep Duration"] != 'Others'] 
df_cleaned = df_cleaned[df_cleaned['Dietary Habits'] != 'Others'] 
#'Dietary Habits'
df_cleaned = df_cleaned.drop(columns=["Work Pressure"])
df_cleaned = df_cleaned.drop(columns=["Job Satisfaction"])
print(df.shape,df_cleaned.shape,sep='\n')
cleaned_dataset_path = os.path.join(cwd, "depression_dataset_cleaned.csv")
df_cleaned.to_csv(cleaned_dataset_path, index=False)