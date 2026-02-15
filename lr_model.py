#AI Model Engine
import numpy as np
import pandas as pd
#from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
import os
os.system('cls')
cwd=os.path.dirname(os.path.abspath(__file__))
dataset_path= os.path.join(cwd, "depression_dataset_cleaned.csv")

#pandas dataframe
df = pd.read_csv(dataset_path)

'''df_cleaned = df[df["Sleep Duration"] != 'Others'] 
df_cleaned = df_cleaned[df_cleaned['Dietary Habits'] != 'Others'] 
#'Dietary Habits'
df_cleaned = df_cleaned.drop(columns=["Work Pressure"])
df_cleaned = df_cleaned.drop(columns=["Job Satisfaction"])
print(df.shape,df_cleaned.shape,sep='\n')
cleaned_dataset_path = os.path.join(cwd, "depression_dataset_cleaned.csv")
df_cleaned.to_csv(cleaned_dataset_path, index=False)'''

#convert graded values to numerical scales
sleep_mapping = {
    "Less than 5 hours": 1,   # very poor
    "5-6 hours": 2,           # below ideal
    "7-8 hours": 3,           # optimal
    "More than 8 hours": 2   # can indicate oversleeping
}

diet_mapping = {
    "Unhealthy": 1,
    "Moderate": 2,
    "Healthy": 3
}

yes_no_mapping = {
    "No": 0,
    "Yes": 1
}

df["Sleep Duration"] = df["Sleep Duration"].map(sleep_mapping)
df["Dietary Habits"] = df["Dietary Habits"].map(diet_mapping)
df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map(yes_no_mapping)
df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map(yes_no_mapping)

#print(df.head())