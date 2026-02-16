#AI Model Engine
import numpy as np
import pandas as pd
#from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
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

gender_mapping = {
    "Male": 1,
    "Female": 0
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
del_cols=["id","City","Profession","Degree"]  #don't impact
df["Sleep Duration"] = df["Sleep Duration"].map(sleep_mapping)
df["Dietary Habits"] = df["Dietary Habits"].map(diet_mapping)
df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map(yes_no_mapping)
df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map(yes_no_mapping)
df['Gender']=df['Gender'].map(gender_mapping)
df.drop(columns=del_cols,inplace=True) #insignificant and redundant columns removed

#print(df.shape)
#print(df.info())
new_cols=["gender", "age", "academic_pressure","cgpa", "study_satisfaction", "sleep_duration","dietary_habits","suicidal_thoughts",
"work_study_hours","financial_stress","family_history","depression"]
df.columns=new_cols
#print(df.info())

#dependent and independent  feature sets 
X=df.drop(columns='depression').copy()
y=df['depression'].copy() 

#train-test split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=42) 

#logistic regression obj
log_regression=LogisticRegression()
print(X_train)
log_regression.fit(X_train, y_train)