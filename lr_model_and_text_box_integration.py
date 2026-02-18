#integrate model with additional context
import os
import joblib
import numpy as np
os.system('cls')

cwd=os.path.dirname(os.path.abspath(__file__))
model_path= os.path.join(cwd, 'logistic_depression_model_revised.joblib')
model = joblib.load(model_path)