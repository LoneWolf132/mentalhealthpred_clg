#integrate model with additional context
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
os.system('cls')
#importing model into the framework
cwd=os.path.dirname(os.path.abspath(__file__))
model_path= os.path.join(cwd, 'logistic_depression_model_revised.joblib')
model = joblib.load(model_path) #let's leave it aside for time being, let's focus on nlp for now

#tw identification - vectorization
class trig_word:
    FLD={'feel like dying', 'done with life', 'don\'t wanna live','suicide','kill myself'}
    DEATH={'death','goodbye','bye'}
    REGRET={'sorry','i\'ve caused so much pain'}

train_X=["i feel like dying",'I am fed up with life', "i\'ve been a burden to everyone",'bye, maybe we shall meeet again in the afterlife']
train_y=[trig_word.FLD,trig_word.FLD,trig_word.REGRET,trig_word.DEATH]
