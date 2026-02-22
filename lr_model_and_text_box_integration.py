#integrate model with additional context
import os
import joblib
import numpy as np
import sys

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

train_X=["i feel like dying",'I am fed up with life', "i\'ve been a burden to everyone",'bye, maybe we shall meet again in the afterlife']
train_y=[trig_word.FLD,trig_word.FLD,trig_word.REGRET,trig_word.DEATH]

vectorizer_object=CountVectorizer(binary=True,ngram_range=(1,3))

train_x_vectors = vectorizer_object.fit_transform(train_X)

print(vectorizer_object.get_feature_names_out())

print(train_x_vectors.toarray())

#print(sys.executable)