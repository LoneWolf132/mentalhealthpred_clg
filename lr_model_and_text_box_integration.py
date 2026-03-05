import os
import joblib
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
os.system('cls')
# -----------------------------
# 1️⃣ Revised Dataset (Balanced)
# -----------------------------
with open("dataset.json", "r") as f:
    dataset = json.load(f)

texts = [item["text"] for item in dataset]
labels = [item["label"] for item in dataset]

'''train_X = [
    # Suicidal (1)
    "i feel like dying",
    "i want to end my life",
    "i am tired of living",
    "goodbye everyone",
    "i cannot do this anymore",
    "i have caused too much pain",
    "there is no reason to live",
    
    # Not Suicidal (0)
    "today was a great day",
    "i love my family",
    "i am excited for my future",
    "life is beautiful",
    "i am working on my goals",
    "i enjoy spending time with friends",
    "i feel motivated today"
]

train_y = [
    1,1,1,1,1,1,1,   # suicidal
    0,0,0,0,0,0,0    # not suicidal
]
'''
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words='english'
)

X_vectors = vectorizer.fit_transform(texts)

model =SVC(kernel='linear',C=0.75, max_iter=6000)
           
'''X_train, X_test, y_train, y_test = train_test_split(
    train_X, train_y, test_size=0.05, random_state=42
)'''

# -----------------------------
# 3️⃣ Vectorization
# -----------------------------

'''vectorizer = CountVectorizer(
    binary=True,
    ngram_range=(1,3),
    stop_words='english'
)
'''
#X_vectors = vectorizer.fit_transform(train_X)

'''X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)'''

# -----------------------------
# 4️⃣ SVM Model
# -----------------------------

'''model = SVC(
kernel='linear',    C=0.75,            # regularization strength
    max_iter=6000
)'''
skf = StratifiedKFold(n_splits=5)

scores = cross_val_score(model, X_vectors, labels, cv=skf)

print("Cross-validation scores:", scores)
print("Average accuracy:", np.mean(scores))

#model.fit(X_train_vectors, y_train)

# -----------------------------
# 5️⃣ Evaluation
# -----------------------------

#predictions = model.predict(X_test_vectors)
#print(classification_report(y_test, predictions))

# -----------------------------
# 6️⃣ Save Model
# -----------------------------

'''joblib.dump(model, "svm_suicide_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")'''

# -----------------------------
# 7️⃣ Prediction Function
# -----------------------------

'''def predict_suicide(text):
    loaded_model = joblib.load("svm_suicide_model.joblib")
    loaded_vectorizer = joblib.load("vectorizer.joblib")
    
    text_vector = loaded_vectorizer.transform([text])
    prediction = loaded_model.predict(text_vector)[0]
    
    return "Suicidal" if prediction == 1 else "Not Suicidal"


# Example
print(predict_suicide("I do not want to live anymore"))
print(predict_suicide("I am planning a vacation"))'''