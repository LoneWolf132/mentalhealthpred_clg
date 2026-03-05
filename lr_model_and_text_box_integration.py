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
# 1️⃣ Load Dataset
# -----------------------------

with open("dataset.json", "r") as f:
    dataset = json.load(f)

texts = [item["text"] for item in dataset]
labels = [item["label"] for item in dataset]

# -----------------------------
# 2️⃣ Vectorization
# -----------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words='english'
)

X_vectors = vectorizer.fit_transform(texts)

# -----------------------------
# 3️⃣ SVM Model
# -----------------------------

model = SVC(
    kernel='linear',
    C=0.75,
    max_iter=6000
)

# -----------------------------
# 4️⃣ Cross Validation
# -----------------------------

skf = StratifiedKFold(n_splits=5)

scores = cross_val_score(model, X_vectors, labels, cv=skf)
print(model.fit(X_vectors, labels))
print("Cross-validation scores:", scores)
print("Average accuracy:", np.mean(scores))
text = "my foot!"

text_vector = vectorizer.transform([text])   # NOTE: list wrapper []
prediction = model.predict(text_vector)

print(prediction)