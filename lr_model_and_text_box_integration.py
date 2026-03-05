import json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------

with open("dataset.json", "r") as f:
    dataset = json.load(f)

texts = [item["text"] for item in dataset]
labels = [item["label"] for item in dataset]

# -----------------------------
# 2️⃣ Build Pipeline
# -----------------------------

pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("svm", SVC(
        kernel="linear",
        C=0.75,
        max_iter=6000
    ))
])

# -----------------------------
# 3️⃣ Cross Validation
# -----------------------------

scores = cross_val_score(pipeline, texts, labels, cv=5)

print("Cross-validation scores:", scores)
print("Average accuracy:", np.mean(scores))

# -----------------------------
# 4️⃣ Train Final Model
# -----------------------------

pipeline.fit(texts, labels)

# -----------------------------
# 5️⃣ Test Prediction
# -----------------------------

text = "my foot!"
prediction = pipeline.predict([text])

print("Prediction:", prediction)