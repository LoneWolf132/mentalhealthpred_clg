import json
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_validate
import joblib
import os

os.system('cls' if os.name == 'nt' else 'clear')

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
log("Loading dataset...")

with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

texts = [item["text"] for item in dataset]
labels = [item["label"] for item in dataset]

log(f"Loaded {len(texts)} samples")

# -----------------------------
# 2️⃣ Build Pipeline
# -----------------------------
log("Building pipeline...")

pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(
        ngram_range=(1,3),
        stop_words="english",
        min_df=5,
        max_df=0.9,
        sublinear_tf=True
    )),
    ("svm", LinearSVC(
        C=1.0,
        max_iter=10000
    ))
])

# -----------------------------
# 3️⃣ Cross Validation
# -----------------------------
log("Starting cross-validation (5 folds)...")

scores = cross_validate(
    pipeline,
    texts,
    labels,
    cv=5,
    scoring=["accuracy", "recall", "f1"],
    n_jobs=-1,
    verbose=2
)

log("Cross-validation complete")

print("\nResults:")
print("Accuracy:", np.mean(scores["test_accuracy"]))
print("Recall:", np.mean(scores["test_recall"]))
print("F1:", np.mean(scores["test_f1"]))

# -----------------------------
# 4️⃣ Train Final Model
# -----------------------------
log("Training final model on full dataset...")
pipeline.fit(texts, labels)
log("Training complete")

# -----------------------------
# 5️⃣ Save Model
# -----------------------------
log("Saving trained model...")

joblib.dump(pipeline, "mental_health_svm_model.joblib")

log("Model saved successfully")

# -----------------------------
# 6️⃣ Test Prediction
# -----------------------------
text = "my foot!"
prediction = pipeline.predict([text])

print("\nTest prediction:", prediction) 