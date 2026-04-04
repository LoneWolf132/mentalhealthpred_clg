import json
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_validate
import joblib
import os

os.system('cls' if os.name == 'nt' else 'clear')
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "dataset.json")

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
log("Loading dataset...")

with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

texts = [item["text"] for item in dataset]
labels = [item["label"] for item in dataset]

log(f"Loaded {len(texts)} samples")

# -----------------------------
# 2️⃣ Build Pipeline (Optimized)
# -----------------------------
log("Building pipeline...")

# The Fix: Calibrated LinearSVC gives probabilities at O(n) linear speed
svm_calibrated = CalibratedClassifierCV(
    estimator=LinearSVC(C=1.0), 
    method='sigmoid', 
    cv=3 # 3-fold internal CV for probability calibration is plenty
)

pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(
        ngram_range=(1,2), # Reduced to bigrams to save memory and time
        stop_words="english",
        min_df=5,
        max_df=0.9,
        sublinear_tf=True
    )),
    ("svm", svm_calibrated)
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

joblib.dump(pipeline, "mental_health_svm_model_LinearSVC_Calibrated.joblib")

log("Model saved successfully")

# -----------------------------
# 6️⃣ Test Prediction (With Probabilities)
# -----------------------------
text = "my foot!"

# Get the standard class prediction
prediction = pipeline.predict([text])

# Get the probability percentages for each class
probabilities = pipeline.predict_proba([text])

print("\nTest prediction:", prediction[0])
print("Probabilities:", probabilities[0])