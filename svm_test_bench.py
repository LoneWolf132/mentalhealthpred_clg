import joblib
import re
import string
import os

os.system('cls' if os.name == 'nt' else 'clear')

# -----------------------------
# 1️⃣ Load trained model
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
joblib_path = os.path.join(current_dir, "mental_health_svm_model_LinearSVC_Calibrated.joblib")

model = joblib.load(joblib_path)

print("✅ Model loaded successfully\n")

# -----------------------------
# 2️⃣ Text Cleaning Function
# -----------------------------
def clean_text(text: str) -> str:

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# 3️⃣ Prediction Function
# -----------------------------
def predict_suicide_risk(text: str):

    cleaned = clean_text(text)

    prediction = model.predict([cleaned])[0]
    probability = model.predict_proba([cleaned])[0][1]

    # Risk tier (useful for LLM later)
    if probability < 0.3:
        risk_level = "LOW"
    elif probability < 0.7:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "flag": int(prediction),
        "probability": float(probability),
        "risk_level": risk_level,
        "message": "Suicide risk detected" if prediction == 1 else "No suicide risk detected"
    }


# -----------------------------
# 4️⃣ CLI Testbench
# -----------------------------
if __name__ == "__main__":

    print("🧠 Suicide Detection Testbench")
    print("Type 'exit' to quit\n")

    while True:

        user_input = input("Enter text: ")

        if user_input.lower() == "exit":
            break

        result = predict_suicide_risk(user_input)

        print("\n--- RESULT ---")
        print(f"Prediction   : {result['message']}")
        print(f"Probability  : {result['probability']:.4f}")
        print(f"Risk Level   : {result['risk_level']}")
        print(f"Flag         : {result['flag']}")
        print("----------------\n")