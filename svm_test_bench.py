import joblib
import re
import string
import os
os.system('cls' if os.name == 'nt' else 'clear')
# -----------------------------
# 1️⃣ Load trained model
# -----------------------------
model = joblib.load("mental_health_svm_model.joblib")


# -----------------------------
# 2️⃣ Text Cleaning Function
# -----------------------------
def clean_text(text: str) -> str:
    
    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# 3️⃣ Prediction Function
# -----------------------------
def predict_suicide_risk(text: str):

    cleaned = clean_text(text)

    prediction = model.predict([cleaned])[0]

    if prediction == 1:
        return {
            "flag": 1,
            "message": "Suicide risk detected"
        }
    else:
        return {
            "flag": 0,
            "message": "No suicide risk detected"
        }


# -----------------------------
# 4️⃣ CLI Testbench
# -----------------------------
if __name__ == "__main__":

    print("\nSuicide Detection Testbench")
    print("Type 'exit' to quit\n")

    while True:

        user_input = input("Enter text: ")

        if user_input.lower() == "exit":
            break

        result = predict_suicide_risk(user_input)

        print("\nPrediction:", result["message"])
        print("Flag:", result["flag"])
        print()