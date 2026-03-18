import json
import pandas as pd

# -----------------------------
# Inverse mappings
# -----------------------------
sleep_inverse = {
    1: "Less than 5 hours",
    2: "5-6 hours",
    3: "7-8 hours",
    4: "More than 8 hours"
}

diet_inverse = {
    1: "Unhealthy",
    2: "Moderate",
    3: "Healthy"
}

gender_inverse = {
    0: "Female",
    1: "Male"
}

yes_no_inverse = {
    0: "No",
    1: "Yes"
}

# -----------------------------
# Input function (same as yours)
# -----------------------------
def take_manual_input():
    print("\nEnter student details:\n")

    data = {
        "gender": int(input("Gender (0 = Female, 1 = Male): ")),
        "age": int(input("Age: ")),
        "academic_pressure": int(input("Academic pressure (1–5): ")),
        "work_pressure": int(input("Work pressure (1–5): ")),
        "study_satisfaction": int(input("Study satisfaction (1–5): ")),
        "job_satisfaction": int(input("Job satisfaction (1–5): ")),
        "work_study_hours": int(input("Work/Study hours per day: ")),
        "financial_stress": int(input("Financial stress (1–5): ")),
        "cgpa": float(input("CGPA: ")),
        "sleep_duration": int(input("Sleep duration (1–4): ")),
        "dietary_habits": int(input("Diet quality (1–3): ")),
        "family_history": int(input("Family history (0 = No, 1 = Yes): ")),
        "suicidal_thoughts": int(input("Suicidal thoughts (0 = No, 1 = Yes): "))
    }

    context = input("\nOptional context (press Enter to skip): ")

    return data, context

# -----------------------------
# Conversion function
# -----------------------------
def convert_to_readable(data, context):

    readable = data.copy()

    readable["gender"] = gender_inverse[data["gender"]]
    readable["sleep_duration"] = sleep_inverse[data["sleep_duration"]]
    readable["dietary_habits"] = diet_inverse[data["dietary_habits"]]
    readable["family_history"] = yes_no_inverse[data["family_history"]]
    readable["suicidal_thoughts"] = yes_no_inverse[data["suicidal_thoughts"]]

    if context:
        readable["context"] = context

    return readable

# -----------------------------
# Save JSON
# -----------------------------
def save_to_json(data):
    with open("external_factors.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print("\n✅ Data saved to external_factors.json")

# -----------------------------
# Main flow
# -----------------------------
if __name__ == "__main__":
    raw_data, context = take_manual_input()
    readable_data = convert_to_readable(raw_data, context)
    save_to_json(readable_data)

    print("\nConverted Data:")
    print(readable_data)