import pandas as pd
from predict import predict_depression
from interpretation import (
    generate_interpretation,
    decide_action,
    escalate_severity
)


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

    return pd.DataFrame([data]), context


def main():
    user_df, context = take_manual_input()

    # 1️⃣ ML risk analysis
    ml_result = predict_depression(user_df)

    # 2️⃣ Safety / diagnostic escalation
    final_severity = escalate_severity(
        ml_severity=ml_result["ml_severity"],
        probability=ml_result["probability"],
        suicidal_thoughts=user_df.iloc[0]["suicidal_thoughts"],
        context=context
    )

    # 3️⃣ Interpretation & action
    interpretation = generate_interpretation(
        final_severity,
        ml_result["probability"],
        context
    )

    action = decide_action(final_severity)

    print("\n📊 DEPRESSION ASSESSMENT RESULT")
    print("--------------------------------")
    print(f"ML Severity     : {ml_result['ml_severity']}")
    print(f"Final Severity  : {final_severity}")
    print(f"Probability     : {ml_result['probability']:.2f}")

    print("\n🧠 Interpretation:")
    print(interpretation)

    print("\n🚦 Recommended Action:")
    print(action)


if __name__ == "__main__":
    main()