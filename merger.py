from openai import OpenAI
from input_table_to_json import take_manual_input, convert_to_readable, get_depression_score
from svm_test_bench import predict_suicide_risk
import os
def build_ml_output(depression_prob, suicide_result):

    return {
        "depression_probability": float(depression_prob),
        "suicide_probability": float(suicide_result["probability"]),
        "suicide_risk_level": suicide_result["risk_level"]
    }
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_analysis(data, context, ml_outputs):

    prompt = f"""
You are a mental health analysis assistant.

Input:
Structured Data: {data}
User Context: {context}
ML Outputs: {ml_outputs}

Generate output STRICTLY in this format:

0. Ethical Disclaimer
1. Depression Probability & Severity
2. Suicide Risk Analysis
3. Contributing Factors
4. Emotional Support
5. Short-term Suggestions
6. Professional Help Recommendation

Rules:
- Do NOT diagnose
- Use probabilities as supporting signals
- Be empathetic but not dramatic
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

if __name__ == "__main__":

    raw_data, context = take_manual_input()

    readable_data = convert_to_readable(raw_data, context)

    depression_prob, _ = get_depression_score(raw_data)

    suicide_result = predict_suicide_risk(context)

    ml_outputs = build_ml_output(depression_prob, suicide_result)

    final_report = generate_analysis(readable_data, context, ml_outputs)

    print("\n🧠 FINAL REPORT:\n")
    print(final_report)