import os
import json
from openai import OpenAI
from transformers import pipeline

# -----------------------------
# Initialization & Models
# -----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Loading NLP Intent Classifier (BART-Large)...")
intent_classifier = pipeline(
    "zero-shot-classification", 
    model="facebook/bart-large-mnli"
)

# -----------------------------
# 1. Contextual Intent Analysis
# -----------------------------
def analyze_intent(text):
    candidate_labels = [
        "venting frustration", 
        "seeking help", 
        "hostility towards assistant", 
        "hopelessness", 
        "situational stress"
    ]
    
    result = intent_classifier(text, candidate_labels)
    return result["labels"][0], result["scores"][0]

# -----------------------------
# 2. Holistic Logic Gate
# -----------------------------
def evaluate_holistic_state(student_data, text_intent, text):
    financial_stress = student_data.get("financial_stress", 1)
    
    risk_assessment = "LOW"
    system_directive = "Standard empathetic listening."

    if text_intent == "venting frustration" and financial_stress >= 4:
        risk_assessment = "LOW-MODERATE (Situational)"
        system_directive = "Validate their frustration. Acknowledge the external pressures (like finances) making things harder. Do NOT treat this as a clinical crisis."
        
    elif text_intent == "hostility towards assistant":
        risk_assessment = "N/A"
        system_directive = "De-escalate. Set gentle boundaries without abandoning the conversation."
        
    elif text_intent == "hopelessness":
        if student_data.get("suicidal_thoughts") == 1:
            risk_assessment = "HIGH (Ideational)"
            system_directive = "Prioritize safety. Ask grounding questions. Gently assess if they have a plan."
        else:
            risk_assessment = "MODERATE"
            system_directive = "Explore the feeling of being stuck. Ask what specifically feels insurmountable right now."

    return risk_assessment, system_directive

# -----------------------------
# 3. Stateful LLM Generation
# -----------------------------
def generate_dynamic_response(user_input, student_data, chat_state):
    
    # 1. Analyze Intent
    intent, confidence = analyze_intent(user_input)
    
    # 2. Pass through Logic Gate
    risk_level, directive = evaluate_holistic_state(student_data, intent, user_input)
    
    # 3. Update Memory
    chat_state["memory"].append({"role": "user", "content": user_input})
    
    # 4. Construct the dynamic system prompt using REAL user data
    data_summary = f"""
    [STATISTICAL DATA]
    - Financial Stress: {student_data.get('financial_stress')}/5
    - Academic Pressure: {student_data.get('academic_pressure')}/5
    - CGPA: {student_data.get('cgpa')}
    - Family History of Mental Health Issues: {"Yes" if student_data.get('family_history') == 1 else "No"}
    """

    system_prompt = f"""
    You are a Technical Mental Health Analyst. 
    The following user has a specific statistical profile:
    {data_summary}

    [EXECUTION DIRECTIVE]
    {directive}

    INSTRUCTIONS:
    1. You MUST acknowledge at least one numerical stressor from the [STATISTICAL DATA].
    2. If the user vents, analyze the 'Friction' between their material metrics (like CGPA) and their internal stress.
    3. Do NOT give a generic "I am sorry" response. 
    4. Use the user's data to explain WHY they might be feeling this way.
    """

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_state["memory"][-10:])

    # Generate Response using gpt-4o-mini
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        messages=messages
    )
    
    bot_reply = response.choices[0].message.content.strip()
    chat_state["memory"].append({"role": "assistant", "content": bot_reply})
    
    return bot_reply

# -----------------------------
# 4. Dynamic Data Collection
# -----------------------------
def collect_student_data():
    print("📋 Let's collect your base parameters for the inference model:\n")
    data = {}
    
    try:
        data["gender"] = int(input("Gender (0 = Female, 1 = Male): "))
        data["age"] = int(input("Age: "))
        data["academic_pressure"] = int(input("Academic pressure (1-5): "))
        data["cgpa"] = float(input("CGPA: "))
        data["financial_stress"] = int(input("Financial stress (1-5): "))
        data["suicidal_thoughts"] = int(input("Suicidal thoughts history (0 = No, 1 = Yes): "))
        data["family_history"] = int(input("Family history of mental illness (0 = No, 1 = Yes): "))
        data["optional_context"] = input("Optional context (press Enter to skip): ")
    except ValueError:
        print("\n⚠️ Invalid input detected. Defaulting to baseline parameters for testing.")
        data = {
            "gender": 1, "age": 21, "academic_pressure": 3, "cgpa": 7.0, 
            "financial_stress": 3, "suicidal_thoughts": 0, "family_history": 0,
            "optional_context": ""
        }
    
    return data

# -----------------------------
# MAIN LOOP
# -----------------------------
if __name__ == "__main__":
    
    # Dynamically grab the data before the chat starts
    student_data = collect_student_data()

    chat_state = {
        "memory": [], 
    }

    print("\n🧠 Integrated Chatbot initialized. (Type 'exit' to quit)\n")

    # If the user provided optional context during setup, we can feed it to the bot immediately
    if student_data.get("optional_context"):
        print(f"You (Context): {student_data['optional_context']}")
        initial_response = generate_dynamic_response(student_data["optional_context"], student_data, chat_state)
        print("\nBot:", initial_response, "\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
            
        if not user_input:
            continue

        response = generate_dynamic_response(user_input, student_data, chat_state)
        print("\nBot:", response, "\n")