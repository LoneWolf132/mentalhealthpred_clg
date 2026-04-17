import os
import json
from openai import OpenAI
from transformers import pipeline
os.system("cls" if os.name == "nt" else "clear")
# -----------------------------
# Initialization & Models
# -----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# -----------------------------
# Dictionary: Digital Somatic Markers & Slang
# -----------------------------
alpha_slang_map = {
    # Gen Z / Gen Alpha Slang
    "cooked": "hopeless and defeated",
    "mid": "mediocre and stressful",
    "tweaking": "frustrated and anxious",
    "delulu": "disconnected from reality",
    "ghosted": "abandoned and lonely",
    "skibidi": "chaotic and overstimulated",
    
    # Emojis (Digital Somatic Markers)
    "💀": "hopeless, completely exhausted",
    "🫠": "melting under situational stress, overwhelmed",
    "🤡": "frustrated at myself, feeling foolish",
    "😭": "venting extreme emotion",
    "🙃": "masking frustration with a smile",
    "🙏": "seeking help desperately"
}
print("Loading NLP Intent Classifier (BART-Large)...")
intent_classifier = pipeline(
    "zero-shot-classification", 
    model="facebook/bart-large-mnli"
)

# -----------------------------
# 1. Contextual Intent Analysis
# -----------------------------
def collect_tas20_data(): #toronto alexythymia scale
    print("--- Emotional Expression Survey (TAS-20) ---")
    print("Please answer 1-5 (1: Strongly Disagree, 5: Strongly Agree)\n")
    
    questions = [
        "I am often confused about what emotion I am feeling.",
        "It is difficult for me to find the right words for my feelings.",
        "I have physical sensations that even doctors don\'t understand.",
        "I am able to describe my feelings easily.", # Reverse
        "I prefer to analyze problems rather than just describe them.", # Reverse
        "When I am upset, I don\'t know if I am sad, frightened, or angry.",
        "I am often puzzled by sensations in my body.",
        "I prefer to just let things happen rather than to understand why.",
        "I have feelings that I can\'t quite identify.",
        "Being in touch with emotions is essential.", # Reverse
        "I find it hard to describe how I feel about people.",
        "People tell me to describe my feelings more.",
        "I don\'t know what\'s going on inside me.",
        "I often don\'t know why I am angry.",
        "I prefer talking to people about daily activities rather than feelings.",
        "I prefer to watch 'light' entertainment rather than psychological dramas.",
        "It is difficult for me to reveal my innermost feelings to friends.",
        "I can feel close to someone, even in moments of silence.", # Reverse
        "I find examination of my feelings useful in solving problems.", # Reverse
        "Looking for hidden meanings in movies distracts from my enjoyment."
    ]

    raw_responses = []
    for i, q in enumerate(questions):
        val = int(input(f"{i+1}. {q}: "))
        raw_responses.append(val)

    # Indices that need flipping (4, 5, 10, 18, 19)
    reverse_indices = {4, 5, 10, 18, 19}
    
    # Processed scores based on your image values
    processed = []
    for i, val in enumerate(raw_responses, 1):
        if i in reverse_indices:
            processed.append(6 - val)
        else:
            processed.append(val)

    # Structure into the dictionary for the LLM
    tas_data = {
        "total_score": sum(processed),
        "factors": {
            "DIF": sum([processed[i-1] for i in [1, 3, 6, 7, 9, 13, 14]]), # Difficulty Identifying
            "DDF": sum([processed[i-1] for i in [2, 4, 11, 12, 17]]),     # Difficulty Describing
            "EOT": sum([processed[i-1] for i in [5, 8, 10, 15, 16, 18, 19, 20]]) # External Thinking
        }
    }
    return tas_data

def analyze_intent(text, age):
    processed_text = text.lower()
    
    # Only apply the Slang Map for Gen Z and Gen Alpha (Age <= 26)
    if age <= 26:
        for slang, translation in alpha_slang_map.items():
            if slang in processed_text:
                # Inject the meaning into the string for BART to read
                processed_text = processed_text.replace(slang, f"{slang} [meaning: {translation}]")
                
    candidate_labels = [
        "venting frustration", 
        "seeking help", 
        "hostility towards assistant", 
        "hopelessness", 
        "situational stress"
    ]
    
    result = intent_classifier(processed_text, candidate_labels)
    return result["labels"][0], result["scores"][0]

# -----------------------------
# 2. Holistic Logic Gate
# -----------------------------
def evaluate_holistic_state(student_data, text_intent, text, tas_data=None):
    # 1. Safety Guard for Missing Psychometric Data
    if not tas_data:
        return "LOW", "Standard empathetic listening. (TAS data missing)"

    # 2. Extract Base Metrics
    financial_stress = student_data.get("financial_stress", 1)
    age = student_data.get("age", 21)
    dif_score = tas_data['factors']['DIF']
    
    # -----------------------------
    # BRANCH A: GEN ALPHA (Developmental Focus)
    # -----------------------------
    if age <= 13:
        if dif_score > 22:
            return "MODERATE (Sensory Overload)", (
                "User is Gen Alpha. Use 'Gamified Inferences'. "
                "Ask about somatic sensations using metaphors (e.g., 'too many tabs open')."
            )
        return "LOW", "Use simplified, visual-heavy language."

    # -----------------------------
    # BRANCH B: GEN Z / MILLENNIAL (Material Friction Focus)
    # -----------------------------
    # Priority 1: Alexithymia Check (Applied to ALL intense intents)
    if dif_score > 20 and text_intent in ["seeking help", "hopelessness", "venting frustration"]:
        return "MODERATE (Masked Distress)", (
            "User has high DIF. Do NOT ask 'How do you feel?'. "
            "Instead, ask about sleep, appetite, or physical tension to deduce stress."
        )

    # Priority 2: Situational Intent Logic
    if text_intent == "venting frustration" and financial_stress >= 4:
        return "LOW-MODERATE (Situational)", "Validate frustration. Acknowledge material/financial friction."
        
    elif text_intent == "hostility towards assistant":
        return "N/A", "De-escalate. Set gentle boundaries."
        
    elif text_intent == "hopelessness":
        if student_data.get("suicidal_thoughts") == 1:
            return "HIGH (Ideational)", "Prioritize safety. Ask grounding questions. Assess for a plan."
        return "MODERATE", "Explore the feeling of being stuck. Ask what feels insurmountable."

    return "LOW", "Standard empathetic listening."

# -----------------------------
# 3. Stateful LLM Generation
# -----------------------------
# -----------------------------
# 3. Stateful LLM Generation (UPDATED)
# -----------------------------
def generate_dynamic_response(user_input, student_data, chat_state, tas_data):
    
    # Fetch age to pass to the intent analyzer
    age = student_data.get("age", 21)
    
    # 1. Analyze Intent (Now Slang/Emoji-Aware)
    intent, confidence = analyze_intent(user_input, age)
    
    # 2. Pass through Logic Gate
    risk_level, directive = evaluate_holistic_state(student_data, intent, user_input, tas_data)
    
    # 3. Update Memory
    chat_state["memory"].append({"role": "user", "content": user_input})
    
    # Base Data Summary for the LLM
    data_summary = f"""
    [STATISTICAL DATA]
    - Financial Stress: {student_data.get('financial_stress')}/5
    - Academic Pressure: {student_data.get('academic_pressure')}/5
    - CGPA: {student_data.get('cgpa')}
    - Alexithymia Profile: {tas_data['factors']} (DIF: Identify, DDF: Describe, EOT: External focus)
    """

    # 4. Construct the dynamic system prompt (PERSONA BRANCHING)
    if age <= 13:
        # Gen Alpha Persona: Short, visual, system-focused.
        system_prompt = f"""
        You are a "System Debugger" helping a Gen Alpha user navigate stress. 
        
        {data_summary}

        [EXECUTION DIRECTIVE]
        {directive}

        [STRICT INSTRUCTIONS]
        1. TL;DR FORMAT ONLY: Maximum 3 short sentences. 
        2. NO ACADEMIC LANGUAGE: Do not use words like 'cognitive dissonance', 'trajectory', or 'friction'. 
        3. TONE: Casual, slightly gamified. Treat their stress like a laggy server or a low battery.
        4. GET TO THE POINT: Execute the directive immediately without introductory filler.
        """
    else:
        # Gen Z / Millennial Persona: Analytical, deep, logical.
        system_prompt = f"""
        You are a Technical Mental Health Analyst. Analyze the user's state using 'Material-Psychometric Incongruence'.

        {data_summary}

        [EXECUTION DIRECTIVE]
        {directive}

        [INSTRUCTIONS]
        1. DIAGNOSTIC FRICTION: Analyze the friction between their material metrics (like CGPA) and their internal stress.
        2. PSYCHE INTERPRETATION: Speak to them as an intellectual equal. If EOT is low, use deep, philosophical language. If DDF is high, offer them vocabulary to describe their state.
        3. NO 'I AM SORRY': Start directly with an analysis of their data. Do not pity them.
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
    student_tas20 = collect_tas20_data()

    chat_state = {
        "memory": [], 
    }

    print("\n🧠 Integrated Chatbot initialized. (Type 'exit' to quit)\n")

    # If the user provided optional context during setup, we can feed it to the bot immediately
    if student_data.get("optional_context"):
        print(f"You (Context): {student_data['optional_context']}")
        initial_response = generate_dynamic_response(student_data["optional_context"], student_data, chat_state, student_tas20)
        print("\nBot:", initial_response, "\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
            
        if not user_input:
            continue

        response = generate_dynamic_response(user_input, student_data, chat_state, student_tas20)
        print("\nBot:", response, "\n")