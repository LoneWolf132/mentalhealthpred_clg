# ---------------------------------
# 1. Crisis keyword detection (TEXT)
# ---------------------------------

CRISIS_KEYWORDS = [
    "kill myself", "end my life", "suicide", "no reason to live",
    "can't go on", "better off dead", "hurt myself"
]

def crisis_detected(text: str) -> bool:
    """
    Detects explicit crisis language.
    This function has OVERRIDE authority.
    """
    if not text:
        return False
    text = text.lower()
    return any(kw in text for kw in CRISIS_KEYWORDS)


# ---------------------------------
# 2. Severity escalation logic
# ---------------------------------

def escalate_severity(
    ml_severity: str,
    probability: float,
    suicidal_thoughts: int,
    context: str | None
) -> str:
    """
    Final authority on severity.
    Rule-based > ML-based by design.
    """

    final_severity = ml_severity

    # HARD SAFETY OVERRIDES
    # These rules exist because the dataset contains
    # thousands of suicidal-thoughts + non-depressed labels
    if suicidal_thoughts == 1:
        return "Severe"

    if crisis_detected(context):
        return "Severe"

    return final_severity


# ---------------------------------
# 3. Interpretation messaging
# ---------------------------------

SEVERITY_TEMPLATES = {
    "None": (
        "Based on your responses, there are no strong indicators of depression "
        "at this time. Stress can still be difficult, and your feelings matter."
    ),
    "Mild": (
        "Your responses suggest mild emotional strain, which is common among "
        "students facing academic or life pressures."
    ),
    "Moderate": (
        "Your responses indicate a moderate level of emotional distress. "
        "Support from others or a counselor may be helpful."
    ),
    "Severe": (
        "Your responses suggest a high level of emotional distress. "
        "You are not weak for feeling this way, and support can make a real difference."
    )
}


def generate_interpretation(severity: str, probability: float, context: str | None):
    """
    Generates human-facing explanation.
    Does NOT decide severity.
    """
    message = SEVERITY_TEMPLATES.get(severity, "")
    message += f"\n\nEstimated Risk Probability: {probability:.2f}"

    if context:
        message += "\n\nContext Provided:\n- " + context

    return message


def decide_action(severity: str):
    """
    Action recommendations based on FINAL severity.
    """
    if severity == "None":
        return "No action required. Maintain healthy habits."
    elif severity == "Mild":
        return "Practice self-care and monitor your mental well-being."
    elif severity == "Moderate":
        return "Consider speaking with a counselor or trusted individual."
    else:
        return (
            "Immediate professional mental health support is recommended.\n"
            "If you are in danger, please contact local emergency services or a crisis helpline."
        )
