import os
from fastapi import FastAPI
from pydantic import BaseModel
from merger import generate_dynamic_response

app = FastAPI()

# -------------------------
# Request Schema
# -------------------------
class ChatRequest(BaseModel):
    user_input: str
    student_data: dict = {}
    tas_data: dict = {}
    external_factors: dict = {}   # ✅ NEW
    memory: list = []

# -------------------------
# Health Check
# -------------------------
@app.get("/")
def home():
    return {"status": "API is running"}

# -------------------------
# Chat Endpoint
# -------------------------
@app.post("/chat")
def chat(req: ChatRequest):

    # 🛡️ Safe defaults (prevents crashes)
    student_data = {
        **(req.student_data or {}),
        **(req.external_factors or {})
    }
    tas_data = req.tas_data or {
        "factors": {"DIF": 15, "DDF": 10, "EOT": 15}
    }

    chat_state = {
        "memory": req.memory or []
    }

    try:
        reply = generate_dynamic_response(
            req.user_input,
            student_data,
            chat_state,
            tas_data
        )

        return {
            "response": reply,
            "memory": chat_state["memory"]
        }

    except Exception as e:
        return {
            "error": str(e)
        }