'''import json
import random

suicidal_templates = [
    "I feel hopeless",
    "I feel like giving up",
    "I don't see a future for myself",
    "I feel empty inside",
    "I am tired of everything",
    "I feel like I am a burden",
    "Sometimes I wish I could disappear"
]

nonsuicidal_templates = [
    "I am excited about my future",
    "I love spending time with friends",
    "Today was a productive day",
    "I feel motivated to work",
    "Life has its challenges but I am hopeful",
    "I am learning new things",
    "I enjoy my hobbies"
]

dataset = []

# Generate 250 suicidal
for _ in range(250):
    sentence = random.choice(suicidal_templates)
    dataset.append({"text": sentence, "label": 1})

# Generate 250 non-suicidal
for _ in range(250):
    sentence = random.choice(nonsuicidal_templates)
    dataset.append({"text": sentence, "label": 0})

random.shuffle(dataset)

# Save to JSON
with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

print("Dataset with 500 samples created.")
import json
import random

suicidal = [
    "i don't want to exist anymore",
    "life feels pointless",
    "idk why im still here",
    "wish i could disappear",
    "im done fr",
    "i hate being alive",
    "what's the point of living",
    "im so tired of everything",
    "maybe people are better without me"
]

non_suicidal = [
    "i'm stressed but i'll manage",
    "life is hard but im trying",
    "i feel overwhelmed lately",
    "just tired from work",
    "need a break",
    "not my best day but it's okay",
    "i feel sad but hopeful",
    "im exhausted but motivated"
]

dataset = []

for _ in range(250):
    dataset.append({
        "text": random.choice(suicidal),
        "label": 1
    })

for _ in range(250):
    dataset.append({
        "text": random.choice(non_suicidal),
        "label": 0
    })

random.shuffle(dataset)

with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)'''
'''import json

with open("dataset.json") as f:
    data = json.load(f)

unique = {item["text"]: item for item in data}

cleaned = list(unique.values())

print("Original:", len(data))
print("After removing duplicates:", len(cleaned))

with open("dataset_clean.json","w") as f:
    json.dump(cleaned, f, indent=4)'''
from pathlib import Path
import pandas as pd
import json
BASE_DIR = Path(__file__).resolve().parent

# Build path to CSV
csv_path = BASE_DIR / "Suicide_Detection.csv"
# Load CSV
df = pd.read_csv(csv_path)

# Remove useless column
df = df.drop(columns=["Unnamed: 0"])

# Convert labels
df["label"] = df["class"].map({
    "suicide": 1,
    "non-suicide": 0
})

# Keep only required columns
df = df[["text", "label"]]

# Convert to list of dictionaries
dataset = df.to_dict(orient="records")

# Save JSON
with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("Dataset converted successfully")