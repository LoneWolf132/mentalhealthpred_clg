import json
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