from sentence_transformers import SentenceTransformer 
import numpy as np
from grok import Grok
from dotenv import load_dotenv
import os

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

raw_document = """
Artificial Intenlligence is something everyone keeps talking about that its going to take over the world and peoples jobs in the future

Is heaven and Hell real. Will there be rapture and if it happens what do you think will happen on that day.

Education is the best thing that can happen in human lives
"""

chunks = raw_document.strip().Split('\n\n')

chunks_embeded = model.encode(chunks)

memory_storage = []

for chunks_text, chunks_vector in zip(chunks, chunks_embeded):
    memory_storage.append({
        "text": chunks_text,
        "vector": chunks_vector
    })

user_question = "can ai actually take my job in the nearest"

question_vector = model.encode([user_question][0])

def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1,vec2)

    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)

    return dot_product/(magnitude_vec1 * magnitude_vec2)

highest_score = -1.0
best_match_text =""

for items in memory_storage():
    score = calculate_cosine_similarity(question_vector, items["vector"])

    if score > highest_score:
        highest_score = score
        best_match_text = items["text"]

print(f"Successfully embedd the user queston and heres the highest score {highest_score:.4f}")
print(f"the best match for the user question is {best_match_text}")

client = Grok(api_key=os.getenv("GROK_API_KEY"))

