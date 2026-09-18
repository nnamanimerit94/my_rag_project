from sentence_transformers import SentenceTransformer 
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

raw_document = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans.

Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.
"""


chunks = raw_document.strip().split('\n\n')

chunks_embeded = model.encode(chunks)

memory_storage = []

for chunks_text, chunks_vector in zip(chunks, chunks_embeded):
    memory_storage.append({
        "text": chunks_text,
        "vector": chunks_vector
    })

user_question = "What is the main focus of AI textbooks?"

question_vector = model.encode([user_question])[0]

def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1,vec2)

    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)

    return dot_product/(magnitude_vec1 * magnitude_vec2)

highest_score = -1.0
best_match_text = ""

for items in memory_storage:
    score = calculate_cosine_similarity(question_vector, items["vector"])

    if score > highest_score:
        highest_score = score
        best_match_text = items["text"]

print(f"Successfully embedd the user queston and heres the highest score {highest_score:.4f}")
print(f"the best match for the user question is {best_match_text}")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("\n--- ASKING GROQ (Qwen) ---")

system_prompt = "You are a helpful assistant. Answer the user's question using ONLY the provided context. If the context does not contain the answer, say 'I don't know based on the provided text'."

user_prompt = f"Context: {best_match_text}\n\n Question: {user_question}"

try:
    #calling the groq api 

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages = [
            {"role": "system","content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature = 0.5
    )
    final_answer = response.choices[0].message.content
    print(f"AI Answer: {final_answer}")

except Exception as e:
    print(f"Error calling Groq API: {e}")
    print("Did you forget to replace 'YOUR_GROQ_API_KEY_HERE' with your real key?")
