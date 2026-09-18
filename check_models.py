from groq import Groq
import os
from dotenv import load_dotenv

# Load the key from your .env file
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Ask Groq for the list of active models
models = client.models.list()

print("--- CURRENTLY ACTIVE GROQ MODELS ---")
for model in models.data:
    print(model.id)