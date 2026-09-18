import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
db_path ="rag_database.py"

raw_document = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans.

Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.
"""

chunks = raw_document.strip().split('\n\n')

chunks_embeddings = model.encode(chunks)

#connect to sqlite
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

#create the table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    embedding_json TEXT
    )
''')

for chunk_text, chunks_vector in zip(chunks, chunks_embeddings):
    #convert the numpy array to a python list and then to a json string 
    vector_list = chunks_vector.tolist()
    embedding_json = json.dumps(vector_list)
  
    #fill in the colums in the table with the exact values using sql
    cursor.execute(
        "INSERT INTO documents (text, embedding_json) VALUES(?, ?)",
        (chunk_text,embedding_json)
    )
conn.commit()
conn.close()

print("✅ Ingestion complete! Your data is now safely stored in rag_database.db")
