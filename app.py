from retriever import hybrid_search
from groq import Groq
import os

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(query):
    vector_ids, keyword_docs = hybrid_search(query)

    # Build context
    context = "\n".join(keyword_docs[:3])

    prompt = f"""
You are an AI assistant. Answer ONLY using the context below.

Context:
{context}

Question: {query}

If answer is not in context, say "Not found in context".
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # 🔥 Fast + good
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")
        print("\nAnswer:", generate_answer(q))