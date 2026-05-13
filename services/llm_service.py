from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_TEMP"))

def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)

    prompt = f"""
    You are a helpful assistant.

    Use the context below to answer the question clearly.
    If the answer is partially available, try to summarize it. If answer not in context, say you don't know.

    Context:
    {context}

    Question:
    {query}

    Answer in a clear and complete sentence.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content