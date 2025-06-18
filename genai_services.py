import os
from typing import List
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI

# Load environment variables
load_dotenv(dotenv_path="E:/j/template.env")

# Get environment variables
api_key = os.getenv("MODEL_API_KEY")
base_url = os.getenv("MODEL_BASE_URL")
model_name = os.getenv("MODEL_NAME")

# Validate .env configuration
if not api_key:
    raise EnvironmentError("❌ MODEL_API_KEY is missing in your .env file.")
if not base_url:
    raise EnvironmentError("❌ MODEL_BASE_URL is missing in your .env file.")
if not model_name:
    raise EnvironmentError("❌ MODEL_NAME is missing in your .env file.")

# Initialize Gemini-compatible OpenAI client
openai_client = OpenAI(api_key=api_key, base_url=base_url)


def call_llm(messages: List[dict]) -> str:
    """Helper function to call Gemini API using OpenAI-style chat interface"""
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    return response.choices[0].message.content


def summarize_text(text: str) -> str:
    """Generate a summary of the text using Gemini API"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes documents accurately and concisely."
        },
        {
            "role": "user",
            "content": f"Please summarize the following text concisely while capturing the key points:\n\n{text}"
        }
    ]
    return call_llm(messages)


def chunk_text(text: str, chunk_size: int = 100, chunk_overlap: int = 10) -> List[str]:
    """Split text into overlapping chunks of specified size"""
    if not text:
        return []

    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)

    chunks = []
    i = 0
    while i < len(tokens):
        chunk_end = min(i + chunk_size, len(tokens))
        chunks.append(enc.decode(tokens[i:chunk_end]))
        i = chunk_end - chunk_overlap if chunk_end < len(tokens) else chunk_end

    return chunks


def answer_with_context(question: str, contexts: List[str]) -> str:
    """Generate a response to a query using context from RAG"""
    combined_context = "\n\n---\n\n".join(contexts)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on the provided context. If you don't know the answer based on the context, say so."
        },
        {
            "role": "user",
            "content": f"Context information:\n\n{combined_context}\n\nQuestion: {question}\n\nAnswer:"
        }
    ]
    return call_llm(messages)


# Optional test block
if __name__ == "__main__":
    try:
        test = call_llm([{"role": "user", "content": "Hello, who are you?"}])
        print("✅ Test Response:", test)
    except Exception as e:
        print("❌ Test Failed:", str(e))
