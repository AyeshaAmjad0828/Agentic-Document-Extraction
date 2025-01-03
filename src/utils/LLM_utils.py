from openai import OpenAI
import os
import json
from dotenv import load_dotenv


load_dotenv('.env')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_completion_gpt4(
    messages: list[dict[str, str]],
    response_format=None,
    model: str = "gpt-4o-mini",
    max_tokens=16000,
    temperature=0.1,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
    top_logprobs=None,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion


def openai_embedding(text):
    """Get OpenAI embedding for text"""
    if not text or not isinstance(text, str):
        print("Warning: Empty or invalid text provided for embedding")
        return None
        
    # Convert to string if it's a dictionary/list
    if isinstance(text, (dict, list)):
        text = json.dumps(text)
        
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None