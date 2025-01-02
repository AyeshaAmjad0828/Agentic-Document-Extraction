from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv('.env')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_completion_gpt4(
    messages: list[dict[str, str]],
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


def openai_embedding(text: str) -> list:
    """
    Get embeddings using OpenAI's embedding model
    
    Args:
        text (str): The text to get embeddings for
        
    Returns:
        list: The embedding vector
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding