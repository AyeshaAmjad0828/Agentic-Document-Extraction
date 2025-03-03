from openai import OpenAI
from together import Together
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
        
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None
    




client2 = Together(api_key=os.getenv("TOGETHER_API_KEY"))

def get_completion_llama(
    messages: list[dict[str, str]],
    response_format=None,
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",) -> str:
    params = {
        "model": model,
        "messages": messages,
        "response_format": response_format,}
    completion = client2.chat.completions.create(**params)
    return completion

   

    print(completion.choices[0].message.content)