from openai import OpenAI
from together import Together
import os
import json
from dotenv import load_dotenv
import numpy as np

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
    temperature=0.1,
    #max_tokens = 8000,
    logprobs=None,
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",) -> str:
    params = {
        "messages": messages,
        "response_format": response_format,
        "temperature": temperature,
        #"max_tokens": max_tokens,
        "logprobs": logprobs,
        "model": model,}
    completion = client2.chat.completions.create(**params)
    return completion



def get_llm_completion(
    messages: list[dict[str, str]],
    llm_choice: str = "gpt",  # "gpt" or "llama"
    response_format=None,
    temperature=0.1,
    model: str = None,
    max_tokens=16000,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,
    top_logprobs=None,
) -> str:
    """
    Select and use appropriate LLM based on user choice
    Args:
        messages: List of message dictionaries
        llm_choice: Either "gpt" or "llama"
        ... other parameters passed to respective LLM functions ...
    Returns:
        LLM completion response
    """
    if llm_choice.lower() == "gpt":
        # Use default GPT model if none specified
        gpt_model = model if model else "gpt-4o-mini"
        return get_completion_gpt4(
            messages=messages,
            response_format=response_format,
            model=gpt_model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            seed=seed,
            tools=tools,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
    elif llm_choice.lower() == "llama":
        # Use default Llama model if none specified
        llama_model = model if model else "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        return get_completion_llama(
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            logprobs=logprobs,
            model=llama_model,
            #max_tokens=8000,
        )
    else:
        raise ValueError("Invalid LLM choice. Please choose either 'gpt' or 'llama'")

# #Example usage:
# prompt = '''Respond with only yes or no to the following question:
# Is earth round?
# '''
# response = get_llm_completion(
#     messages=[{"role": "user", "content": prompt}],
#     llm_choice="llama",
#     response_format={"type": "json_object"},
#     temperature=0.1,
#     logprobs=True,
# )
# print(response.choices[0].message.content)
# linear_probability = np.round(np.exp(response.choices[0].logprobs.token_logprobs[0])*100,2)
# print(linear_probability)
