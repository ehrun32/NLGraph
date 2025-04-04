# models/wrappers.py

import os
import openai
from dotenv import load_dotenv

load_dotenv()

def call_openai_chat(model, prompt, temperature=0, max_tokens=400):
    """
    Calls OpenAI's ChatCompletion endpoint.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if "gpt" not in model:
        raise ValueError("Use 'gpt-3.5-turbo', 'gpt-4', or 'gpt-4o' with this wrapper.")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def call_anthropic_claude(model, prompt, temperature=0.7, max_tokens=400):
    """
    Stub for Claude 3 models via Anthropic API.
    """
    # import anthropic
    # client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # response = client.messages.create(...)
    # return response.content[0].text
    raise NotImplementedError("Claude API wrapper not yet implemented.")


def call_gemini(model, prompt, temperature=0.7, max_tokens=400):
    """
    Stub for Gemini via Google Generative AI SDK.
    """
    # import google.generativeai as genai
    # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    # model = genai.GenerativeModel('gemini-pro')
    # response = model.generate_content(...)
    # return response.text
    raise NotImplementedError("Gemini API wrapper not yet implemented.")

