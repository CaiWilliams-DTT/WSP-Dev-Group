import os
from dotenv import load_dotenv
from groq import Groq


# Considerations:
# Multi Model?
# Ask the LLM to re-write something (possibly user input) or prompt it   

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=API_KEY)

PROMPTS = [
    "Write me an email [context] using the following writing style"
]


def get_groq(prompt_text, writing_style):
    prompt = (prompt_text)
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
                "role": "user",
                "content": prompt + writing_style,
            }],
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
        # reasoning_effort="medium",
        stream=True,
        stop=None
    )

    response = ""

    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="")
            response += content

    return response


