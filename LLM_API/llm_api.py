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

# 150
# Paragraph examples:
# Resourcing email 
# 
#
#
#

# on the create profile -> pop-up where the user will select what it will be used for (e.g., email, report, teams message etc.) and then the LLM will generate a paragraph based on that.


def style_dict_to_guide(style: dict) -> str:
    """
    Convert a writing style dictionary into a style guide string.

    Parameters
    ----------
    style : dict
        Dictionary containing writing style attributes.

    Returns
    -------
    str
        A formatted writing style guide.
    """
    if not style:
        return ""

    lines = [
        "Follow the writing style below when generating text:"
    ]

    for key, value in style.items():
        lines.append(f"- {key}: {value}")

    return "\n".join(lines)

def get_llm(writing_style):
    
    prompt_text = """
    Write a single paragraph of 120–150 words explaining why a city might 
    choose to replace an ageing bridge rather than keep repairing it. 
    Cover the trade-off between upfront capital cost and ongoing maintenance,
    Return only the paragraph, with no preamble or commentary.
    """
    
    prompt = (prompt_text)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
                "role": "user",
                "content": prompt + writing_style,
            }],
        temperature=1,
        max_completion_tokens=512,
        top_p=1,
        # reasoning_effort="medium",
        stream=True,
        stop=None
    )

    response = ""

    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            response += content

    return response

def get_llm_placeholder(writing_style):
    return 'sample output'

