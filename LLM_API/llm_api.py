import os
import threading
from dotenv import load_dotenv
from groq import Groq
import random

PROMPTS = [
    # --- Internal logistics (low-stakes, terse) ---
    "Write an email to a colleague asking them to move 8 hours of charged time from the ECL audit engagement code to the internal audit engagement code, as the work was booked against the wrong code last Thursday.",
    "Write an email to your manager requesting approval for five days of annual leave from 14-18 September for a family holiday, noting that your current engagements have coverage in place.",
    "Write a Teams message asking a colleague to review your workpaper on the LGD recalibration before 5pm today, as it needs to go to the manager tonight.",
    "Write an email declining a request to join a new engagement starting next week, as you are fully utilised across two existing engagements until the end of the month.",

    # --- Client-facing delivery (formal, external) ---
    "Write an email to a client attaching the final model validation report, confirming the engagement is complete and noting that the invoice will follow separately.",
    "Write an email to a client requesting a loan-level data extract covering the last 24 months, with the fields agreed at the kick-off meeting, required by close of business Friday.",
    "Write an email to a client confirming a 90-minute walkthrough session on Wednesday at 10am, listing the four attendees and the three agenda items to be covered.",
    "Write an email following up on an information request sent two weeks ago that remains outstanding, noting that this is the second follow-up and that the delay is now affecting the delivery timeline.",

    # --- Bad news / friction (hedging, diplomacy) ---
    "Write an email advising the client that the draft report will be delivered one week later than agreed, because the data received from their team was incomplete.",
    "Write an email flagging to the client that their request to add two additional portfolios falls outside the agreed scope, and that a variation to the engagement letter may be required.",
    "Write an email pushing back on the client's proposal to change the PD segmentation approach halfway through the engagement, explaining the impact on comparability of results already produced.",
    "Write an email to the engagement partner escalating a data quality issue, where 30 per cent of records have missing origination dates, which is currently blocking substantive testing.",

    # --- Technical explanation (density, jargon control) ---
    "Write a two-paragraph summary for a non-technical audience explaining a validation finding that the model's LGD estimates are materially understated for the unsecured portfolio.",
    "Write an email explaining to the client why the SICR threshold has been recalibrated from three times to two times lifetime PD, and what effect this has on Stage 2 balances.",
    "Write speaker notes for a slide explaining PD segmentation to an audit committee with limited modelling background.",
    "Write a response to a client query asking whether their IRB model documentation meets the requirements of APRA APS 113.",

    # --- Review and feedback (tone toward juniors) ---
    "Write review comments on a junior analyst's draft workpaper that reaches the right conclusions but has poor structure, unclear referencing, and no documented rationale for the sample selection.",
    "Write an email thanking the team after the final deliverable was issued ahead of a tight deadline that required two weekends of work.",
    "Write feedback notes for a performance conversation with an analyst who produces accurate technical work but rarely speaks up in client meetings.",

    # --- Upward and status (concision under scrutiny) ---
    "Write a three-bullet status update to the director covering testing progress to date, one open issue, and the date the draft report will be ready.",
    "Write a weekly summary email to the engagement lead covering work completed this week, work planned for next week, and any blockers.",
    "Write a one-line Teams reply confirming that the requested reconciliation has been completed and saved to the shared folder.",
]

# .env is a local development convenience only. A hosted deployment takes its
# configuration from real environment variables (Azure application settings),
# and must not read a .env that happened to get bundled into the deployment:
# this module is imported before app.py reads DEV_METRICS, so a stray file
# would silently switch on the dev diagnostics routes and install its
# GROQ_API_KEY as the deployment default. Mirrors IS_PRODUCTION in UI/app.py.
_IS_PRODUCTION = os.environ.get(
    "APP_ENV", "production" if os.environ.get("WEBSITE_SITE_NAME") else "development"
).strip().lower() == "production"
if not _IS_PRODUCTION:
    load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

# Clients are built on demand and cached per key rather than once at import.
# Importing this module must not require a key: a deployment may have none of
# its own and serve only users who bring their own, and a missing .env should
# not take the whole app down at start-up. The cache is bounded so a stream of
# user-supplied keys cannot grow it without limit.
_CLIENT_CACHE_MAX = 32
_clients = {}
_clients_lock = threading.Lock()


class MissingAPIKey(RuntimeError):
    """No usable Groq key: none supplied by the caller and none in the environment."""


def has_default_key() -> bool:
    """Whether this deployment has its own GROQ_API_KEY to fall back on."""
    return bool(API_KEY)


def get_client(api_key: str | None = None) -> Groq:
    """
    Return a Groq client for `api_key`, falling back to GROQ_API_KEY.

    Parameters
    ----------
    api_key : str, optional
        A caller-supplied key — typically one a user entered in the UI so
        that their own account, not the deployment's, is billed.
    """
    key = (api_key or API_KEY or "").strip()
    if not key:
        raise MissingAPIKey("No Groq API key available — set GROQ_API_KEY in .env "
                            "or supply one in the app.")
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            if len(_clients) >= _CLIENT_CACHE_MAX:
                _clients.pop(next(iter(_clients)))     # evict oldest
            client = _clients[key] = Groq(api_key=key)
        return client


def forget_client(api_key: str) -> None:
    """Drop a cached client so its key stops being held in memory."""
    with _clients_lock:
        _clients.pop((api_key or "").strip(), None)


# def generate_image(
#     prompt: str,
#     output_file: str = "image.jpg",
#     api_url: str = "https://imagegen.cwilliams1.workers.dev/",
#     api_key: str = "YOUR_API_KEY",
# ):
#     """
#     Generate an image from a text prompt and save it to disk.

#     Parameters
#     ----------
#     prompt : str
#         The text prompt describing the image.
#     output_file : str
#         Path where the image will be saved.
#     api_url : str
#         Image generation API endpoint.
#     api_key : str
#         Bearer token for authentication.
#     """
#     # needs `requests` added back to the dependencies
#     response = requests.post(
#         api_url,
#         headers={
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json",
#         },
#         json={"prompt": prompt},
#         timeout=300,
#     )

#     response.raise_for_status()

#     with open(output_file, "wb") as f:
#         f.write(response.content)

#     return output_file


# Connect to deloitte LLM gateway


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

def pick_prompt() -> str:
    """
    Choose a writing task at random from PROMPTS.

    Callers that generate more than one sample for the same comparison should
    call this once and pass the result to every `get_llm` call, so the samples
    differ only by writing style and not by the task being written about.
    """
    return random.choice(PROMPTS)


def get_llm(writing_style, api_key: str | None = None, prompt: str | None = None):
    """
    Generate one writing sample in the given style.

    `api_key` overrides the deployment's GROQ_API_KEY for this call, so a
    user who supplies their own key is billed for their own generations.

    `prompt` is the writing task to perform; when omitted a random one is
    drawn, which is only appropriate for a standalone sample.
    """
    prompt = prompt if prompt is not None else pick_prompt()

    completion = get_client(api_key).chat.completions.create(
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

def get_llm_placeholder(writing_style, api_key: str | None = None,
                        prompt: str | None = None):
    """Offline stand-in for get_llm; costs nothing and calls nothing."""
    return 'sample output'

if __name__ == "__main__":
    # generate_image(
    #     prompt="Create me a vintage retro tech style picture of a character face. " \
    #     "It should resemble an old retro ui style. Ensure the image has a " \
    #     "black background and the face is constrcted with just neon green. " \
    #     "The quality should look like it was rentered on an old computer terminal. " \
    #     "This will be used for a profile picture. so ensure there is no text in the image.",
    #     output_file="image.jpg",
    #     api_key=API_KEY,
    # )

    # print("Image saved to image.jpg")
    pass