# Gemini Inference
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import re

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Set GOOGLE_API_KEY in your environment (or load it via a .env)."
    )

# Configure the generative AI library
gemini_client = genai.Client(api_key=API_KEY)

def extract_answer(text: str) -> str | None:
    m = re.search(r"<<\s*(.+?)\s*>>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else "I don't know"

def extract_thoughts(response):
    """Return only the reasoning/thought summaries as a list of strings."""
    thoughts = []
    for cand in getattr(response, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            if getattr(part, "thought", False) and getattr(part, "text", None):
                thoughts.append(part.text.strip())
    return thoughts or None

def clean_thoughts(thought_list):
    """Flatten list of thought strings into one continuous cleaned string (no newlines)."""
    if not thought_list:
        return ""
    combined = " ".join(t.strip() for t in thought_list if t and t.strip())
    return _strip_markdown(combined)

def _strip_markdown(s: str) -> str:
    if not isinstance(s, str):
        return s
    # Remove markdown links: [text](url) -> text
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    # Remove plain (http...) style links
    s = re.sub(r"\s*\(https?:[^)]+\)", "", s)

    # headings (## Title) -> Title
    s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s, flags=re.MULTILINE)
    # bullets and numbered lists
    s = re.sub(r"^\s*[-*+]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*\d+[\.)]\s+", "", s, flags=re.MULTILINE)
    # inline code / bold / italics markers
    s = s.replace("**", "").replace("*", "").replace("`", "")
    # collapse whitespace/newlines
    s = re.sub(r"\s+", " ", s).strip()
    return s

def run_gemini_inference(image_path, query):
    MODEL_ID = "gemini-2.5-pro"

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # Determine image MIME type based on file extension
    if image_path.lower().endswith('.png'):
        mime_type = "image/png"
    elif image_path.lower().endswith(('.jpg', '.jpeg')):
        mime_type = "image/jpeg"
    else:
        mime_type = "image/jpeg"  # default fallback

    image = types.Part.from_bytes(
        data=image_bytes, mime_type=mime_type
    )

    # Add system prompt for consistent formatting
    system_prompt = (
        "Give a singular final answer to the best of your abilities (i.e a one word answer, a list of items, a date, a percentage statistic, an address, a name, etc.). "
        "The final answer MUST be enclosed in double angle brackets like << answer >>. "
        "The content inside the brackets must be plain text only, with no markdown or links. "
        "If a definitive answer cannot be found, you MUST respond with << I don't know >> and nothing else."
    )
    # Combine system prompt with user query
    formatted_query = f"{system_prompt}\n\nUser Query: {query}"

    response = gemini_client.models.generate_content(
        model=MODEL_ID,
        contents=[formatted_query, image],
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            thinking_config=types.ThinkingConfig(include_thoughts=True)
            )
    )
    # print(response)

    # --- FIX IS HERE ---
    # The response text is nested inside candidates -> content -> parts
    text_content = ""  # Default to empty string
    if response.text:
        text_content = response.text
    # --- END OF FIX ---
    cleaned_text = _strip_markdown(text_content)
    final_answer = extract_answer(cleaned_text)
    reasoning_trace = clean_thoughts(extract_thoughts(response))

    return final_answer, reasoning_trace

if __name__ == "__main__":
    result, trace = run_gemini_inference("../images/sports/3.png", "What is the defender's 3P percentage in the 2023-24 college season? Their team is a part of the Big Ten.")
    print("Answer:", result, "Trace:", trace)