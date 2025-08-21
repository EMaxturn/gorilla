import os
import base64
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment (or load it via a .env).")

gpt_client = OpenAI(api_key=API_KEY)

def get_all_summaries(resp) -> str:
    parts = []
    for item in getattr(resp, "output", []):
        if getattr(item, "type", "") == "reasoning":
            for s in getattr(item, "summary", []):
                if hasattr(s, "text"):
                    parts.append(s.text)
                elif isinstance(s, dict) and "text" in s:
                    parts.append(s["text"])
    return _strip_markdown(" ".join(parts))

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
    # collapse extra whitespace/newlines
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def run_gpt_inference(image_path, query):
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    response = gpt_client.responses.create(
        model="o4-mini",
        reasoning={"effort": "high","summary": "detailed" },
        # If you don't need web links, you can REMOVE this tools line to further reduce list-y outputs.
        tools=[{"type": "web_search_preview"}],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Give a singular final answer to the best of your abilities (i.e a one word answer, a list of items, a date, a percentage statistic, an address, a name, etc.) "+
                            "No markdown or links in the final answer."
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": query},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_base64}"}
                ]
            }
        ],
    )
    print(response)
    final_answer = _strip_markdown(response.output_text)
    reasoning_trace = get_all_summaries(response)
    return final_answer, reasoning_trace

if __name__ == "__main__":
    result, trace = run_gpt_inference("../images/sports/3.png", "What is the defender's 3P percentage in the 2023-24 college season? Their team is a part of the Big Ten.")
    print("Answer:", result, "Trace:", trace)
