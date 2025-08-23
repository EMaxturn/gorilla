
import os
import base64
import re
import time
import random
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
    s = re.sub(r"$$([^$$]+)\]$[^)]+$", r"\1", s)
    # Remove plain (http...) style links
    s = re.sub(r"\s*$https?:[^)]+$", "", s)

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
    """Run GPT inference with infinite retry logic."""
    attempt = 0
    base_wait = 1  # Start with 1 second
    max_wait = 300  # Cap at 5 minutes
    
    while True:
        attempt += 1
        try:
            print(f"[GPT] Attempt {attempt}...")
            
            with open(image_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")

            response = gpt_client.responses.create(
                model="o4-mini",
                reasoning={"effort": "high","summary": "detailed" },
                tools=[{"type": "web_search_preview"}],
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Give a singular final answer to the best of your abilities (i.e a one word answer, a list of items, a date, a percentage statistic, an address, a name, etc.). No markdown or links in the final answer."
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
            
            # Success!
            print(f"[GPT] Success on attempt {attempt}")
            final_answer = _strip_markdown(response.output_text)
            reasoning_trace = get_all_summaries(response)
            return final_answer, reasoning_trace
            
        except Exception as e:
            error_msg = str(e)
            print(f"[GPT] Attempt {attempt} failed: {type(e).__name__}: {error_msg}")
            
            # Calculate exponential backoff with jitter
            wait_time = min(base_wait * (2 ** (attempt - 1)), max_wait)
            jitter = wait_time * 0.1 * random.random()  # Add up to 10% jitter
            total_wait = wait_time + jitter
            
            # Check for rate limit errors and use retry-after if available
            if "rate" in error_msg.lower() or "429" in error_msg:
                if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                    retry_after = e.response.headers.get('retry-after')
                    if retry_after:
                        total_wait = float(retry_after) + random.uniform(1, 5)
                        print(f"[GPT] Rate limited. Server says retry after {retry_after}s")
            
            print(f"[GPT] Waiting {total_wait:.1f}s before retry...")
            time.sleep(total_wait)

if __name__ == "__main__":
    result, trace = run_gpt_inference("../images/sports/3.png", "What is the defender's 3P percentage in the 2023-24 college season? Their team is a part of the Big Ten.")
    print("Answer:", result, "Trace:", trace)