import anthropic
import base64
import json
import os
import re
from dotenv import load_dotenv


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Set ANTHROPIC_API_KEY in your environment (or load it via a .env)."
    )


def extract_answer(text: str) -> str | None:
    m = re.search(r"<<\s*(.+?)\s*>>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else "I don't know"


client = anthropic.Anthropic(api_key=API_KEY, max_retries=5)

def get_final_answer_text_json(msg):
    text_blocks = [b["text"] for b in msg.get("content", []) if b.get("type") == "text"]
    return " ".join(text_blocks) if text_blocks else ""

def get_clean_thinking_text(msg_dict: dict) -> str:
    # Join all thinking blocks, then collapse whitespace/newlines
    raw = " ".join(
        b.get("thinking", "")
        for b in msg_dict.get("content", [])
        if b.get("type") == "thinking"
    )
    return re.sub(r"\s+", " ", raw).strip()

def run_claude_inference(image_path: str, query: str) -> str:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    image_media_type = "image/png" if image_path.lower().endswith(".png") else (
        "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "application/octet-stream"
    )

    with client.messages.stream(
        model="claude-opus-4-20250514",
        max_tokens=32000,
        thinking={
        "type": "enabled",
        "budget_tokens": 1024, # this is the minimum value, I don't think there's a default
        },
        system="Give a singular final answer to the best of your abilities (i.e a one word answer, a list of items, a date, a percentage statistic, an address, a name, etc.). No markdown or links in the final answer. Always format final answer as << FINAL ANSWER >>. If you are not sure of the final answer or require more information, respond with << I don't know >>.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": query
                    }
                ],
            }
        ],
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
        }],
    )   as stream:
            message = stream.get_final_message()

    # Extract & return final text
    # print (message)
    message_dict = json.loads(message.model_dump_json())
    final_answer = extract_answer(get_final_answer_text_json(message_dict))
    reasoning_trace =   get_clean_thinking_text(message_dict)
    # print(reasoning_trace)
    return final_answer, reasoning_trace

# standalone test
if __name__ == "__main__":
    result, trace = run_claude_inference("../images/sports/3.png", "What is the defender's 3P percentage in the 2023-24 college season? Their team is a part of the Big Ten.")
    print("Answer:", result, "Trace:", trace)
