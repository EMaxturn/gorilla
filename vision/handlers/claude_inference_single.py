import anthropic
import base64
import json
import os

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Set ANTHROPIC_API_KEY in your environment (or load it via a .env)."
    )

client = anthropic.Anthropic(api_key=API_KEY)

def get_final_answer_text_json(msg):
    text_blocks = [b["text"] for b in msg.get("content", []) if b.get("type") == "text"]
    return " ".join(text_blocks) if text_blocks else ""

def run_claude_inference(image_path: str, query: str) -> str:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    image_media_type = "image/png" if image_path.lower().endswith(".png") else (
        "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "application/octet-stream"
    )

    message = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=1024,
        system="Give a singular final answer to the best of your abilities (i.e a one word answer, a percentage statistic, an address, a name, etc.)",
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
        }]
    )

    # Extract & return final text
    # return (message)
    message_dict = json.loads(message.model_dump_json())
    return get_final_answer_text_json(message_dict)

# standalone test
if __name__ == "__main__":
    result = run_claude_inference("../images/sports/3.png", "What is the defender's 3P percentage in the 2023-24 college season? Their team is a part of the Big Ten.")
    print("Answer:", result)
