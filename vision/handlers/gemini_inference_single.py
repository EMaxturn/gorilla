# Gemini Inference
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Set GOOGLE_API_KEY in your environment (or load it via a .env)."
    )

# Configure the generative AI library
genai.configure(api_key=API_KEY)
gemini_client = genai.Client()

def run_gemini_inference(query, image_path):
    MODEL_ID = "gemini-2.5-pro"

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    image = types.Part.from_bytes(
        data=image_bytes, mime_type="image/jpeg"
    )

    response = gemini_client.models.generate_content(
        model=MODEL_ID,
        contents=[query, image],
        config={"tools": [{"google_search": {}}]},
    )

    return response.text

if __name__ == '__main__':
    print(
        run_gemini_inference(
            "puppy.jpg",
            "What is the thing in the image and find latest news about it in Delhi, India"
        )
    )