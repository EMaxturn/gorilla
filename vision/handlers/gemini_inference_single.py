# Gemini Inference

import google.generativeai as genai
from google.colab import userdata
from google.colab import userdata


# Access your API key from Colab's secrets manager
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

# Configure the generative AI library
genai.configure(api_key=GOOGLE_API_KEY)

from google.genai import types
from google.generativeai import GenerativeModel # Import GenerativeModel

with open('Screenshot 2025-08-15 at 6.05.47 PM.png', 'rb') as f:
      image_bytes = f.read()

# Instantiate the model
# Available models with v1API: https://ai.google.dev/gemini-api/docs/models
model = GenerativeModel('gemini-2.5-pro') # or 2.5-flash, 2.5-flash-lite, etc 

response = model.generate_content(
    contents=[
      {
          'mime_type': 'image/png',
          'data': image_bytes
      },
      'what is going on here' # prompt here
    ]
  )
 
print(response.text)
