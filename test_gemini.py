from google import genai
from settings import GEMINI_TEST_MODEL

# Initialize client (settings.py loads the project .env file)
client = genai.Client()

# 4. Generate content (asking in English to avoid terminal encoding issues)
response = client.models.generate_content(
    model=GEMINI_TEST_MODEL,
    contents="Hi! Write a short joke about Python programmers in English."
)

print(response.text)
