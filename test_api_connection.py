import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

def test_anthropic_connection():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in .env file")
        return

    print(f"✅ API Key found: {api_key[:15]}...")

    client = anthropic.Anthropic(api_key=api_key)

    print("⏳ Testing connection to Anthropic API...")

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say 'API connection successful!' and nothing else."}]
    )

    print(f"✅ Connection successful!")
    print(f"📝 Response: {response.content[0].text}")
    print(f"📊 Tokens used — Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}")


if __name__ == "__main__":
    test_anthropic_connection()
