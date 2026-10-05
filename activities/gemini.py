from google import genai

API_KEY = "<your_api_key_here>"  # better: os.environ["GEMINI_API_KEY"]

client = genai.Client(api_key=API_KEY)

def ask_gemini(prompt, model="gemini-3.5-flash-lite", temperature=1.0):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": temperature},
    )
    return response.text


if __name__ == "__main__":
    print(ask_gemini("Explain softmax temperature in two sentences."))