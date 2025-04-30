import os
from openai import AzureOpenAI
from services import use_model


# Initialize the client
os.environ["AZURE_OPENAI_ENDPOINT"] = "ENDPOINT"
os.environ["AZURE_OPENAI_API_KEY"] = "API_KEY"

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01")

def get_chat_response(role: str, prompt: str):
    selected_model = use_model.select_model(prompt)
    print(f"Selected Model: {selected_model}")
    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": role, "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    input_prompt = {"request": "multiply 5 and 4"}
    response = get_chat_response("user", "Test prompt")
    print(f"Test response from get_chat_response: {response}")