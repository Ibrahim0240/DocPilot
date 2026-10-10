from ollama import Client

# Connect to local ollama server
client = Client(host="http://localhost:11434")

# send question to ollama model
response = client.chat(
    model="qwen3:1.7b",
    messages=[
        {
            "role": 'user',
            "content": "What is FastAPI? Answer in one sentence."
        }
    ],
    think=False
)

print(response.message.content)
