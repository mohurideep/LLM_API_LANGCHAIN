from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout=30, max_retries=1)

SYSTEM = SystemMessage("You are a friendly assistant. Keep answer short.")
WINDOW = 3
history = []

print ("Chat started, type 'Quit' to stop.")

while True:
    user_test = input("You: ")
    if user_test.lower() in ("quit","exit"):
        break

    history.append(HumanMessage(user_test))
    recent = history[-(WINDOW*2 + 1):]

    response = llm.invoke([SYSTEM]+recent)
    history.append(response)

    print("Bot: ", response.text)
    print(f"    [sent {len(recent)} of {len(history)-1} messages,"
          f"input tokens: {response.usage_metadata['input_tokens']}]")