from datetime import date

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout=30, max_retries=1)

@tool
def get_current_year() -> int:
    """Return the current year."""
    return date.today().year

@tool
def subtract(a:int, b:int ) -> int:
    """Subtract b from a and return the result."""
    return a-b




llm_with_tools = llm.bind_tools([get_current_year, subtract])

# questions = [
#     "How old is someone who was born in 1971?",
#     "What is the capital of France?",
# ]

# for q in questions:
#     response = llm_with_tools.invoke(q)
#     print("QUESTION:", q)
#     print("TEXT:", response.text)
#     print("TOOL CALLS:", response.tool_calls)
#     print("-" * 40)


tools_by_name = {"get_current_year": get_current_year, "subtract": subtract}

messages = [HumanMessage("How old is someone who was born in 1971?")]

for step in range(5):
    ai = llm_with_tools.invoke(messages)
    messages.append(ai)


    if not ai.tool_calls:
        break

    for call in ai.tool_calls:
        tool_to_run = tools_by_name[call["name"]]
        result = tool_to_run.invoke(call["args"])
        print(f"Step {step + 1}: {call['name']}({call['args']}) -> {result}")
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

print("FINAL ANSWER:", ai.text)
