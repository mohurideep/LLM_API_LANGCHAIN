from datetime import date

from dotenv import load_dotenv
from langchain.agents import create_agent
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
def subtract(a: int, b: int) -> int:
    """Does stuff."""
    return a - b


QUESTION = "How old is someone who was born in 1971?"


# ---------- Version 1: the loop by hand ----------
print("=== HAND-MADE LOOP ===")
llm_with_tools = llm.bind_tools([get_current_year, subtract])
tools_by_name = {"get_current_year": get_current_year, "subtract": subtract}
messages = [HumanMessage(QUESTION)]

for step in range(5):
    ai = llm_with_tools.invoke(messages)
    messages.append(ai)
    if not ai.tool_calls:
        break
    for call in ai.tool_calls:
        result = tools_by_name[call["name"]].invoke(call["args"])
        print(f"Step {step + 1}: {call['name']}({call['args']}) -> {result}")
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

print("FINAL ANSWER:", ai.text)
print()

# ---------- Version 2: create_agent ----------
print("=== CREATE_AGENT ===")
agent = create_agent(
    model=llm,
    tools=[get_current_year,subtract],
    system_prompt="You are a helpful assistant. Always use tools for dates and math.",
)
result = agent.invoke({"messages":[{"role": "user", "content": QUESTION}]})

for message in result["messages"]:
    message.pretty_print()

print("FINAL ANSWER:", result["messages"][-1].text)