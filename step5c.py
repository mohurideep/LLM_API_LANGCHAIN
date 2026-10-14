from datetime import date

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout=30, max_retries=1)

HEADERS = {"User-Agent": "LangChainCourse/1.0 (learning project)"}

# @tool
# def get_current_year() -> int:
#     """Return the current year."""
#     return date.today().year

# @tool
# def subtract(a:int, b:int) -> int:
#     """Subtract b from a and  return the result."""
#     return a-b

@tool
def calculate_age(birth_year: int, birth_month: int, birth_day: int) -> int:
    """Calculate age based on birth date."""
    today = date.today()
    age = today.year - birth_year
    if (today.month, today.day) < (birth_month, birth_day):
        age -= 1

    return age

@tool
def search_wikipedia(query:str) -> str:
    """Look up a topic on Wikipedia and return the introduction of the best matching page.
    Use this for facts about people, places, events or things."""
    try:
        search = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": 1,
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "format": "json",
            },
            headers=HEADERS,
            timeout=10
        )
        search.raise_for_status()
        pages = search.json().get("query", {}).get("pages", {})
        if not pages:
            return f"No WikiPedia page found for '{query}'. Try a different search."
        page = next(iter(pages.values()))
        return f"{page['title']}: {page['extract'][:1500]}"
    except requests.exceptions.RequestException as e:
        return f"Wikipedia could not be reached: {e}"
    
agent = create_agent(
    model=llm,
    tools=[search_wikipedia, calculate_age],
    system_prompt=(
        "You are a helpful assistant. Use search_wikipedia for facts. "
    "Use calculate_age for ages. "
    "If a tool fails or does not give you the answer, say clearly that you could not "
    "verify it, and mark any answer from your own knowledge as unverified. "
    "Keep answers short."
    ),
    middleware=[
        ToolCallLimitMiddleware(tool_name="search_wikipedia", run_limit=3),
        ModelCallLimitMiddleware(run_limit=8)
    ]
)

questions = [
    "When was Elon Musk born, and how old is he now?",
    "How old is Rajinikanth?",
]

for q in questions:
    print("QUESTION :", q)
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    for m in result["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            for call in m.tool_calls:
                print(f"  → calls {call['name']}({call['args']})")
        elif isinstance(m, ToolMessage):
            print(f"  ← got: {str(m.content)[:100]}...")
    print("ANSWER:", result["messages"][-1].text)
    print("=" * 50)