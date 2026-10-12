from datetime import date

import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage,ToolMessage
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout=30, max_retries=1)

HEADERS = {"User-Agent": "LangChainCourse/1.0 (learning project)"}

@tool
def get_current_year() -> int:
    """Return the current year."""
    return date.today().year

@tool
def subtract(a:int, b:int) -> int:
    """Subtract b from a and  return the result."""
    return a-b

@tool
def search_wikipedia(query:str) -> str:
    """Look up a topic on wikipedia and  return short summary.
    Use this for facts about people, places, events or things."""
    try:
        search = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "format": "json", "srlimit": 1},
            headers=HEADERS,
            timeout=10
        )
        search.raise_for_status()
        result = search.json()["query"]["search"] #[][] digs into json reply and give it in json format
        if not result:
            return f"No WikiPedia page found for '{query}'. Try a different search."
        
        title = result[0]["title"]
        summary = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_"),
            headers=HEADERS,
            timeout=10,
        )
        summary.raise_for_status()
        return f"{title}: {summary.json()['extract'][:1500]}"
    except requests.exceptions.RequestException as e:
        return f"Wikipedia could not be reached: {e}"
    
agent = create_agent(
    model=llm,
    tools=[get_current_year, subtract, search_wikipedia],
    system_prompt=(
        "You are a helpful assistant. Use search_wikipedia for facts you are not sure about. "
        "Always use tools for date and math. keep answer short."
    )
)

questions = [
    "When was Elon Musk born, and how old is he now?",
    "Who founded Infosys?",
    "Hi! How are you?",
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