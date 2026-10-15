from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from step5c import calculate_age, search_wikipedia

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout=30, max_retries=1)

agent = create_agent(
    model=llm,
    tools=[calculate_age,search_wikipedia],
    system_prompt=(
        "You are a helpful assistant. Use calculate_age for ages and "
        "search_wikipedia for facts. Keep answers short."
    ),
    checkpointer=InMemorySaver(),
    middleware=[
        ModelCallLimitMiddleware(run_limit=8),
        ToolCallLimitMiddleware(tool_name="search_wikipedia", run_limit=3),
        SummarizationMiddleware(model=llm, trigger=("messages",10), keep=("messages",4))
    ]
)

def ask(text, thread_id):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    print(f"[{thread_id}] You: {text}")
    print(f"[{thread_id}] Bot: {result['messages'][-1].text}")
    print(f"[{thread_id}] (messages stored in this thread: {len(result['messages'])})\n")

ask("Hi, my name is Deep and I was born on 5 March 1995.", "deep")
ask("How old am I?", "deep")
ask("How old am I?", "priya")
ask("What is my name?", "deep")
ask("When was Sachin Tendulkar born, and how old is he?", "deep")
ask("Remind me: what is my name and how old am I?", "deep")