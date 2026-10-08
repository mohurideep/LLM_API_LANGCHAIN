from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    timeout=30,
    max_retries=1
    )
response = llm.invoke("Suggest any fancy name for a Mexican resturant")
print(response.text)
print("________")
print(response)