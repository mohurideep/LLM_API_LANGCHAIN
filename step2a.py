from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout=30, max_retries=1)

prompt = PromptTemplate.from_template(
    "I want to write joke about {topic}. "
    "Suggest ONE funny Joke. Reply with only the Joke, nothing else."
)

chain = prompt | llm | StrOutputParser()

for topic in ["Mexican food", "Indian food", "Italian food"]:
    response = chain.invoke({"topic": topic})
    print (topic, ":", response)