from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout=30, max_retries=1)

prompt = PromptTemplate.from_template(
    "Write one short and orginal Joke, that is specifically about {topic}. "
    "Do not use 'Impasta' Joke. Reply with only the Joke, nothing else."
)

chain = prompt | llm | StrOutputParser()

for topic in ["Mexican food", "Indian food", "Italian food"]:
    response = chain.invoke({"topic": topic})
    print (topic, ":", response)