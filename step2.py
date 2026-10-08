from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout=30, max_retries=1)

prompt = PromptTemplate.from_template(
    "I want to open a restaurant for {cuisine} food. "
    "Suggest ONE fancy name. Reply with only the name, nothing else."
)

chain = prompt | llm | StrOutputParser()

for cuisine in ["Mexican", "Indian", "Italian"]:
    response = chain.invoke({"cuisine": cuisine})
    print (cuisine, ":", response)