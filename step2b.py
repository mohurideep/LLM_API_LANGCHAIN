from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout=30, max_retries=1)

name_prompt = PromptTemplate.from_template(
    "I want to open a resturant for {Cuisine} food."
    "Suggest one fancy name. Reply with only the name."
)
name_chain = name_prompt | llm | StrOutputParser()

menu_prompt = PromptTemplate.from_template(
    "Suggest 5 menu items for a resturant called {resturant_name}"
    "Return them as one comma-seperated line, nothing else."
)
menu_chain = menu_prompt | llm | StrOutputParser()

simple_chain = {"resturant_name": name_chain} | menu_chain

print(simple_chain.invoke({"Cuisine":"Indian"}))