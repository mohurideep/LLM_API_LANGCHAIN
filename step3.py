from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout=30, max_retries=1)

#chain 1
name_prompt = PromptTemplate.from_template(
    "I want to open a restaurant for {cuisine} food. "
    "Suggest ONE fancy name. Reply with only the name."
)
name_chain = name_prompt | llm | StrOutputParser()

#chain 2 , name > menu
menu_prompt = PromptTemplate.from_template(
    "Suggest 5 menu items for a restaurant called {restaurant_name}. "
    "Return them as one comma-separated line, nothing else."
)
menu_chain = menu_prompt | llm | StrOutputParser()

full_chain = (
    RunnablePassthrough.assign(restaurant_name=name_chain)
    .assign(menu_items=menu_chain)
)
result = full_chain.invoke({"cuisine":"Indian"})
print(result)
print("---------------")
print("Name: ", result["restaurant_name"])
print("Menu: ", result["menu_items"])