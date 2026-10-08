
# def generate_restaurant_name_and_items(cuisine):
#     return {
#         "cuisine": cuisine,
#         "restaurant_name": "Curry Delight",
#         "menu_items": "Paneer Tikka, Dal Makhani, Butter Naan",
#         "slogan": "Spice up your life",
#     }

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout=30, max_retries=1)

name_prompt = PromptTemplate.from_template(
        "I want to open a restaurant for {cuisine} food. "
        "Suggest ONE fancy name. Reply with only the name."
    )
name_chain = name_prompt | llm | StrOutputParser()

menu_prompt = PromptTemplate.from_template(
        "Suggest 5 {cuisine} dishes for a restaurant called {restaurant_name}.  "
        "Return them as one comma-separated line, nothing else."
    )
menu_chain = menu_prompt | llm | StrOutputParser()

slogan_prompt = PromptTemplate.from_template(
    "Create one catchy slogan for a {cuisine} restaurant called {restaurant_name}." \
    " Reply with only the slogan, plain text, no formatting."
    )
slogan_chain = slogan_prompt | llm | StrOutputParser()

full_chain = (RunnablePassthrough
                  .assign(restaurant_name = name_chain)
                  .assign(menu_items = menu_chain, slogan = slogan_chain))


def generate_restaurant_name_and_items(cuisine):
    
    try:
        result = full_chain.invoke({"cuisine": cuisine})

    except Exception as e:
        raise RuntimeError(f"Could not generate restaurant details for {cuisine} Cuisine") from e

    return {
        "restaurant_name" : result["restaurant_name"],
        "menu_items" : result["menu_items"],
        "slogan" : result["slogan"]
    }

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))