from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model = "gemini-3.5-flash-lite",
#     temperature = 1,
#     timeout = 30,
#     max_retries = 1
# )

llm = ChatOllama(model="mistral", temperature=1)

prompt = PromptTemplate.from_template(
    "I want to open a resturant for {cuisine} food. "
    "Suggest one fancy name. reply with only the name, nothing else"
)
text = prompt.format(cuisine="mexican")
print("Prompt: ", text)

response = llm.invoke(text)
print("Answer: ", response.text)
print("Output Tokens:", response.usage_metadata["output_tokens"])