import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import langchain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

response = model.invoke("What is the capital of india?")
print(response.content)
