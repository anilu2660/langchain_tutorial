import os

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template='Answer the following quesition \n {quesition} from given text -\n  {text}',
  
    input_variables=['quesition','text']
)

parser = StrOutputParser()
url = 'https://timesofindia.indiatimes.com/entertainment/hindi/bollywood/news/lilliput-says-shah-rukh-khan-doesnt-have-script-sense-like-aamir-khan-clarifies-he-never-predicted-zero-box-office-failure/articleshow/129178164.cms'
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

final = chain.invoke({'quesition': 'What is role of aamir khan in this article ?', 'text': docs[0].page_content })

print(final)