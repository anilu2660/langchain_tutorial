from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

# Chains works because of runnables 
load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate a detailed report on {topic} economy',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Summarize the following report in 5 bullet points: {report}',
    input_variables=['report']
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser


result = chain.invoke({'topic': 'India'})
print(result)

chain.get_graph().print_ascii()