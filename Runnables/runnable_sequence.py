from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()


prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 =  PromptTemplate(
    template='Explain the joke from following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({'topic': 'AI'})

print(result)

'''
LCEL(Langchain expression language): Piped operator (|)

In langchain we can use piped operator instead RunnableSequence() runnable multiple times.

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'AI'})

print(result)
 
'''