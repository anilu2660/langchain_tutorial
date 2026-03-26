from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence ,RunnableParallel, RunnablePassthrough

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'Generate a joke on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Explain the joke {text}',
    input_variables=['text']
)


joke_generator = RunnableSequence(prompt1 , model , parser)

chain_parallel = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'joke_explanation': RunnableSequence(prompt2 , model , parser)

    }
)

final_chain = RunnableSequence(joke_generator, chain_parallel)

result = final_chain.invoke({'topic': 'cricket'})
print(result)
