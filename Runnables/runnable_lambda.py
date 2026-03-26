from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence ,RunnableParallel, RunnableLambda, RunnablePassthrough


load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= 'Generate a joke on {topic}',
    input_variables=['topic']
)

chain1 = RunnableSequence(prompt1 , model, parser)

chain2_parallel = RunnableParallel({
    'joke': RunnablePassthrough(),
    'no_of_words': RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(chain1 , chain2_parallel)

result =final_chain.invoke({'topic': 'cats'})

final_result = '''joke: {} \n word count - {}'''.format(result['joke'], result['no_of_words'])

print(final_result)