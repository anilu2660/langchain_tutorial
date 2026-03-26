from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence ,RunnableParallel


load_dotenv()

prompt1 = PromptTemplate(
    template='generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='generate a linkedin post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

model = ChatOpenAI()

parallel_chain = RunnableParallel(
    {
        'tweet': RunnableSequence(prompt1 , model ,parser),
        'linkedin': RunnableSequence(prompt2 , model ,parser)
    }
)

result = parallel_chain.invoke({'topic': 'AI'})
print('twiiter tweet: ')
print(result['tweet'])
print()
print('linkedin post: ')
print(result['linkedin'])
