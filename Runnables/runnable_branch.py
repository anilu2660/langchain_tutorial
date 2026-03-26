from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnablePassthrough


load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template='summarise the following text \n {text}',
    input_variables=['text']
)

report_gen_chain = RunnableSequence(prompt1 ,model, parser )

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 300 , RunnableSequence(prompt2 , model , parser)),
    RunnablePassthrough()
)

final_chain =  RunnableSequence(report_gen_chain, branch_chain)

final_result = final_chain.invoke({'topic': 'Iran vs USA'})

print(final_result)