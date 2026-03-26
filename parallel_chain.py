from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

model1 = ChatOpenAI(model='gpt-5')


model2 = ChatOpenAI(model='gpt-4o-mini')

prompt1 = PromptTemplate(
    template = 'Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template = 'generate 5 short quesitions and answer from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template = 'merge the provied notes and quiz into a single documents \n {notes} and \n {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Finite automata are abstract machines used to recognize patterns in input sequences, forming the basis for understanding regular languages in computer science.

Consist of states, transitions, and input symbols, processing each symbol step-by-step.
If ends in an accepting state after processing the input, then the input is accepted; otherwise, rejected.
Finite automata come in deterministic (DFA) and non-deterministic (NFA), both of which can recognize the same set of regular languages.
Widely used in text processing, compilers, and network protocols.

Features of Finite Automata: 
Input: Set of symbols or characters provided to the machine.
Output: Accept or reject based on the input pattern.
States of Automata: The conditions or configurations of the machine.
State Relation: The transitions between states.
Output Relation: Based on the final state, the output decision is made.
"""

result = chain.invoke({'text': text})
print(result)

chain.get_graph().print_ascii()


