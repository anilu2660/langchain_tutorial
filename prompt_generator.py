
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template= """Please summarise titled "{paper_input}" with the following specifications:
    Explanation Style: {style_input}
    Explanation Length: {length_input}
    1. Mathematical details:
    - include relevant mathematical equations if present in the paper
    - Explain the mathematical concepts using simple , intuitive code snippets where applicable.
    2. Analogies:
     - Use relatable anolaogies to simplify complex ideas.
    if certain informaation is not available in the paper , respond with : "Insufficient information available" instead of guessing.
    Ensure the summary is clear and concise and aligned with provided style and length.
     """,
     
input_variables = ['paper_input', 'style_input', 'length_input']
)

template.save('template.json')