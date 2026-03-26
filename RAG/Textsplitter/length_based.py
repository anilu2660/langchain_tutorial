import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('D:\langchain\RAG\Textsplitter\main.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap= 0,
    separator= ''
)

result = splitter.split_documents(docs)
print(result[0])





