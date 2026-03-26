
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path=r'D:\langchain\RAG\books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()
# Lazy_load() is used when we have to load heavy number of documents
# It is used to load documents in chunks
print(len(docs))
print(docs[2].page_content)
print(docs[2].metadata)