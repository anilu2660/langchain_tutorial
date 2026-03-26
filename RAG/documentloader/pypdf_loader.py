# it works pages by pages if we provide 25 pages pdf then it will different 25 document object and store in list

# It uses the Pypdf under the hood --not great for scanned pdf or complex layouts for that we have to use different loaders

# different pdf loader: PypdfLoader , PDFPlumberLoader , AmazonTextractLoader , UnstructuredPDFLoader etc

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r'D:\langchain\RAG\documentloader\main.pdf')

docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)

