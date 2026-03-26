from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='D:\langchain\RAG\documentloader\data.csv')

docs = loader.load()

print(len(docs))
print(docs[300])