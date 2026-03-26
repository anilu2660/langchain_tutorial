
from langchain_text_splitters import RecursiveCharacterTextSplitter


text = '''
       Major space missions in 2026 focus on returning humans to the Moon (NASA’s Artemis II), exploring Jupiter’s moon Europa (Europa Clipper), and developing commercial space stations (Axiom Space). These efforts, alongside India's upcoming Chandrayaan-4/5 lunar sample returns, aim to strengthen long-term exploration of the solar system.
'''

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
    
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)