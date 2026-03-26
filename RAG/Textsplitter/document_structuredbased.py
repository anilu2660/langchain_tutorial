from langchain_text_splitters import RecursiveCharacterTextSplitter ,Language

text = '''
class Dog:
    # Constructor to initialize attributes
    def __init__(self, name, breed):
        self.name = name   # Instance attribute
        self.breed = breed

    # Method representing behavior
    def bark(self):
        return f"} says Woof!"

# Creating an object (instantiation)
my_dog = Dog("Buddy", "Golden Retriever")
print(my_dog.bark())  # Output: Buddy says Woof!

'''

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=175,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[1])