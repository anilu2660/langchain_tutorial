# 🦜🔗 LangChain Learning Project

A comprehensive, hands-on exploration of the **LangChain** framework — covering Chat Models, Prompt Engineering, Output Parsers, Runnables, Chains, Retrieval-Augmented Generation (RAG), Vector Stores, Document Loaders, Text Splitters, and Tools.

---

## 📁 Project Structure

```
langchain/
├── .env                        # Environment variables (API keys)
├── requirements.txt            # Project dependencies
├── template.json               # Saved PromptTemplate for research paper summarizer
│
├── ChatModel/                  # Basic Chat Model usage
│   └── langchain1.py
│
├── EmbeddingModel/             # Text Embeddings & Semantic Similarity
│   └── langchain1.py
│
├── Runnables/                  # Core Runnable primitives (LCEL)
│   ├── runnable_sequence.py
│   ├── runnable_parallel.py
│   ├── runnable_passthrough.py
│   ├── runnable_branch.py
│   └── runnable_lambda.py
│
├── Tools/                      # LangChain Tools & Toolkits
│   └── tools_in_langchain.py
│
├── RAG/                        # Retrieval-Augmented Generation
│   ├── Textsplitter/           # Text splitting strategies
│   ├── books/                  # Sample book assets
│   ├── documentloader/         # Various document loaders
│   ├── retriever/              # Retriever examples
│   ├── vectorstores/           # Vector store (Chroma) usage
│   └── yt_chatbot_rag/         # Full YouTube RAG chatbot app
│
├── chatbot.py                  # Stateful multi-turn chatbot
├── conditional_chain.py        # Sentiment-based conditional branching chain
├── langchain_runnables.py      # From-scratch Runnable implementation
├── messages.py                 # LangChain message types demo
├── parallel_chain.py           # Parallel chain for note + quiz generation
├── prompt_generator.py         # Research paper summarization prompt template
├── prompt_ui.py                # Streamlit UI for research paper summarizer
├── sequential_chain.py         # Two-step sequential chain
├── simple_chain.py             # Basic prompt → model → parser chain
└── typeofparsers.py            # All output parser types (commented examples)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- An **OpenAI API key** (required for most examples)
- Optional: Anthropic / Google Gemini / HuggingFace API keys

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd langchain
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS/Linux
source myenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here       # optional
GOOGLE_API_KEY=your_google_genai_key_here       # optional
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here    # optional
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Core LangChain framework |
| `langchain-core` | Runnables, prompts, parsers, messages |
| `langchain-openai` | OpenAI Chat & Embedding integrations |
| `langchain-anthropic` | Anthropic Claude integration |
| `langchain-google-genai` | Google Gemini integration |
| `langchain-huggingface` | HuggingFace models integration |
| `langchain-community` | Community tools, loaders, vector stores |
| `python-dotenv` | `.env` file loading |
| `streamlit` | Web UI for interactive apps |
| `youtube_transcript_api` | Fetch YouTube captions |
| `numpy` / `scikit-learn` | Numerical computation & cosine similarity |

---

## 🧩 Module Descriptions

---

### 🤖 ChatModel — `ChatModel/langchain1.py`

Demonstrates the most basic usage of LangChain's chat interface: initializing `ChatOpenAI` and invoking it with a plain string query.

```python
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
response = model.invoke("What is the capital of india?")
print(response.content)
```

**Concepts:** `ChatOpenAI`, `.invoke()`, temperature settings

---

### 📐 EmbeddingModel — `EmbeddingModel/langchain1.py`

Shows how to generate **vector embeddings** for documents and a query, then use **cosine similarity** to find the most semantically similar document.

```python
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)
scores = cosine_similarity([query_embedding], doc_embeddings)[0]
```

**Concepts:** `OpenAIEmbeddings`, `embed_documents()`, `embed_query()`, cosine similarity ranking

---

### 💬 Messages — `messages.py`

Illustrates LangChain's **message types**: `SystemMessage`, `HumanMessage`, and `AIMessage`, which are used to build structured conversation history.

**Concepts:** `SystemMessage`, `HumanMessage`, `AIMessage`, multi-role prompt construction

---

### 🤖 Chatbot — `chatbot.py`

A terminal-based, **stateful multi-turn chatbot** that maintains a running `chat_history` list across turns. The model plays the role of a poet.

```python
chat_history = [SystemMessage(content="you are poet.")]
while True:
    user_input = input("You: ")
    # ...appends and invokes model with full history
```

**Concepts:** Stateful conversation, `chat_history`, `HumanMessage`, `AIMessage`, loop-based interaction

---

### ✍️ Simple Chain — `simple_chain.py`

The entry point to **LangChain Expression Language (LCEL)**. Uses the pipe operator `|` to compose a `PromptTemplate → ChatOpenAI → StrOutputParser` chain.

```python
chain = prompt | model | parser
result = chain.invoke({'topic': 'AI'})
chain.get_graph().print_ascii()
```

**Concepts:** LCEL, `PromptTemplate`, `StrOutputParser`, chain composition, ASCII graph visualization

---

### 🔄 Sequential Chain — `sequential_chain.py`

A **two-step sequential chain** where the output of the first step (detailed report) feeds into the second step (5-bullet summary) automatically.

```python
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({'topic': 'India'})
```

**Concepts:** Multi-step LCEL, output chaining between prompts

---

### ⚡ Parallel Chain — `parallel_chain.py`

Uses `RunnableParallel` to run two independent chains (notes generation and Q&A quiz generation) **simultaneously**, then merges the results with a third chain.

```python
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz':  prompt2 | model2 | parser
})
chain = parallel_chain | merge_chain
```

**Concepts:** `RunnableParallel`, concurrent execution, result merging, multi-model usage

---

### 🌿 Conditional Chain — `conditional_chain.py`

An advanced chain that first **classifies feedback sentiment** (positive/negative) using a Pydantic parser, then **branches** to different response prompts using `RunnableBranch`.

```python
classifier_chain = prompt1 | model | PydanticOutputParser(...)
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    ...
)
chain = classifier_chain | branch_chain
```

**Concepts:** `PydanticOutputParser`, `RunnableBranch`, `RunnableLambda`, structured output schemas

---

### 🧬 LangChain Runnables — `langchain_runnables.py`

A deep-dive implementation that **builds the Runnable interface from scratch** to understand how LangChain chains work internally. Implements `NakliLLM`, `NakliPromptTemplate`, `NakliStrOutputParser`, and a `RunnableConnector`.

**Concepts:** Abstract base classes, custom Runnable implementations, `invoke()` interface, chain composition internals

---

### 🔧 Output Parsers — `typeofparsers.py`

A commented reference file demonstrating all major LangChain output parser types:

| Parser | Description |
|---|---|
| `StrOutputParser` | Raw string output |
| `JsonOutputParser` | Unstructured JSON output |
| `StructuredOutputParser` | JSON with a response schema |
| `PydanticOutputParser` | JSON validated against a Pydantic model |

**Concepts:** `get_format_instructions()`, `partial_variables`, Pydantic `BaseModel` + `Field`

---

### 🖥️ Prompt Generator — `prompt_generator.py` & `prompt_ui.py`

- **`prompt_generator.py`** — Creates and **saves a rich PromptTemplate** to `template.json` for summarizing research papers with configurable style and length.
- **`prompt_ui.py`** — A **Streamlit web app** that loads the saved template and presents an interactive UI for summarizing landmark AI research papers (e.g., "Attention Is All You Need", "BERT", "GPT-3").

```bash
streamlit run prompt_ui.py
```

**Concepts:** `PromptTemplate.save()`, `load_prompt()`, Streamlit `selectbox`, multi-variable prompts

---

### ⚙️ Runnables — `Runnables/`

Focused examples of each core LCEL Runnable primitive:

| File | Runnable | Description |
|---|---|---|
| `runnable_sequence.py` | `RunnableSequence` | Explicit sequential chaining |
| `runnable_parallel.py` | `RunnableParallel` | Concurrent branch execution |
| `runnable_passthrough.py` | `RunnablePassthrough` | Pass input unchanged through chain |
| `runnable_branch.py` | `RunnableBranch` | Conditional routing between runnables |
| `runnable_lambda.py` | `RunnableLambda` | Wrap any Python function as a runnable |

---

### 🛠️ Tools — `Tools/tools_in_langchain.py`

Comprehensive reference for LangChain Tools covering three approaches:

1. **Built-in tools** — `DuckDuckGoSearchRun` (web search), `ShellTool` (run shell commands)
2. **Custom tools** — Using the `@tool` decorator with type-hinted functions
3. **BaseTool + Pydantic** — Class-based structured tool with Pydantic input validation
4. **Toolkits** — Grouping multiple tools into a `Toolkit` class

**Concepts:** `@tool` decorator, `BaseTool`, `args_schema`, `invoke()`, tool `.name`, `.description`, `.args`

---

### 📚 RAG — `RAG/`

The most advanced section, covering the complete **Retrieval-Augmented Generation** pipeline.

#### 📄 Document Loaders — `RAG/documentloader/`

| File | Loader | Source |
|---|---|---|
| `text_loader.py` | `TextLoader` | `.txt` files |
| `pypdf_loader.py` | `PyPDFLoader` | PDF files |
| `csv_loader.py` | `CSVLoader` | CSV files |
| `WebBaseLoader.py` | `WebBaseLoader` | Web pages (BeautifulSoup) |
| `Directory_loader.py` | `DirectoryLoader` | All files in a directory |

#### ✂️ Text Splitters — `RAG/Textsplitter/`

| File | Strategy | Description |
|---|---|---|
| `length_based.py` | `CharacterTextSplitter` | Split by character count |
| `text_structuredbased.py` | `RecursiveCharacterTextSplitter` | Split by text structure (paragraphs, sentences) |
| `document_structuredbased.py` | `MarkdownHeaderTextSplitter` | Split Markdown by headers |
| `semantic_meaningbased.py` | `SemanticChunker` | Split by semantic similarity |

#### 🗄️ Vector Stores — `RAG/vectorstores/chroma_vectordb.py`

Demonstrates full **Chroma vector store** lifecycle:
- Creating and adding `Document` objects with metadata
- Performing **similarity search**
- **Updating** documents by ID
- **Deleting** documents

```python
vector_store = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory='my_chroma_db')
vector_store.add_documents(docs)
vector_store.similarity_search(query='who among these are a batsman?', k=3)
```

**Concepts:** `Chroma`, `Document`, `persist_directory`, CRUD operations on vector stores

#### 🎥 YouTube RAG Chatbot — `RAG/yt_chatbot_rag/`

A **full end-to-end RAG pipeline** built on YouTube video transcripts:

| File | Description |
|---|---|
| `rag_using_langchain.py` | Core RAG script — transcript fetch → chunk → embed → FAISS → QA chain |
| `stremalit_app.py` | Full Streamlit web app with chat UI |
| `README.md` | Dedicated documentation for the chatbot |

**Pipeline:**
```
YouTube Video ID
   → YouTubeTranscriptApi (fetch captions)
   → RecursiveCharacterTextSplitter (chunk transcript)
   → OpenAIEmbeddings (vectorize)
   → FAISS (vector store)
   → Retriever (similarity search, k=4)
   → RunnableParallel (context + question)
   → PromptTemplate + ChatOpenAI + StrOutputParser
   → Answer
```

```bash
streamlit run RAG/yt_chatbot_rag/stremalit_app.py
```

**Concepts:** `YouTubeTranscriptApi`, `FAISS`, `RunnablePassthrough`, `RunnableLambda`, `RunnableParallel`, full RAG QA chain

---

## 🏃 Running Examples

### Terminal Scripts

```bash
python simple_chain.py
python sequential_chain.py
python parallel_chain.py
python conditional_chain.py
python chatbot.py
python ChatModel/langchain1.py
python EmbeddingModel/langchain1.py
python Runnables/runnable_sequence.py
python RAG/vectorstores/chroma_vectordb.py
python RAG/yt_chatbot_rag/rag_using_langchain.py
```

### Streamlit Apps

```bash
# Research Paper Summarizer
streamlit run prompt_ui.py

# YouTube RAG Chatbot
streamlit run RAG/yt_chatbot_rag/stremalit_app.py
```

---

## 🗺️ Learning Path

If you are new to LangChain, follow this recommended order:

```
1. messages.py              → Message types
2. ChatModel/langchain1.py  → Basic ChatModel
3. simple_chain.py          → LCEL basics
4. sequential_chain.py      → Multi-step chains
5. typeofparsers.py         → Output parsers
6. parallel_chain.py        → Parallel execution
7. conditional_chain.py     → Conditional branching
8. langchain_runnables.py   → Runnables internals
9. Runnables/               → Runnable primitives
10. EmbeddingModel/         → Embeddings & similarity
11. RAG/documentloader/     → Loading documents
12. RAG/Textsplitter/       → Splitting strategies
13. RAG/vectorstores/       → Vector store operations
14. RAG/yt_chatbot_rag/     → Full RAG application
15. Tools/                  → Tools & Toolkits
16. chatbot.py              → Stateful chatbot
17. prompt_ui.py            → Streamlit UI
```

---

## ⚠️ Important Notes

- **API Keys** — Never commit your `.env` file or hardcode API keys in source files.
- **`myenv/`** — The virtual environment folder is local and should be added to `.gitignore`.
- **`rag_using_langchain.py`** — Contains a hardcoded API key (line 2); replace it with the `.env`-based approach before sharing.
- **`tools_in_langchain.py`** — Originally a Colab notebook; the `!pip install` lines should be removed when running locally.
- **`RAG/retriever/`** — Currently an empty placeholder directory for future retriever examples.

---

## 📄 License

This project is intended for educational purposes. Feel free to use and adapt the examples for your own LangChain learning journey.
