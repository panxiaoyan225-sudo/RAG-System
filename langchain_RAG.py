# run the following command to pull the models:
# ollama pull llama3
# ollama pull nomic-embed-text
# run the following command to install the dependencies:

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import RetrievalQA

# 1. Load Documents
# Scans your local directory for .txt files.

loader = DirectoryLoader('./my_data/', glob="./*.txt", loader_cls=TextLoader)
docs = loader.load()

# 2. Chunking for Precision
# Keeps context together with overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)


# 3. Local Embeddings & Vector Store
# Uses Ollama's local embedding model (nomic-embed-text)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 4. Initialize Local LLM (Ollama)
llm = ChatOllama(model="llama3", temperature=0)

# 5. Build RAG Chain
# Connects the retriever to the LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3})
)

# 6. Query the System
query = "What are the main functions of the OpenClaw framework?"
response = qa_chain.invoke({"query": query})

print(f"Answer: {response['result']}")