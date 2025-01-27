import warnings
warnings.filterwarnings('ignore')

import os

from langchain_openai import ChatOpenAI

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_text_splitters.character import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_text_splitters.base import TextSplitter, TokenTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the directory containing text file, data base directory, embeddings and loader
current_path = os.path.dirname(os.path.abspath(__file__))
file_path =os.path.join(current_path, 'odyssey.txt')
persist_directory = os.path.join(current_path, "db", "chroma_db_sentence")
db_dir = os.path.join(current_path, 'db')

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

loader = TextLoader(file_path)
documents = loader.load()

# Function to create vector store
def create_vector_store(docs, store_name):
    persist_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persist_directory):
        print(f"--- Creating vector store {store_name} ---")
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persist_directory
        )
        print(f"--- Vector store {store_name} created ---")
    else:
        print(f"--- Vector store {store_name} already exists ---")

# Create differente splitters

# 1. Character-based text splitter
print("\n--- Character-based text splitter ---")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, "chroma_db_char")

# 2. Sentence-based text splitter
print("\n--- Sentence-based text splitter ---")
sentence_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sentence_docs = sentence_splitter.split_documents(documents)
create_vector_store(sentence_docs, "chroma_db_sentence")

# 3. Token-based text splitter
print("\n--- Token-based text splitter ---")
token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, "chroma_db_token")

# 4. Recursive character-based text splitter
print("\n--- Recursive character-based text splitter ---")
recursive_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
recursive_char_docs = recursive_char_splitter.split_documents(documents)
create_vector_store(recursive_char_docs, "chroma_db_recursive_char")

# 5. Custom text splitter
print("\n--- Custom text splitter ---")
class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split("\n\n")
    
custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs, "chroma_db_custom")

# Functon to query vector store
def query_vector_store(store_name,
                       query,
                       embedding_function,
                       search_type,
                       search_kwargs):
    
    persist_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persist_directory):
        print(f"--- Vector store {store_name} exists. Querying it ---")
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
            )
        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs

        )
        
        relevant_docs = retriever.invoke(query)

        print(f"\n--- Relevant documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs):
            print(f"Document {i+1}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print("Vector store does not exist\n")



query = "Who is Odysseus' wife?"

# Query each vector store
query_vector_store("chroma_db_char", query, embeddings, "similarity", {"k": 2})
query_vector_store("chroma_db_sentence", query, embeddings, "similarity", {"k": 2})
query_vector_store("chroma_db_token", query, embeddings, "similarity", {"k": 2})
query_vector_store(
    "chroma_db_recursive_char",
    query,
    embeddings,
    "mmr",
    {"k": 2, "fetch_k": 20, "lambda_mult":1}
)

