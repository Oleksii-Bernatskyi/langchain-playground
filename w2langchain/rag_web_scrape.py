import warnings
warnings.filterwarnings('ignore')

import os

from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

current_path = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_path, 'db')
persist_directory = os.path.join(db_dir, "chroma_db_apple")

url = "https://www.apple.com/"

loader = WebBaseLoader(url)
documents = loader.load()

text_spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_spliter.split_documents(documents)



print("blah blah")

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

if not os.path.exists(persist_directory):
    print("blah")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    print("finish blah")
else:
    print("other blah")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

query = "What new products announsed on apple.com?"


relevant_docs = retriever.invoke(query)



print("blaaa")

for i, doc in enumerate(relevant_docs):
    print(f"Document {i+1}:\n{doc.page_content}\n")
