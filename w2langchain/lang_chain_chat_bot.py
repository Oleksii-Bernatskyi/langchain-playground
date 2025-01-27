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


model = ChatOpenAI(model="gpt-4o-mini")
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_type = "similarity", search_kwargs = {"k": 3})


system_prompt = (
    "Given chat history and latest user question "
    "which might reference context in the chat history, "
    "formulate a standalon question that can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
# This uses model to help formulate question based on chat history
history_aware_retriever = create_history_aware_retriever(
    model, retriever, question_prompt
)

# Instruction for the model to give answer based on given context
qa_system_prompt = (
    "You are an assistant for question answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you dont know the answer just say that you dont know. "
    " Use three sentances at max and keep answer consice."
    "\n\n"
    "{context}"
)

# Pormpt template for answering questions using chat history
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
# Create_stuff_documents_chain grab documents and pass them to the model
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def continual_chat():
    print("Start chatting with the AI, type exit to end the conversation.")
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})

        print(f"AI: {result['answer']}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))


if __name__ == "__main__":
    continual_chat()

