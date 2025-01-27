import warnings
warnings.filterwarnings('ignore')

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

def run_search(query: str, ) -> str:
        """use the tool blah"""
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY")

        client = TavilyClient(api_key=api_key)
        # response = client.get_search_context(query=query)
        response = client.qna_search(query=query)
        return f"Search results for {query}:\n\n\n{response}\n"

query = "Google Brains achievements"
search_res = run_search(query)

messages = [
    ('system', 'Given input and output of a search query provide consice summary of output in bullet points'),
    ('human', '{input}'),
    ]
prompt_template = ChatPromptTemplate.from_messages(messages)

llm = ChatOpenAI(model="gpt-4o-mini")


chain = prompt_template | llm | StrOutputParser() 
result = chain.invoke({"input": search_res})

print(result)