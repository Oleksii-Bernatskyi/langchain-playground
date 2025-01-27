import warnings
warnings.filterwarnings('ignore')

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

model = ChatOpenAI(model="gpt-4o-mini")


messages = [
    ('system', 'You are comedian who tels really funny jokes in Tonny Hinchcliff style.'),
    ('human', 'Tell me joke about {chicken}'),
    ]
prompt_template = ChatPromptTemplate.from_messages(messages)

uper_out = RunnableLambda(lambda x: x.upper())
count_out = RunnableLambda(lambda x: f"{len(x.split())}\n{x}")

chain = prompt_template | model | StrOutputParser() | uper_out | count_out
result = chain.invoke({"chicken": "chicken"})
print(result)