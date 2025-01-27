import warnings
warnings.filterwarnings('ignore')

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

model = ChatOpenAI(model="gpt-4o-mini")

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'you are helpful assistant'),
        ('human', 'Generate a thank you note for this positive feedback: {feedback}'),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'you are helpful assistant'),
        ('human', 'Generate a response addressing this negative feedback: {feedback}'),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'you are helpful assistant'),
        ('human', 'Generate a request for more datails for this neutral feedback: {feedback}'),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'you are helpful assistant'),
        ('human', 'Generate a message to escalate this feedback to a human agent: {feedback}'),
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'you are helpful assistant'),
        ('human', 
         'Classify the santiment of this feedback as positive, negative, neuteral, or escalate: {feedback}'),
    ]
)

branches = RunnableBranch(
    (
        lambda x: 'positive' in x, positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: 'negative' in x, negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: 'neutral' in x, neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()

chain = classification_chain #| branches

# review = 'This product is exellent. I really enjoyed using it and found it very helpful!'
# review = 'This product is terrible. It broke just after one use and the quality is very poor.'
review = 'Meh.. cant say its bad but also cant recomend it.'

result = chain.invoke({'feedback': review})

print(result)