import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("API KEY") #open ai api key need to place here 

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory
)

def get_chat_response(user_input):
    response = conversation.predict(input=user_input)
    return response
