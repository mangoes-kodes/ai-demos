# import requests
# import json
# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import gptmodel
# import streamlit as st
# import os
# from dotenv import load_dotenv


# load_dotenv()

# # First API call with reasoning
# response = requests.post(
#   url="https://openrouter.ai/api/v1/chat/completions",
#   headers={
#     "Authorization": f"Bearer {os.getenv('GPTOSS20BMODEL')}",
#     "Content-Type": "application/json",
#   },
#   data=json.dumps({
#     "model": "openai/gpt-oss-20b:free",
#     "messages": [
#         {
#           "role": "user",
#           "content": "How many r's are in the word 'strawberry'?"
#         }
#       ],
#     "reasoning": {"enabled": True}
#   })
# )
 
# print("\n", response.status_code) 
# response = response.json()
# print("\n", response) 
# print("\n") 
# response = response['choices'][0]['message']

# messages = [
#   {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
#   {
#     "role": "assistant",
#     "content": response.get('content'),
#     "reasoning_details": response.get('reasoning_details') 
#   },
#   {"role": "user", "content": "Are you sure? Think carefully."}
# ]


import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.6
)

chat_history = []

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))
    response = llm.invoke(chat_history)
    print("Bot:", response.content)
    chat_history.append(AIMessage(content=response.content))