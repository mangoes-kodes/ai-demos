from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

demo_template = """You are a kind and polite medical advisor who listens to patients' concerns and provides helpful advice.
Also provide options to consider for treatment for {illness}."""

prompt_template = PromptTemplate(
    input_variables=["illness"],
    template=demo_template,
)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
gpt_model_env = os.getenv("GPTOSS20BMODEL")
if gpt_model_env:
    os.environ["GPTOSS20BMODEL"] = gpt_model_env

st.title("LangChain Demo with Ollama")
input_text = st.text_input("Enter the illness/topic")

llm = Ollama(model="llama2", temperature=0.9)
output_parser = StrOutputParser()
chain = prompt_template | llm | output_parser

if input_text.strip():
    st.write(chain.invoke({"illness": input_text.strip()}))

# minor changes