from langchain_community.document_loaders import PyPDFLoader
from typing_extensions import Concatenate
from PyPDF2 import PdfReader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

loader = PdfReader("D:\\Projects\\chatbot\\files\\car_insurance.pdf")
try:
    raw = ""
    for i, page in enumerate(loader.pages):
        content = page.extract_text()
        if content:
            raw += content + "\n\n"
            print(r"\n")
    text_splitter = CharacterTextSplitter(
        chunk_size=800, chunk_overlap=200, separator="\n\n", length_function=len
    )
    texts = text_splitter.split_text(raw)
    len(texts)

    # 4. Initialize the embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text", temperature=0.1)

    # 4. Create and populate the vector store
    vector_store = Chroma.from_texts(texts, embeddings)

    # Optional: Verify it works by testing a query search
    query = "till what date this insurance is valid?"
    docs = vector_store.similarity_search(query, k=2)

    context_text = "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the provided context:\n\n"
        "Context:\n{context_text}\n\n"
        "Question: {question}"
    )

    # 7. Create a clean QA Prompt Template
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the provided context:\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    llm = OllamaLLM(model="llama2", temperature=0.9, model_kwargs={"num_predict": 1000})
    modern_chain = prompt | llm | StrOutputParser()

    print("Final Answer:********************")
    response = modern_chain.invoke({"context": context_text, "question": query})
    print(response)


except Exception as e:
    print(f"An error occurred: {e}")
