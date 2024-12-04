import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langfuse import Langfuse    
from langfuse.callback import CallbackHandler
from langchain_ollama import OllamaEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

langfuse_handler = CallbackHandler()
langfuse_handler.auth_check()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
pgvector_host = os.getenv("PGVECTOR_HOST", "localhost") 
pgvector_port = os.getenv("PGVECTOR_PORT", 5432)
pgvector_user = os.getenv("PGVECTOR_USER", "postgres")
pgvector_password = os.getenv("PGVECTOR_PASSWORD", "postgres")
pgvector_database = os.getenv("PGVECTOR_DATABASE", "llmdb")

llm = OllamaLLM(model="llama3.2", base_url=ollama_host)
embeddings = OllamaEmbeddings(model="llama3.2", base_url=ollama_host)
connection_url = f"postgresql+psycopg://{pgvector_user}:{pgvector_password}@{pgvector_host}:{pgvector_port}/{pgvector_database}"

prompt_template = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3-4 sentences.

Context: {context}

Question: {input}

Helpful Answer:
"""

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = []
    for page in loader.load():
        pages.append(page)
    return pages

def store_documents_in_pgvector(pages):
    global embeddings

    vector_store = PGVector.from_documents(
        documents=pages,
        embedding=embeddings,
        collection_name="mydocs",
        connection=connection_url
    )
    return vector_store

def sendPrompt(prompt):
    global llm
    global embeddings

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="mydocs",
        connection=connection_url
    )

    retriever = vector_store.as_retriever()

    retrieval_qa_chat_prompt = PromptTemplate.from_template(prompt_template)

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    result = rag_chain.invoke({"input": prompt}, config={"callbacks":[langfuse_handler]})
    return result["answer"]


def process_file(file_path):
     pages = load_pdf(file_path)
     st.write(f"Pages generated for file: {file_path}")

     vector_store = store_documents_in_pgvector(pages)
     st.write(f"Embeddings generated for file: {file_path}")


async def async_process_file(file_path):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, process_file, file_path)
    return result


st.title("RAG app demo")

uploaded_files = st.file_uploader(
    "Choose PDF file(s)", accept_multiple_files=True
)

if st.button("Generate embeddings"):
    for uploaded_file in uploaded_files:
         file_content = uploaded_file.read()
         st.write("filename:", uploaded_file.name)

         folder_path = "./uploaded_files"
         os.makedirs(folder_path, exist_ok=True)

         file_path = os.path.join(folder_path, uploaded_file.name)
         with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

         result = asyncio.run(async_process_file(file_path))

         st.success(f"File processed successfully: {file_path}")


user_input = st.text_area("Enter your input:", "", height=150)

if st.button("Submit"):
    if user_input.strip():
        response = f"You entered: {user_input}"
        
        response = sendPrompt(user_input)
        st.write("Prompt Response:")
        st.success(response)
    else:
        st.warning("Please enter some input before submitting.")
