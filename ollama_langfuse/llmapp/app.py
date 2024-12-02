import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langfuse import Langfuse    
from langfuse.callback import CallbackHandler

import os

langfuse_handler = CallbackHandler()
langfuse_handler.auth_check()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
llm = OllamaLLM(model="llama3.2", base_url=ollama_host)

def sendPrompt(prompt):
    global llm
    response = llm.invoke(prompt, config={"callbacks":[langfuse_handler]})
    return response

st.title("Input and Prompt Handler")

user_input = st.text_area("Enter your input:", "", height=150)

if st.button("Submit"):
    if user_input.strip():
        response = f"You entered: {user_input}"
        
        response = sendPrompt(user_input)
        st.write("Prompt Response:")
        st.success(response)
    else:
        st.warning("Please enter some input before submitting.")
