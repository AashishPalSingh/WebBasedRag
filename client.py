import requests
import streamlit as st


def get_groq_response(input_text):

    json_body = {"input": input_text}
    response = requests.post("http://127.0.0.1:8000/chain/invoke", json=json_body)

    print(response.json())

    return response.json()


## Streamlit app
st.title("RAG over technia value components site")
input_text = st.text_input("Enter the question")

if input_text:
    st.write(get_groq_response(input_text))
