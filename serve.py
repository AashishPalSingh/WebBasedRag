from fastapi import FastAPI
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)


embeddings = OpenAIEmbeddings()
db2 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 1. Create prompt template
prompt = ChatPromptTemplate.from_template(
    """
Answer the following question based only on the provided context:
{input}
<context>
{context}
</context>


"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
document_chain


retriever = db2.as_retriever(search_type="similarity", search_kwargs={"k": 4})


##create chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)


## App definition
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain runnable interfaces",
)


@app.get("/health")
def read_root():
    return {"message": "Hello, FastAPI!"}


class Item(BaseModel):
    input: str


## Adding chain routes
# add_routes(app, retrieval_chain, path="/chain")
@app.post("/chain/invoke")
def create_item(item: Item):
    print(item)
    config = {
        "run_name": "personal-webbased-rag",
        "tags": ["rag"],
        "metadata": {"model": "groq-gemma"},
    }
    result = retrieval_chain.invoke({"input": item.input, **config})
    return result["answer"]


def setup_pipeline():
    loader = WebBaseLoader(
        "https://www.technia.com/en/user-experience/software/value-components/"
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory="./chroma_db"
    )


if __name__ == "__main__":

    CHROMA_DB_DIR = "chroma_db"
    # Run setup_pipeline only if Chroma DB does not exist
    if not os.path.exists(CHROMA_DB_DIR):
        print("Chroma DB not found, setting up pipeline...")
        setup_pipeline()
    else:
        print("Chroma DB exists, skipping setup_pipeline.")
    uvicorn.run(app, host="127.0.0.1", port=8000)
