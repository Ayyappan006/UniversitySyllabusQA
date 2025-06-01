import os
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS


load_dotenv()  # Load env vars from .env

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_USER_AGENT = os.getenv("COHERE_USER_AGENT")

def get_embeddings():
    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY environment variable not set")
    return CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-v3.0")


def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()

def chunk_and_embed(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    embeddings = get_embeddings()  # uses updated get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("syllabus_index")

def create_qa_chain():
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local("syllabus_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    
    llm = ChatCohere(model="command-r", cohere_api_key=COHERE_API_KEY)  # Use chat-compatible model
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

