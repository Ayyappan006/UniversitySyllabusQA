import streamlit as st
import os

COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
COHERE_USER_AGENT = st.secrets["COHERE_USER_AGENT"]

# Manually ensure COHERE_USER_AGENT is set in environment
if "COHERE_USER_AGENT" not in os.environ:
    os.environ["COHERE_USER_AGENT"] = COHERE_USER_AGENT

# Optional debug prints
print("COHERE_API_KEY:", os.getenv("COHERE_API_KEY"))
print("COHERE_USER_AGENT:", os.getenv("COHERE_USER_AGENT"))

from utils import load_pdf, chunk_and_embed, create_qa_chain
import streamlit as st


st.title("🎓 University Syllabus QA Bot with Cohere")

uploaded_file = st.file_uploader("Upload your syllabus (PDF format only)", type=["pdf"])

if uploaded_file:
    with open("uploaded_syllabus.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.info("🔄 Processing and indexing syllabus...", icon="ℹ️")
    docs = load_pdf("uploaded_syllabus.pdf")
    chunk_and_embed(docs)
    st.success("✅ Syllabus indexed successfully! Now ask me anything.")

query = st.text_input("Ask your question about the syllabus")

if st.button("Ask") and query:
    qa_chain = create_qa_chain()
    response = qa_chain.invoke({"query": query})
    st.subheader("📘 Answer:")
    st.write(response)
