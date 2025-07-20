import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile

# Load .env file
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="PDF Chatbot")
st.title("📄 Chatbot Built on GEMINI MODEL")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load and process PDF
        loader = PyPDFLoader(tmp_path)
        data = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(data)

        # Embedding
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Vector DB
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        # LLM (Gemini)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, max_tokens=500)

        # Prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of context to answer the question. "
            "If you don't know the answer, say you don't know. Keep it short.\n\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Chat input
        query = st.chat_input("Ask a question about the PDF document:")

        if query:
            context = retriever.invoke(query)
            response = llm.invoke(prompt.format(input=query, context=context))
            st.write("🤖", response.content)
            st.success("Response generated successfully.")
        else:
            st.info("Please enter a question.")

    except Exception as e:
        st.error(f"🚨 Error: {e}")
else:
    st.info("📄 Please upload a PDF file to get started.")
