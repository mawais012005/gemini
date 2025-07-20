import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile  # ✅ Must be outside of try

# ✅ Load environment variables from .env
load_dotenv()

# ✅ Streamlit page config
st.set_page_config(page_title="PDF Chatbot")
st.title("📄 Chatbot Built on GEMINI MODEL")

try:
    # ✅ Load and parse the PDF
    loader = PyPDFLoader("FGCarPolicy.pdf")  # Make sure this file exists in the same directory
    data = loader.load()

    # ✅ Split PDF into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)

    # ✅ Generate embeddings using Google Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # ✅ Temporary vector store (safe for Streamlit Cloud)
    temp_dir = tempfile.TemporaryDirectory()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=temp_dir.name
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # ✅ Setup Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, max_tokens=500)

    # ✅ Define system prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say you don't know. Keep the answer short.\n\n{context}"
    )

    # ✅ Chat template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # ✅ User question input
    query = st.chat_input("Ask a question about the PDF document:")

    if query:
        # Retrieve context and generate response
        context = retriever.invoke(query)
        response = llm.invoke(prompt.format(input=query, context=context))

        st.write("🤖", response.content)
        st.success("Response generated successfully.")
    else:
        st.info("Please enter a question to get a response.")

except Exception as e:
    st.error(f"🚨 An error occurred: {e}")
