import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load .env variables
load_dotenv()

st.set_page_config(page_title="PDF Chatbot")
st.title("📄 Chatbot Built on GEMINI MODEL")

try:
    # Load and split the PDF
    loader = PyPDFLoader("FGCarPolicy.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)

    # Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # ✅ Use FAISS instead of Chroma
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Gemini model setup
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, max_tokens=500)

    # Prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of context to answer the question concisely.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    query = st.chat_input("Ask a question about the PDF document:")

    if query:
        context = retriever.invoke(query)
        response = llm.invoke(prompt.format(input=query, context=context))
        st.write("🤖", response.content)
        st.success("Response generated successfully.")
    else:
        st.info("Please enter a question to get a response.")

except Exception as e:
    st.error(f"🚨 Error: {e}")
