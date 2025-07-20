import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="PDF Chatbot")
st.title("📄 Chatbot Built on GEMINI MODEL")

try:
    # Load the PDF file
    loader = PyPDFLoader("FGCarPolicy.pdf")  # Ensure this file exists in the same folder
    data = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)

    # Create embeddings using Google Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create vector store using Chroma
    from langchain_chroma import Chroma
import tempfile

# Create a temporary directory (won't persist, good for Streamlit Cloud)
temp_dir = tempfile.TemporaryDirectory()

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=temp_dir.name
)


    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, max_tokens=500)

    # Define system prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    # Set up chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Streamlit input box
    query = st.chat_input("Ask a question about the PDF document:")

    if query:
        # Retrieve relevant documents and generate a response
        context = retriever.invoke(query)
        response = llm.invoke(prompt.format(input=query, context=context))

        st.write("🤖", response.content)
        st.success("Response generated successfully.")
    else:
        st.info("Please enter a question to get a response.")

except Exception as e:
    st.error(f"🚨 An error occurred: {e}")
