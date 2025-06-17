import streamlit as st
import os
from dotenv import load_dotenv

# Set environment variables to avoid PyTorch issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# --- LLM and Embeddings Imports ---
from langchain_openai import AzureChatOpenAI # For Azure OpenAI LLM
# from langchain_community.embeddings import HuggingFaceEmbeddings # Will be imported inside the function
from langchain_community.vectorstores import FAISS # For in-memory vector store

# --- LangChain Core Components ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.retrievers.multi_query import MultiQueryRetriever # For MultiQueryRetriever
from langchain_core.documents import Document # For Document type hinting

# --- API Key & Azure OpenAI Configuration Management ---
load_dotenv()

def get_azure_openai_config():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    return api_key, azure_endpoint, azure_deployment, openai_api_version

api_key, azure_endpoint, azure_deployment, openai_api_version = get_azure_openai_config()

if not all([api_key, azure_endpoint, azure_deployment, openai_api_version]):
    st.error("❌ Azure OpenAI configuration missing. Please ensure all necessary environment variables are set:")
    st.info("💡 `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`, `AZURE_OPENAI_API_VERSION`")
    st.info("💡 Create a `.env` file in the same folder as `bot2.py` and add these variables.")
    st.stop()

# Debug: Display configuration (hide sensitive info)
st.sidebar.write("🔧 **Configuration Debug:**")
st.sidebar.write(f"- Endpoint: {azure_endpoint}")
st.sidebar.write(f"- Deployment: {azure_deployment}")
st.sidebar.write(f"- API Version: {openai_api_version}")
st.sidebar.write(f"- API Key: {'✅ Set' if api_key else '❌ Missing'}")

# Set environment variables for AzureChatOpenAI
os.environ["AZURE_OPENAI_API_KEY"] = api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint

# --- Streamlit App Title ---
st.title("Metayb HR Assistant")

# --- Initialize Main LLM (Cached) ---
@st.cache_resource
def initialize_llm():
    """Initializes and caches the main LLM for answer generation using Azure OpenAI."""
    return AzureChatOpenAI(
        azure_deployment=azure_deployment,
        api_version=openai_api_version,
        temperature=0.1,
    )

# Test Azure OpenAI connection
@st.cache_data
def test_azure_connection():
    """Test the Azure OpenAI connection"""
    try:
        test_llm = AzureChatOpenAI(
            azure_deployment=azure_deployment,
            api_version=openai_api_version,
            temperature=0.1,
        )
        response = test_llm.invoke([("human", "Hello, can you respond with just 'Connection successful'?")])
        return True, response.content
    except Exception as e:
        return False, str(e)

# Test connection on sidebar
connection_success, connection_msg = test_azure_connection()
if connection_success:
    st.sidebar.success("✅ Azure OpenAI Connection: Working")
else:
    st.sidebar.error(f"❌ Azure OpenAI Connection: Failed")
    st.sidebar.error(f"Error: {connection_msg}")

llm = initialize_llm()

# --- Prompt Template (Optimized for Confidence and Inference) ---
prompt = ChatPromptTemplate.from_template("""
You are an expert HR assistant. Your goal is to provide comprehensive and helpful answers based on the HR policy context provided.

**Instructions:**
1.  **Strictly use only the provided context.** Do not bring in outside knowledge.
2.  If the user asks for information that can be *calculated or derived* from the context (e.g., monthly from yearly data, total from components), please perform that calculation or derivation and present the answer confidently.
3.  If the exact answer is not explicitly stated, but related information is available, summarize or synthesize that related information clearly.
4.  If the information is genuinely not present or cannot be inferred/calculated from the context, then and only then state: "I don't have enough information on that specific detail in the provided policies."

<context>
{context}
</context>

Question: {input}
""")

# --- Document Embedding and Vector Store Setup (Cached) ---
@st.cache_resource
def vector_embedding():
    """
    Loads, splits, embeds documents using Hugging Face, and creates a FAISS vector store.
    This function is cached to prevent re-processing on every Streamlit rerun.
    """
    try:
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        # Import HuggingFaceEmbeddings with proper error handling
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as e:
            st.error(f"❌ Failed to import HuggingFaceEmbeddings: {e}")
            return None
        
        hr_docs_path = "policy_docs" 
        
        if not os.path.isdir(hr_docs_path):
            raise FileNotFoundError(f"Document folder not found. Expected: '{hr_docs_path}'")

        loader = PyPDFDirectoryLoader(hr_docs_path)
        docs = loader.load()

        if not docs:
            raise ValueError(f"No documents found in: {hr_docs_path}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = splitter.split_documents(docs)

        # Initialize Hugging Face embeddings model with explicit device mapping
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Create an in-memory FAISS vector store from the documents and embeddings
        vectorstore = FAISS.from_documents(final_documents, embeddings)
        
        vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        return {
            'vectorstore_retriever': vectorstore_retriever,
            'doc_count': len(docs),
            'chunk_count': len(final_documents),
            'sources': list(set([d.metadata.get('source', 'Unknown Source') for d in docs]))
        }

    except Exception as e:
        st.error(f"❌ Error while loading documents: {e}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return None

# --- Auto-load documents on startup ---
if 'vector_data' not in st.session_state:
    st.session_state.vector_data = vector_embedding()

# --- Display Error if Documents Failed to Load ---
if not st.session_state.vector_data:
    st.error("❌ Failed to load documents. Please check your document folder and try refreshing the page.")
    st.info("💡 Make sure your PDF files are in the 'policy_docs' folder")
    st.stop()

# --- User Input Text Box ---
prompt1 = st.text_input("🔍 Ask a question about HR policies:", placeholder="e.g., What is the leave policy?")

# --- Handle User Query ---
if prompt1:
    if st.session_state.vector_data and st.session_state.vector_data['vectorstore_retriever']:
        with st.spinner("🤖 Generating answer..."):
            try:
                base_retriever = st.session_state.vector_data['vectorstore_retriever']

                # Fixed Azure OpenAI configuration for MultiQueryRetriever
                query_generator_llm = AzureChatOpenAI(
                    azure_deployment=azure_deployment,
                    api_version=openai_api_version,
                    temperature=0.1,
                )
                
                multiquery_retriever = MultiQueryRetriever.from_llm(
                    retriever=base_retriever,
                    llm=query_generator_llm,
                    include_original=True
                )

                top_docs = multiquery_retriever.invoke(prompt1)

                document_chain = create_stuff_documents_chain(llm, prompt)
                response = document_chain.invoke({"input": prompt1, "context": top_docs})

                st.write("**Answer:**")
                st.write(response)

            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")
                st.error("Please try rephrasing your question or contact support.")
                import traceback
                st.error(f"Full traceback: {traceback.format_exc()}")
    else:
        st.error("❌ Documents not loaded. Please refresh the page.")

# --- Footer with Tips ---
st.markdown("---")
st.markdown("💡 **Tip**: Ask specific questions about HR policies, procedures, benefits, or company guidelines for best results.")