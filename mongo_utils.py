 
'''
file contains functions for interacting with the Chroma vector store, 
which is essential for our RAG system's retrieval capabilities.
'''

'''
# Here, we import necessary modules and initialize our text splitter, 
# embedding function, and Chroma vector store. The RecursiveCharacterTextSplitter is used 
# to split documents into manageable chunks, while OpenAIEmbeddings provides the embedding function for our documents.
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
#from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os

# Initialize text splitter and embedding function
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)

#embedding_function = OpenAIEmbeddings()
# Initialize Chroma vector store
#vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
import os 
from dotenv import load_dotenv 
load_dotenv() # This loads the environment
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings

# Create embeddings for documents and store them in a vector store
#vectorstore = SKLearnVectorStore.from_documents(
#    documents=doc_splits, embedding=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),)
#retriever = vectorstore.as_retriever(k=4)


# Document Loading and Splitting
# This function handles loading different document types (PDF, DOCX, HTML) and splitting them into chunks. 
# It uses the appropriate loader based on the file extension and then applies our text 
# splitter to create manageable document chunks.
def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    documents = loader.load()
    return text_splitter.split_documents(documents)


# Indexing Documents
#This function takes a file path and a file ID, loads and splits the document, 
# adds metadata (file ID) to each split, and then adds these document chunks to our Chroma vector store. 
# The metadata allows us to link vector store entries back to our database records.
def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    try:
        splits = load_and_split_document(file_path)
        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id
        #vectorstore.add_documents(splits)
        vectorstore = SKLearnVectorStore.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),)
        #return True
        return vectorstore
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False


# Deleting Documents
# This function deletes all document chunks associated with a given file ID 
# from the Chroma vector store. It first retrieves the documents to confirm their existence, then performs the deletion.
def delete_doc_from_chroma(file_id: int):
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")
        vectorstore._collection.delete(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id}")
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False

'''

################################################################################################################

import os
from pymongo import MongoClient
from dotenv import load_dotenv 

load_dotenv() # This loads the environment

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
MONGODB_URL = os.getenv("MONGODB_URL")
mongodb_client = MongoClient(MONGODB_URL, appname="fastAPItest")

# Now configure LLMs and embeddings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings 

# Create MongoDB Atlas Vector Store 
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex, StorageContext
from pymongo.errors import OperationFailure 

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool

# they are big chunks so let's split them
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
# split using tiktoken tokenizer from openAI 
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name ="cl100k_base", keep_separator=False, chunk_size=200, chunk_overlap=30
)

#Here are the LlamaIndex Settings
Settings.embed_model =OpenAIEmbedding(
    model = "text-embedding-3-small",
    dimensions=256,
    embed_batch_size=10,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
llm=OpenAI(model="gpt-4o", temperature=0)


DB_NAME = "fastAPItest"
COLLECTION_NAME = "tests1"
VS_INDEX_NAME = "vector_index"
FTS_INDEX_NAME = "fts_index"
collection = mongodb_client[DB_NAME][COLLECTION_NAME]

# we are doing hybrid search. so we need both vector and full text search
vector_store=MongoDBAtlasVectorSearch(
    mongodb_client, 
    db_name=DB_NAME,
    collection_name = COLLECTION_NAME,
    vector_index_name=VS_INDEX_NAME,
    fulltext_index=FTS_INDEX_NAME,
    embedding_key="embedding",
    text_key="text",
)
#vector_store_context=StorageContext.from_defaults(vector_store=vector_store)
#vector_store_index=VectorStoreIndex.from_documents(
#    llama_documents, storage_context=vector_store_context, show_progress=True)
#vector_store_index = VectorStoreIndex.from_documents(storage_context=vector_store_context, show_progress=True)


# function to take in a large document and return chunks
#def split_texts(texts:List[str])->List[str]:
#    chunked_texts = []
#    for text in texts:
#        chunks = text_splitter.create_documents([text])
#        chunked_texts.extend([chunk.page_content for chunk in chunks])
#    return chunked_texts

def load_and_split_document(file_path: str): # -> List[Document]:
    if file_path.endswith('.pdf'):
        #loader = PyPDFLoader(file_path)
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    elif file_path.endswith('.docx'):
        #loader = Docx2txtLoader(file_path)
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    elif file_path.endswith('.html'):
        #loader = UnstructuredHTMLLoader(file_path)
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    #documents = loader.load()
    #return split_texts(docs)
    return text_splitter.split_documents(docs)

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    try:
        splits = load_and_split_document(file_path)
        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id
        vector_store.add_documents(splits)
        #vector_store_context.add_vector_store(splits)
        #vectorstore = SKLearnVectorStore.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False















