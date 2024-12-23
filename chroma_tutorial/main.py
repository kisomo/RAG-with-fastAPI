
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8)

#loader = PyPDFLoader("data/Hypothesis.pdf")
#docs = loader.load_and_split()

chroma_db = Chroma(persist_directory="data", embedding_function=embeddings, collection_name="lc_chroma_demo")
#collection = chroma_db.get()
docs = chroma_db.get()

#chroma_db = Chroma.from_documents(
#    documents=docs, 
#    embedding=embeddings, 
#    persist_directory="data", 
#    collection_name="lc_chroma_demo"
#)

query = "What is this document about?"
docs = chroma_db.similarity_search(query)
# The chain_type="stuff" lets LangChain take the list of matching documents 
# from the retriever (Chroma DB in our case), insert everything all into a prompt, and pass it over to the llm.
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=chroma_db.as_retriever())
response = chain(query)

print(response)

'''
#Update a document in the collection
#We can also filter the results based on metadata that we assign to documents using the where parameter.
#To demonstrate this, I am going to add a tag to the first document called demo, update the database,
# and then find vectors tagged as demo:

# assigning custom tag metadata to first document
docs[0].metadata = {
    "tag": "demo"
}
# updating the vector store
chroma_db.update_document(
    document=docs[0],
    document_id=collection['ids'][0]
)
# using the where parameter to filter the collection
collection = chroma_db.get(where={"tag" : "demo"})
'''


'''
# Delete a collection
#The delete_collection() simply removes the collection from the vector store.
#  Here's a quick example showing how you can do this:
chroma_db.delete_collection()
'''