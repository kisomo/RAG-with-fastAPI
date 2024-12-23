
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os
from mongo_utils import vector_store #, index_document_to_chroma

#retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
#output_parser = StrOutputParser()

query_engine=vector_store.as_query_engine(
    similarity_top_k=5, vector_store_query_mode="hybrid",alpha=0.7) #,filters=filters)
#response=query_engine.query(query)


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


def get_rag_chain(model="gpt-4o-mini"):
    llm = ChatOpenAI(model=model)
    #history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    history_aware_retriever = create_history_aware_retriever(llm, query_engine, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)    
    return rag_chain

