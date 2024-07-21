from flask import Flask, render_template, jsonify, request
from src.helper import huggingface_embeddings
from langchain.vectorstores import Pinecone as PineconeStore
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage,AIMessage
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

embeddings = huggingface_embeddings()

pc = Pinecone(os.getenv("PINECONE_API_KEY"))
index = pc.Index("chatbot1")

index_name="chatbot1"

#Loading the index
docsearch=PineconeStore.from_existing_index(index_name, embeddings)

import os
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN3')
# print(HUGGINGFACEHUB_API_TOKEN)
from langchain_huggingface import HuggingFaceEndpoint
repo_id="mistralai/Mistral-7B-Instruct-v0.3"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128, temperature=0.7,huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

question_maker_prompt = ChatPromptTemplate.from_messages (
[
    ("system", instruction_to_system),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

question_chain = question_maker_prompt | llm

qa_prompt = ChatPromptTemplate.from_messages (
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder (variable_name="chat_history"),
        ("human", "{question}"),
    ])
def contextualized_question (input: dict):
    if input.get ("chat_history"):
        return question_chain
    else:
        return input["question"]

retriever=docsearch.as_retriever(search_kwargs={'k': 2})

retriever_chain = RunnablePassthrough.assign(context=contextualized_question | retriever) #|format_docs)



rag_chain = (
    retriever_chain
    |qa_prompt
    | llm
)
chat_history = []
memory = ConversationBufferMemory(memory_key="chat_history", output_key='result')
qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    # chain_type_kwargs=chain_type_kwargs,
    memory=memory)

@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    user_input = msg
    print(user_input)
    result=  qa({"query": user_input})['result']

    print("Response : ", result)
    return str(result)



if __name__ == '__main__':
    app.run(host="0.0.0.0")