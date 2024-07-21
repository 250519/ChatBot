prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer with detail below and nothing else.
Helpful answer:
"""

instruction_to_system = '''
Given a chat history and the latest user question, which might reference context in the chat history, reformulate the latest user question into a standalone question that can be understood without the chat history. Do NOT answer the question; only reformulate it if needed and otherwise return it as is. Reformulate it only once per user input.
'''

qa_system_prompt = '''You are a helpful assistant for question-answering tasks. 
Use the provided context to answer the question accurately. 
If you don't know the answer, simply state that you don't know; do not attempt to fabricate an answer. 
{context}'''