from src.helper import load_pdf,text_split,huggingface_embeddings
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as PineconeStore
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
print(PINECONE_API_KEY)

extracted_data = load_pdf("data/")
text_chunks=text_split(extracted_data)
embeddings = huggingface_embeddings()


pc = Pinecone(os.getenv("PINECONE_API_KEY"))
index = pc.Index("corpus")

#Creating Embeddings for Each of The Text Chunks & storing
docsearch=PineconeStore.from_texts(texts=[t.page_content for t in text_chunks],
                                   embedding=embeddings,
                                   index_name='corpus')