from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def embed_And_Store(split):
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store=Chroma.from_documents(documents=split,embedding=embeddings)
    retriever=vector_store.as_retriever()
    return retriever