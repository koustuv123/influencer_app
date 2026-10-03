from langchain_text_splitters import RecursiveCharacterTextSplitter

def text_split(doc):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    split=text_splitter.split_documents(doc)
    return split