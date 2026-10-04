import streamlit as st
import tempfile
from rag.loader import load_csv
import os
from dotenv import load_dotenv
from rag.embeddings import embed_And_Store
load_dotenv()
from rag.text_splitter import text_split
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
st.title("INFLUENCER SEARCH")

uploaded_file=st.file_uploader("Upload your file",
                               type=["csv"])

#here we create a tempfile with the same data as uploaded_file because Pypdfloader needs the whole
#file path, not just file name.

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".csv"
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        csv_path=temp_file.name
    document=load_csv(csv_path)
    chunks=text_split(document)
    retriever=embed_And_Store(chunks)
    st.session_state.retriever = retriever
    st.write("Agent is ready for queries")

#get the model
from langchain_groq import ChatGroq
groq_api_key= os.environ.get("GROQ_API_KEY")
model=ChatGroq(model="openai/gpt-oss-20b",api_key=groq_api_key)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt=ChatPromptTemplate.from_messages([
    ("system",
        "You are a friend and also a helpful assistant "
        "provide answers with the best of your abilities. "
        "Read the csv and provide the name of all the influencers who matches the requirements of the user. Try to provide as much information as possible. If you are not sure about the answer, say 'I don't know'.  "
        "\n\n"
        "{context}"),
    ("human","{input}")
])
context_prompt=ChatPromptTemplate.from_messages([
    ("system",
        "Rewrite the user's question using the chat history. "
        "Do not answer the question."
    ),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
])

from langchain_core.output_parsers import StrOutputParser
parser=StrOutputParser()

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

#how to make sure one session is not affecting the other session, we can use the 
# following code snippet to test it out.
#here BaseChatMessageHistory is an interface which just provides the behavior and this is 
# implemented/ inherited by chatmessageHistory
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "store" not in st.session_state:
    st.session_state.store = {}


def get_session_history(
    session_id: str
) -> BaseChatMessageHistory:

    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()

    return st.session_state.store[session_id]

from operator import itemgetter
contextualize_chain = (
    {
        "chat_history": itemgetter("chat_history"),
        "input": itemgetter("input")
    }
    | context_prompt
    | model
    | parser
)

from langchain_core.runnables import RunnablePassthrough

if st.session_state.retriever is not None:
    retriever=st.session_state.retriever
    chain=(
        {
            "context": contextualize_chain | retriever,
            "input": itemgetter("input")
        }
        | prompt
        | model 
        | parser
    )
    agent = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    #display previous messages

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    #user question
    question = st.chat_input(
        "Ask something about the uploaded CSV..."
    )


    if question:
        with st.chat_message("user"):
            st.write(question)

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        with st.chat_message("assistant"):

            with st.spinner("Finding answer..."):

                response = agent.invoke(
                    {
                        "input": question
                    },
                    config={
                        "configurable": {
                            "session_id": "user_1"
                        }
                    }
                )

            st.write(response)
            st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        

# The user uploads a CSV, which is loaded into documents, split into smaller chunks, 
# converted into embeddings, and stored in Chroma to create a retriever; when the user
# asks a question, agent combines the question with previous chat history,
#  the contextualization chain uses the LLM to reframe the question, the reframed question 
#  is sent to the retriever to fetch relevant documents, those documents become the context
#  while the original question is passed through using itemgetter("input"), and finally the
#   context and question are given to the final prompt and Groq LLM, whose response is
#   converted into the final answer by StrOutputParser.
