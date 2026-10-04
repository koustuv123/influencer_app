import streamlit as st
import pandas as pd
import tempfile
import os

from dotenv import load_dotenv

# ============================================================
# LANGCHAIN / LANGGRAPH
# ============================================================

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# ============================================================
# YOUR EXISTING RAG MODULES
# ============================================================

from rag.loader import load_csv
from rag.text_splitter import text_split
from rag.embeddings import embed_And_Store


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    os.environ["HF_TOKEN"] = hf_token


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Influencer Search",
    page_icon="📊",
    layout="wide"
)

st.title("📊 INFLUENCER SEARCH")


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "agent" not in st.session_state:
    st.session_state.agent = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# MODEL
# ============================================================

# SAME MODEL YOU WERE USING BEFORE
model = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=groq_api_key
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your Instagram influencer CSV",
    type=["csv"]
)


# ============================================================
# FUNCTION TO CREATE TOOLS
# ============================================================

def create_tools(df, retriever):

    # ========================================================
    # TOOL 1: FILTER INFLUENCERS
    # ========================================================

    @tool
    def filter_influencers(
        min_followers: int = 0,
        genre: str = "",
        n: int = 20
    ):
        """
        Find influencers using exact structured filters.

        Parameters:
        min_followers:
            Minimum number of followers.

        genre:
            Genre of the influencer, such as comedy,
            fashion, fitness, travel, etc.

        n:
            Maximum number of influencers to return.

        Use this tool for exact filtering questions.
        """

        try:

            # Make a copy
            result = df.copy()

            # ------------------------------------------------
            # FILTER BY FOLLOWERS
            # ------------------------------------------------

            if min_followers > 0:

                result = result[
                    result["followers"] >= min_followers
                ]


            # ------------------------------------------------
            # FILTER BY GENRE
            # ------------------------------------------------

            if genre:

                result = result[
                    result["genre"]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    == genre.lower().strip()
                ]


            # ------------------------------------------------
            # CHECK EMPTY RESULT
            # ------------------------------------------------

            if result.empty:

                return (
                    "No influencers match the given criteria."
                )


            # ------------------------------------------------
            # LIMIT RESULTS
            # ------------------------------------------------

            result = result.head(n)


            # ------------------------------------------------
            # ONLY RETURN USEFUL COLUMNS
            # ------------------------------------------------

            columns = [
                "creator_name",
                "instagram_handle",
                "genre",
                "followers",
                "avg_reel_views",
                "engagement_rate_pct"
            ]

            # Only select columns that actually exist
            available_columns = [
                column
                for column in columns
                if column in result.columns
            ]

            result = result[available_columns]


            # ------------------------------------------------
            # RETURN COMPACT RESULT
            # ------------------------------------------------

            return result.to_string(index=False)

        except Exception as e:

            return (
                f"Error while filtering influencers: {str(e)}"
            )


    # ========================================================
    # TOOL 2: TOP INFLUENCERS
    # ========================================================

    @tool
    def top_influencers(
        metric: str = "avg_reel_views",
        n: int = 10
    ):
        """
        Return the top N influencers based on a numerical metric.

        Valid metrics:

        followers
        avg_reel_views
        avg_likes
        avg_comments
        avg_shares
        avg_saves
        engagement_rate_pct
        view_rate_pct
        """

        allowed_metrics = [
            "followers",
            "avg_reel_views",
            "avg_likes",
            "avg_comments",
            "avg_shares",
            "avg_saves",
            "engagement_rate_pct",
            "view_rate_pct"
        ]


        # ------------------------------------------------
        # VALIDATE METRIC
        # ------------------------------------------------

        if metric not in allowed_metrics:

            return (
                f"Invalid metric: {metric}. "
                f"Valid metrics are: {allowed_metrics}"
            )


        try:

            # ------------------------------------------------
            # SORT AND LIMIT
            # ------------------------------------------------

            result = (
                df.sort_values(
                    by=metric,
                    ascending=False
                )
                .head(n)
            )


            # ------------------------------------------------
            # ONLY RETURN USEFUL COLUMNS
            # ------------------------------------------------

            columns = [
                "creator_name",
                "instagram_handle",
                "genre",
                "followers",
                "avg_reel_views",
                "engagement_rate_pct"
            ]

            available_columns = [
                column
                for column in columns
                if column in result.columns
            ]

            result = result[available_columns]


            return result.to_string(index=False)

        except Exception as e:

            return (
                f"Error while finding top influencers: {str(e)}"
            )


    # ========================================================
    # TOOL 3: SEMANTIC SEARCH
    # ========================================================

    @tool
    def search_influencers(query: str):
        """
        Search influencers using semantic/vector search.

        Use this for questions involving:

        - genre
        - sub-genre
        - descriptions
        - content
        - topics
        - general semantic similarity

        Do NOT use this for exact numerical filtering.
        """

        try:

            docs = retriever.invoke(query)


            if not docs:

                return "No relevant influencers found."


            results = []

            for doc in docs:

                results.append(
                    doc.page_content
                )


            return "\n\n".join(results)

        except Exception as e:

            return (
                f"Error during semantic search: {str(e)}"
            )


    # ========================================================
    # RETURN TOOLS
    # ========================================================

    return [
        filter_influencers,
        top_influencers,
        search_influencers
    ]


# ============================================================
# CREATE AGENT
# ============================================================

def create_influencer_agent(df, retriever):

    tools = create_tools(
        df,
        retriever
    )


    # ========================================================
    # CHECKPOINTER
    # ========================================================

    checkpointer = InMemorySaver()


    # ========================================================
    # LANGCHAIN AGENT
    # ========================================================

    agent = create_agent(

        # SAME MODEL
        model=model,

        # TOOLS
        tools=tools,

        # SYSTEM PROMPT
        system_prompt="""

You are an Instagram influencer analytics assistant.

You have access to a CSV containing Instagram influencer data.

Your job is to answer the user's questions accurately
using the available tools.


============================================================
1. EXACT FILTERING
============================================================

For exact numerical or categorical filtering,
ALWAYS use filter_influencers.

Examples:

User:
"Give me influencers with more than 10000 followers"

Use:

filter_influencers(
    min_followers=10000
)


User:
"Give me influencers from the comedy genre"

Use:

filter_influencers(
    genre="comedy"
)


User:
"Give me 20 influencers with more than 10000 followers
and genre should be comedy"

Use:

filter_influencers(
    min_followers=10000,
    genre="comedy",
    n=20
)


============================================================
2. NUMBER OF RESULTS
============================================================

If the user asks for a specific number:

"Give me 20 influencers"

set:

n=20

Do not return more than the requested number.


============================================================
3. RANKING
============================================================

For questions such as:

- Who has the most followers?
- Give me the top 10 influencers by followers.
- Who has the highest average reel views?
- Give me the influencers with the highest engagement.

Use:

top_influencers


Example:

top_influencers(
    metric="followers",
    n=10
)


============================================================
4. SEMANTIC SEARCH
============================================================

Use search_influencers for semantic/content-based questions.

Examples:

- Find influencers related to travel.
- Find creators whose content is about technology.
- Find influencers similar to fitness creators.

Do NOT use semantic search for exact numerical filtering.


============================================================
5. RETURN ALL REQUESTED MATCHES
============================================================

If the user asks for 20 influencers,
return up to 20 matching influencers.

If fewer than 20 influencers match,
return all available matches.


============================================================
6. DO NOT INVENT DATA
============================================================

Only use information available in the CSV
or returned by the tools.

Never invent:

- influencer names
- followers
- engagement rates
- views
- genres
- handles


============================================================
7. NO MATCHES
============================================================

If no influencers match the criteria,
clearly tell the user that no matching influencers
were found.


============================================================
8. RESPONSE
============================================================

Keep responses concise and useful.

When appropriate, include:

- Creator name
- Instagram handle
- Genre
- Followers
- Average reel views
- Engagement rate

Do not unnecessarily repeat large amounts of raw data.

""",

        # MEMORY
        checkpointer=checkpointer
    )


    return agent


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    # Only process when a new file is uploaded
    if st.session_state.file_name != uploaded_file.name:

        with st.spinner(
            "Loading influencer data..."
        ):

            # ------------------------------------------------
            # SAVE UPLOADED FILE
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".csv"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                csv_path = temp_file.name


            # ------------------------------------------------
            # LOAD DATAFRAME
            # ------------------------------------------------

            df = pd.read_csv(
                csv_path
            )

            st.session_state.df = df


            # ------------------------------------------------
            # LOAD DOCUMENTS
            # ------------------------------------------------

            document = load_csv(
                csv_path
            )


            # ------------------------------------------------
            # SPLIT DOCUMENTS
            # ------------------------------------------------

            chunks = text_split(
                document
            )


            # ------------------------------------------------
            # CREATE VECTOR STORE / RETRIEVER
            # ------------------------------------------------

            retriever = embed_And_Store(
                chunks
            )

            st.session_state.retriever = retriever


            # ------------------------------------------------
            # CREATE LANGCHAIN AGENT
            # ------------------------------------------------

            agent = create_influencer_agent(
                df,
                retriever
            )

            st.session_state.agent = agent


            # ------------------------------------------------
            # SAVE FILE NAME
            # ------------------------------------------------

            st.session_state.file_name = (
                uploaded_file.name
            )


        st.success(
            "✅ Agent is ready for queries"
        )


# ============================================================
# DISPLAY DATASET INFORMATION
# ============================================================

if st.session_state.df is not None:

    df = st.session_state.df


    col1, col2, col3 = st.columns(3)


    # ------------------------------------------------
    # TOTAL INFLUENCERS
    # ------------------------------------------------

    with col1:

        st.metric(
            "Total Influencers",
            len(df)
        )


    # ------------------------------------------------
    # TOTAL FOLLOWERS
    # ------------------------------------------------

    with col2:

        st.metric(
            "Total Followers",
            f"{df['followers'].sum():,.0f}"
        )


    # ------------------------------------------------
    # AVERAGE ENGAGEMENT
    # ------------------------------------------------

    with col3:

        st.metric(
            "Average Engagement",
            f"{df['engagement_rate_pct'].mean():.2f}%"
        )


# ============================================================
# CHAT
# ============================================================

if st.session_state.agent is not None:

    st.divider()

    st.subheader(
        "💬 Ask about your influencers"
    )


    question = st.chat_input(
        "Example: Give me 20 comedy influencers with more than 10000 followers"
    )


    if question:

        # ------------------------------------------------
        # USER MESSAGE
        # ------------------------------------------------

        with st.chat_message("user"):

            st.write(
                question
            )


        # ------------------------------------------------
        # ASSISTANT
        # ------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    # ----------------------------------------
                    # INVOKE AGENT
                    # ----------------------------------------

                    response = (
                        st.session_state.agent.invoke(

                            {
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": question
                                    }
                                ]
                            },

                            config={
                                "configurable": {
                                    "thread_id": "user_1"
                                }
                            }
                        )
                    )


                    # ----------------------------------------
                    # GET FINAL ANSWER
                    # ----------------------------------------

                    answer = (
                        response["messages"][-1].content
                    )


                    st.write(
                        answer
                    )


                except Exception as e:

                    st.error(
                        f"Something went wrong: {str(e)}"
                    )


# ============================================================
# BEFORE FILE UPLOAD
# ============================================================

else:

    st.info(
        "👆 Upload an influencer CSV file to start asking questions."
    )
