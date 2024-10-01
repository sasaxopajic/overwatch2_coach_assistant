# The Overwatch 2 Coach chatbot helps users extract and explore video transcriptions to learn strategies, tips, and lore about their favorite Overwatch heroes.

# First thing I do, is importing the Youtube loader 

# # pip install --upgrade --quiet  youtube-transcript-api

from langchain_community.document_loaders import YoutubeLoader

# Import text splitter, because the videos can be up to 1 hour long so the transcripts will be large.

# # pip install -qU langchain-text-splitters

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import OpenAI LLM for which I need an OPENAI_API_KEY that is stored as an environmental variable.

from langchain_openai import ChatOpenAI

# I will also need OpenAI embeddings.

# # pip install langchain-openai

from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain.chains import  LLMChain

# I will be using FAISS library for vector stores.

from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

# Firstly, let's load our local environment variables.

from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings without any parameters because I will be using variables from .env

embeddings = OpenAIEmbeddings()

# Now I create vector database for the video transcripts.

def create_vector_database_from_yt_url(video_url: str) -> FAISS:
    # Load and transcript the youtube video    
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    # Very large transcript of the video is being splitted into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    # Initialize database. FAISS does similarity search
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=5):
    # 4-o mini can handle up tp 8k tokens but I will send 5k
    docs = db.similarity_search(query, k=k)
    # Join all documents
    docs_page_content = " ".join([d.page_content for d in docs])
    
    # Set GPT to 4-mini
    llm = ChatOpenAI(model_name="gpt-4o-mini")

    prompt = PromptTemplate(
        input_variables=["question", docs],
        template="After watching the video, what are some key strategies and gameplay tips to effectively utilize this hero? Please include detailed strategies on positioning, optimal ability usage, and synergy with teammates. Discuss how to counter common enemy heroes and share tips for adapting to different map environments. Additionally, offer insights on improving overall gameplay mechanics with this hero, including decision-making during team fights and when to engage or disengage. Answer the following question: {question} By searching the following transcript: {docs}. Your answers should be detailed. Don't hallucinate. If you don't know the answer, simply say 'I don't know' "
    )

    chain = prompt | llm | parser

    inputs = {
    "question": query,
    "docs": docs_page_content
    }

    response = chain.invoke(inputs)

    return response