# The Overwatch 2 Coach chatbot helps users extract and explore video transcriptions to learn strategies, tips, and lore about their favorite Overwatch heroes.

from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

# Load our local environment variables.

from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings without any parameters because I will be using variables from .env

embeddings = OpenAIEmbeddings()

# Create vector database for the video transcripts.

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
        template="Answer the following question: {question} By searching the following transcript: {docs}. Your answers should be detailed. Don't hallucinate. If you don't know the answer, simply say 'I don't know'.Generate a well-structured, readable response using bullet points for key ideas, each starting on a new line, with sub-points where necessary, ensuring clear flow and concise information; avoid long paragraphs, make the text engaging and easy to follow, and use a professional but creative tone."
    )

    chain = prompt | llm | parser

    inputs = {
    "question": query,
    "docs": docs_page_content
    }

    response = chain.invoke(inputs)

    return response