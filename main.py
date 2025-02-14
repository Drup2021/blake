from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import asyncio
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from langchain_huggingface import HuggingFaceEmbeddings
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from pinecone import Pinecone as pince, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from dotenv import load_dotenv

# ----------------------------
# Create FastAPI app and rate limiter
# ----------------------------
app = FastAPI()

# Set up a rate limiter: 10 requests per minute per client IP
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# Custom exception handler for rate limits
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )

# ----------------------------
# Configure external API keys
# ----------------------------
# Google Generative AI API key and configuration
load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# ----------------------------
# Define Pydantic model for request
# ----------------------------
class QueryRequest(BaseModel):
    query: str

# ----------------------------
# Global variables for models and vector store
# ----------------------------
embedding_model = None
vectorstore = None
gemini_model = None

# ----------------------------
# Startup event: Connect to existing Pinecone index and initialize models
# ----------------------------
@app.on_event("startup")
async def startup_event():
    global embedding_model, vectorstore, gemini_model

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")
    
    # Connect to the existing Pinecone index
    index_name = "testingcsv"
    pc = pince(api_key=os.environ["PINECONE_API_KEY"])
    
    # Verify that the index exists
    if index_name not in pc.list_indexes().names():
        raise RuntimeError(
            f"Pinecone index '{index_name}' does not exist. Please create and populate it first."
        )
    
    # Load the vector store from the existing index.
    # This method avoids re-indexing any documents.
    vectorstore = LC_Pinecone.from_existing_index(index_name, embedding_model)
    
    # Initialize the Gemini generative model
    gemini_model = genai.GenerativeModel("gemini-pro")

# ----------------------------
# Helper function to retrieve relevant documents via similarity search
# ----------------------------
async def get_relevant_documents(query: str, k: int = 3) -> str:
    """
    Uses the vectorstore's similarity_search to retrieve the top k documents.
    Runs in a separate thread to avoid blocking the event loop.
    """
    docs = await asyncio.to_thread(vectorstore.similarity_search, query, k)
    # Combine the content from retrieved documents (adjust as needed)
    return "\n\n".join([doc.page_content for doc in docs])

# ----------------------------
# Helper function to query Gemini with exponential backoff
# ----------------------------
async def query_gemini(context: str, user_query: str) -> str:
    """
    Calls the Gemini API using the provided context and user query.
    Implements exponential backoff if a ResourceExhausted (429) error is raised.
    """
    max_attempts = 5
    attempt = 0
    delay = 1  # initial delay in seconds


    #print(f"THE CONTEXT IS : {context}")
    # Construct the prompt
    prompt = f"""
    You are a helpful assistant. Use the following context to answer the user's query.
    Remove any newline (\\n) and tab (\\t) characters in your answer.
    Context:
    {context}
    User Query:
    {user_query}
    """

    while attempt < max_attempts:
        try:
            await asyncio.sleep(0.5)  # Short non-blocking delay
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        except ResourceExhausted as e:
            attempt += 1
            #print(f"Resource exhausted on attempt {attempt}/{max_attempts}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff

    raise HTTPException(
        status_code=500,
        detail="Failed to generate content after multiple attempts due to quota/resource exhaustion."
    )

# ----------------------------
# /ask endpoint: Combine similarity search and Gemini response generation
# ----------------------------
@app.post("/ask")
@limiter.limit("10/minute")  # Apply rate limiting: 10 requests per minute per IP
async def ask_question(request: Request, query_request: QueryRequest):
    user_query = query_request.query
    
    # Retrieve relevant context using similarity search
    context = await get_relevant_documents(user_query)
    
    # Query Gemini using the context and user query
    response_text = await query_gemini(context, user_query)
    
    return {"response": response_text}
