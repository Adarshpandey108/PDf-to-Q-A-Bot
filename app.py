import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
import faiss
import numpy as np
import pypdf
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load API key from .env and set up Groq client
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Local embedding model — runs on your computer, no API needed
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------------------
# Extract all text from a PDF, page by page
# -----------------------------------------------
def extract_text_from_pdf(file_path):
    reader = pypdf.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# -----------------------------------------------
# Split text into overlapping chunks
# Overlap ensures context isn't lost at boundaries
# -----------------------------------------------
def split_into_chunks(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# -----------------------------------------------
# Build a FAISS vector store from chunks
# FAISS is faster and more stable than ChromaDB
# Each chunk gets converted to a vector using
# sentence-transformers running locally
# -----------------------------------------------
def build_vector_store(chunks):
    # Convert all chunks to vectors at once
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    # Create a FAISS index using L2 (Euclidean) distance
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add all chunk vectors to the index
    index.add(embeddings)

    # Return both the index and original chunks
    # We need the chunks later to retrieve the actual text
    return index, chunks

# -----------------------------------------------
# Find the chunks most relevant to a question
# by comparing vector similarity using FAISS
# -----------------------------------------------
def retrieve_relevant_chunks(index_and_chunks, question, top_k=3):
    index, chunks = index_and_chunks

    # Convert the question to a vector
    question_embedding = np.array(
        embedder.encode([question])
    ).astype("float32")

    # Search the FAISS index for the top_k closest chunks
    _, indices = index.search(question_embedding, top_k)

    # Return the actual text of the matched chunks
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# -----------------------------------------------
# Ask LLaMA 3 a question using the PDF context
# We only pass relevant chunks, not the whole PDF
# -----------------------------------------------
def ask_question(index_and_chunks, question, chat_history):
    # Step 1: find the most relevant parts of the PDF
    relevant_chunks = retrieve_relevant_chunks(index_and_chunks, question)
    context = "\n\n".join(relevant_chunks)

    # Step 2: build the full message list for the model
    # including system instructions and chat history
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions strictly "
                "based on the PDF document provided below. Use only the context "
                "to answer. If the answer is not in the context, say clearly: "
                "'I could not find that in the document.'\n\n"
                f"Context from PDF:\n{context}"
            )
        }
    ]

    # Add previous conversation turns so the model remembers
    for turn in chat_history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Add the new question
    messages.append({"role": "user", "content": question})

    # Step 3: send to LLaMA 3 via Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


# -----------------------------------------------
# Streamlit UI
# -----------------------------------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")
st.write("Upload any PDF and ask it anything — powered by LLaMA 3 via Groq!")

# Session state persists data across Streamlit reruns
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index_and_chunks" not in st.session_state:
    st.session_state.index_and_chunks = None

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and st.session_state.index_and_chunks is None:
    with st.spinner("Reading your PDF and building the knowledge base..."):

        # Save uploaded file temporarily so pypdf can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Full RAG pipeline: extract → chunk → embed → store
        text = extract_text_from_pdf(tmp_path)
        chunks = split_into_chunks(text)
        st.session_state.index_and_chunks = build_vector_store(chunks)
        os.unlink(tmp_path)

    st.success(f"PDF ready! ({len(chunks)} chunks indexed). Ask your first question!")

# Replay all previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if st.session_state.index_and_chunks:
    user_input = st.chat_input("Ask something about your PDF...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("LLaMA 3 is thinking..."):
                answer = ask_question(
                    st.session_state.index_and_chunks,
                    user_input,
                    st.session_state.chat_history[:-1]
                )
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("Please upload a PDF to get started.")
