import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
import chromadb
import pypdf
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load API key from .env and set up Groq client
# Groq runs LLaMA 3 — Meta's powerful open-source model
# It's free, fast, and works from anywhere including India
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# This embedding model runs 100% locally on your computer
# No internet needed for this part — completely free forever
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------------------
# Extract all text from a PDF, page by page
# -----------------------------------------------
def extract_text_from_pdf(file_path):
    reader = pypdf.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        # Some PDFs have scanned images instead of text
        # extract_text() handles normal text-based PDFs
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# -----------------------------------------------
# Split text into overlapping chunks
# Why overlap? So sentences at chunk boundaries
# don't lose their context and meaning
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
# Build a ChromaDB vector store from chunks
# Each chunk gets converted to a vector (list of
# numbers) that captures its meaning mathematically
# -----------------------------------------------
def build_vector_store(chunks):
    chroma_client = chromadb.Client()

    # Always start fresh so re-uploading a PDF works cleanly
    try:
        chroma_client.delete_collection("pdf_chunks")
    except:
        pass

    collection = chroma_client.create_collection("pdf_chunks")

    # encode() converts all chunks to vectors at once
    # .tolist() converts numpy arrays to plain Python lists
    embeddings = embedder.encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))]
    )
    return collection

# -----------------------------------------------
# Find the top_k chunks most relevant to a question
# This is the "retrieval" part of RAG —
# we search by meaning, not just keyword matching
# -----------------------------------------------
def retrieve_relevant_chunks(collection, question, top_k=3):
    question_embedding = embedder.encode([question]).tolist()[0]
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    return results["documents"][0]

# -----------------------------------------------
# Ask LLaMA 3 a question using the retrieved context
# We only pass the relevant chunks, not the whole PDF
# This keeps the prompt focused and accurate
# -----------------------------------------------
def ask_question(collection, question, chat_history):
    # Step 1: find the most relevant parts of the PDF
    relevant_chunks = retrieve_relevant_chunks(collection, question)
    context = "\n\n".join(relevant_chunks)

    # Step 2: build conversation history so the model
    # remembers what was said earlier in the chat
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

    # Add previous turns so the model has memory of the conversation
    for turn in chat_history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Add the new question at the end
    messages.append({"role": "user", "content": question})

    # Step 3: send everything to LLaMA 3 via Groq
    # llama-3.3-70b-versatile is a very powerful free model
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0  # 0 = factual and consistent answers
    )
    return response.choices[0].message.content


# -----------------------------------------------
# Streamlit UI — everything the user sees
# -----------------------------------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")
st.write("Upload any PDF and ask it anything — powered by LLaMA 3 via Groq (free!)")

# session_state persists data across Streamlit reruns
# Without this, variables reset every time the user types
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "collection" not in st.session_state:
    st.session_state.collection = None

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and st.session_state.collection is None:
    with st.spinner("Reading your PDF and building the knowledge base..."):

        # Save uploaded file temporarily so pypdf can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Full RAG pipeline: extract → chunk → embed → store
        text = extract_text_from_pdf(tmp_path)
        chunks = split_into_chunks(text)
        st.session_state.collection = build_vector_store(chunks)
        os.unlink(tmp_path)  # Clean up the temp file from disk

    st.success(f"PDF ready! ({len(chunks)} chunks indexed). Ask your first question!")

# Replay all previous messages so chat looks continuous
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input at the bottom of the screen
if st.session_state.collection:
    user_input = st.chat_input("Ask something about your PDF...")
    if user_input:
        # Show user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Get LLaMA's answer and display it
        with st.chat_message("assistant"):
            with st.spinner("LLaMA 3 is thinking..."):
                answer = ask_question(
                    st.session_state.collection,
                    user_input,
                    st.session_state.chat_history[:-1]  # everything except current question
                )
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("Please upload a PDF to get started.")

