# PDF Chatbot — Chat with any PDF using AI

I built this because I was tired of scrolling through long PDFs trying to find specific information. Just upload any PDF and ask it questions in plain English — it finds the relevant parts and answers you instantly.

## What it does

You upload a PDF, and the app reads through it, breaks it into chunks, and builds a searchable knowledge base from it. When you ask a question, it finds the most relevant parts of your document and sends them to an AI model (LLaMA 3, running via Groq) to generate a clear, grounded answer. It also remembers your previous questions in the same session, so you can have a real back-and-forth conversation with your document.

The key thing here is that the AI only answers from your PDF — it won't hallucinate or make things up from general knowledge. If the answer isn't in the document, it tells you so.

## How it works (the RAG pipeline)

This project is built on a technique called Retrieval-Augmented Generation (RAG). Here's the flow:

1. **Extract** — the PDF text is extracted page by page using `pypdf`
2. **Chunk** — the text is split into overlapping chunks of ~800 characters so no context is lost at boundaries
3. **Embed** — each chunk is converted into a vector (a list of numbers representing its meaning) using `sentence-transformers`, which runs fully locally with no API needed
4. **Store** — the vectors are stored in ChromaDB, an in-memory vector database
5. **Retrieve** — when you ask a question, it gets embedded the same way and the most semantically similar chunks are pulled out
6. **Answer** — the relevant chunks + your question are sent to LLaMA 3 via Groq API, which generates the final answer

## Tech stack

- **Streamlit** — the web interface
- **pypdf** — PDF text extraction
- **sentence-transformers** — local embeddings (`all-MiniLM-L6-v2` model)
- **ChromaDB** — vector store for similarity search
- **Groq API** — fast, free inference for LLaMA 3.3 70B
- **python-dotenv** — environment variable management

## Getting started

Clone the repo and navigate into it, then create and activate a virtual environment.

```bash
git clone https://github.com/YOUR_USERNAME/pdf-chatbot.git
cd pdf-chatbot
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux
```

Install the dependencies.

```bash
pip install -r requirements.txt
```

Create a `.env` file in the root folder and add your Groq API key. You can get one for free at https://console.groq.com.

```
GROQ_API_KEY=your_key_here
```

Run the app.

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`, upload any PDF, and start asking questions.

## Project structure

```
pdf-chatbot/
├── app.py          # main application — all the RAG logic and UI
├── .env            # your API key (never commit this to GitHub)
├── requirements.txt
└── README.md
```

## Things I learned building this

Chunk size matters a lot more than I expected. Too small and you lose context, too large and the retrieval gets noisy. 800 characters with 100-character overlap hit a good balance for most documents. Running embeddings locally with sentence-transformers also turned out to be faster and more reliable than calling an external embedding API — and it's completely free.

## What's next

- Add support for multiple PDFs at once
- Show which page of the PDF the answer came from
- Add a document summary feature on upload
- Persistent vector store so you don't re-process the same PDF every session

---

Built by Adarsh Pandey
