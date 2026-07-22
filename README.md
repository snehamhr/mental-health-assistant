# MindPal

MindPal is a supportive mental-health chatbot built with FastAPI, Groq, local semantic retrieval, and a simple web interface.

The application uses content from psychology books to retrieve relevant information and provide short, empathetic, context-aware responses.

## Features

- Supportive mental-health conversations
- Retrieval-Augmented Generation using psychology books
- Local semantic search with Sentence Transformers
- Groq-powered language model responses
- Conversation context within the current browser session
- Crisis-message detection and safety response
- Out-of-scope question handling
- Responsive web interface
- New Chat functionality
- No permanent storage of user conversations

## Technology Stack

### Backend

- Python
- FastAPI
- Uvicorn
- Groq API
- Pydantic

### Retrieval Pipeline

- PyMuPDF
- Sentence Transformers
- `all-MiniLM-L6-v2`
- NumPy
- Pickle

### Frontend

- HTML
- CSS
- JavaScript
- Browser `sessionStorage`

## How MindPal Works

```text
User message
    ↓
Safety and crisis check
    ↓
Conversation-aware retrieval query
    ↓
Local embedding generation
    ↓
Cosine-similarity search
    ↓
Relevant psychology-book chunks
    ↓
Groq language model
    ↓
Supportive response
Knowledge Base

The knowledge base is created from PDF books stored inside:

source_books/

The preprocessing pipeline:

Opens each PDF
Extracts text page by page
Cleans unnecessary whitespace and broken formatting
Divides the text into overlapping chunks
Generates normalized MiniLM embeddings
Saves the processed chunks and embeddings

Generated artifacts:

data/combined_chunks.pkl
data/embeddings.npy

Current knowledge-base configuration:

Embedding model: all-MiniLM-L6-v2
Embedding dimension: 384
Chunk size: approximately 900 characters
Chunk overlap: approximately 150 characters
Project Structure
mindpal-chatbot/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── llm.py
│   ├── main.py
│   ├── rag.py
│   ├── safety.py
│   ├── schemas.py
│   │
│   └── static/
│       ├── index.html
│       ├── styles.css
│       └── app.js
│
├── data/
│   ├── combined_chunks.pkl
│   └── embeddings.npy
│
├── scripts/
│   └── build_knowledge_base.py
│
├── source_books/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
Local Setup

Create and activate a virtual environment:

python -m venv .venv

Windows:

.venv\Scripts\activate

Install the required packages:

python -m pip install -r requirements.txt

For CPU-only Torch on Windows:

python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
Environment Variables

Create a .env file in the project root.

Example:

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant

EMBEDDING_MODEL=all-MiniLM-L6-v2

CHUNKS_PATH=data/combined_chunks.pkl
EMBEDDINGS_PATH=data/embeddings.npy

SIMILARITY_THRESHOLD=0.55
TOP_K_RESULTS=3
MEMORY_WINDOW=20
RETRIEVAL_HISTORY_TURNS=3
MAX_CONTEXT_CHARACTERS=6000

The actual .env file must not be committed to GitHub.

Build the Knowledge Base

Place PDF books inside:

source_books/

Then run:

python scripts/build_knowledge_base.py

The script creates:

data/combined_chunks.pkl
data/embeddings.npy

It also validates that the number of chunks matches the number of embedding rows.

Run the Application
python -m uvicorn app.main:app --reload

Open:

http://127.0.0.1:8000
Conversation Memory

MindPal does not permanently store conversations on the server.

Conversation history is stored temporarily using browser sessionStorage.

This means:

Refreshing the same browser tab keeps the current conversation
Clicking New Chat clears the current conversation
Closing the tab ends the stored session
Conversations are not shared between users
Conversations are not written to a backend database
Safety

MindPal includes a safety-routing layer for messages involving possible self-harm or immediate danger.

For crisis-related messages, the application provides immediate safety-oriented guidance before normal retrieval or language-model generation.

MindPal is not a replacement for:

A licensed mental-health professional
Medical diagnosis
Emergency services
Crisis intervention services
Scope

MindPal is designed to discuss topics such as:

Anxiety
Stress
Depression
Low mood
Anger
Emotional regulation
Coping
Self-esteem
Relationships
Feeling overwhelmed
General psychological concepts

Clearly unrelated questions may receive an out-of-scope response.

Limitations
Responses depend on the quality of the uploaded psychology books
PDF extraction may skip image-only or scanned pages
Retrieved content may include formatting artifacts from the original books
The chatbot may make mistakes
The application does not provide clinical diagnosis
Crisis detection is rule-based and may not identify every possible situation
Privacy

MindPal does not intentionally store chat conversations in a permanent backend database.

The browser temporarily stores the current session so the conversation remains available after refreshing the same tab.

Users should avoid sharing highly sensitive personal or identifying information.

Disclaimer

MindPal is an educational and supportive conversational tool.

It is not intended to diagnose, treat, cure, or prevent any mental-health condition.

For professional guidance, users should contact a qualified healthcare or mental-health professional.
