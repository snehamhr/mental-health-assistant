````markdown
## MindPal

MindPal is a supportive mental-health chatbot built with FastAPI, Groq, Hugging Face Inference API, local semantic retrieval artifacts, and a responsive web interface.

The application retrieves relevant information from psychology books and uses it as background context to generate short, empathetic, and context-aware responses.

## Features

- Supportive mental-health conversations
- Retrieval-Augmented Generation using psychology books
- Remote query embeddings through Hugging Face Inference API
- Pre-generated local document embeddings
- Groq-powered language-model responses
- Conversation context within the current browser session
- Crisis-message detection and safety routing
- Out-of-scope question handling
- Responsive web interface
- New Chat functionality
- No permanent backend conversation storage

## Technology Stack

### Backend

- Python
- FastAPI
- Uvicorn
- Groq API
- Pydantic
- Pydantic Settings

### Retrieval

- Hugging Face Inference API
- `sentence-transformers/all-MiniLM-L6-v2`
- NumPy
- Pickle
- Pre-generated document embeddings

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
Hugging Face feature-extraction API
    ↓
384-dimensional MiniLM query embedding
    ↓
Cosine similarity against stored document embeddings
    ↓
Top matching psychology-book chunks
    ↓
Groq language model
    ↓
Supportive response
````

Only the user-query embedding is generated remotely during deployment.

The psychology-book chunks and their pre-generated embeddings remain stored locally inside the application repository.

This reduces deployment memory usage because Torch, Transformers, and Sentence Transformers are not loaded inside the Render web service.

## Knowledge Base

The knowledge base is created from PDF books stored inside:

```text
source_books/
```

The preprocessing pipeline:

1. Opens each PDF
2. Extracts text page by page
3. Cleans unnecessary whitespace and broken formatting
4. Divides the text into overlapping chunks
5. Generates normalized MiniLM document embeddings
6. Saves the processed chunks and embeddings
7. Validates that every chunk has one corresponding embedding

Generated artifacts:

```text
data/combined_chunks.pkl
data/embeddings.npy
```

Current knowledge-base configuration:

```text
Embedding model: sentence-transformers/all-MiniLM-L6-v2
Embedding dimension: 384
Chunk size: approximately 900 characters
Chunk overlap: approximately 150 characters
```

The document embeddings must be generated using the same embedding model configured for remote query embeddings.

## Project Structure

```text
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
```

## Local Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root.

```env
APP_NAME=MindPal

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant

HF_API_KEY=your_huggingface_token
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

CHUNKS_PATH=data/combined_chunks.pkl
EMBEDDINGS_PATH=data/embeddings.npy

SIMILARITY_THRESHOLD=0.55
TOP_K_RESULTS=4
MEMORY_WINDOW=16
RETRIEVAL_USER_TURNS=6
MAX_USER_MESSAGE_CHARS=4000
```

The actual `.env` file must not be committed to GitHub.

The Hugging Face token must have permission to use inference services.

## Build the Knowledge Base

The deployed application does not require Torch or Sentence Transformers at runtime.

However, the offline knowledge-base build script uses the local embedding model to generate document embeddings.

Place the PDF books inside:

```text
source_books/
```

Install the preprocessing dependencies in a local development environment:

```bash
python -m pip install sentence-transformers PyMuPDF
```

Then run:

```bash
python scripts/build_knowledge_base.py
```

The script creates:

```text
data/combined_chunks.pkl
data/embeddings.npy
```

It also validates that:

* the chunk file contains usable text
* the embedding array is two-dimensional
* the chunk count matches the number of embedding rows
* the embedding values are finite
* the embedding dimension is consistent

## Run the Application Locally

```bash
python -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Deployment Architecture

The deployed Render service loads:

* FastAPI
* Uvicorn
* Groq client
* Hugging Face Inference client
* NumPy
* stored chunks
* stored document embeddings

The deployed service does not load:

* Torch
* Sentence Transformers
* Transformers
* scikit-learn
* CUDA libraries

This keeps the runtime memory footprint lower and makes deployment more suitable for a free Render web service.

## Render Configuration

Build command:

```bash
python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt
```

Start command:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Recommended Python version:

```text
3.11.9
```

Required Render environment variables:

```text
APP_NAME
GROQ_API_KEY
GROQ_MODEL
HF_API_KEY
EMBEDDING_MODEL
CHUNKS_PATH
EMBEDDINGS_PATH
SIMILARITY_THRESHOLD
TOP_K_RESULTS
MEMORY_WINDOW
RETRIEVAL_USER_TURNS
MAX_USER_MESSAGE_CHARS
PYTHON_VERSION
```

## Conversation Memory

MindPal does not permanently store conversations on the server.

Conversation history is temporarily stored using browser `sessionStorage`.

This means:

* Refreshing the same browser tab keeps the current conversation
* Clicking New Chat clears the current conversation
* Closing the browser tab ends the stored session
* Conversations are not shared between users
* Conversations are not written to a backend database

## Safety

MindPal includes a rule-based safety-routing layer for messages involving possible self-harm or immediate danger.

Crisis-related messages are handled before normal retrieval and language-model generation.

MindPal is not a replacement for:

* A licensed mental-health professional
* Medical diagnosis
* Emergency services
* Crisis intervention services

## Scope

MindPal is designed to discuss topics such as:

* Anxiety
* Stress
* Depression
* Low mood
* Anger
* Emotional regulation
* Coping
* Self-esteem
* Relationships
* Feeling overwhelmed
* General psychological concepts

Clearly unrelated questions may receive an out-of-scope response.

## Limitations

* Responses depend on the quality of the psychology books
* Image-only or scanned PDF pages may not be extracted
* PDF formatting may introduce text artifacts
* Remote embedding generation depends on Hugging Face service availability
* Groq response generation depends on Groq API availability
* The chatbot may make mistakes
* The application does not provide clinical diagnosis
* Crisis detection is rule-based and may not identify every situation
* Free Render services may sleep after inactivity and take time to restart

## Privacy

MindPal does not intentionally store conversations in a permanent backend database.

The current conversation is stored temporarily inside the user’s browser tab.

Users should avoid sharing highly sensitive personal or identifying information.

API keys are stored as environment variables and are not exposed in the frontend.

## Disclaimer

MindPal is an educational and supportive conversational tool.

It is not intended to diagnose, treat, cure, or prevent any mental-health condition.

For professional guidance, users should contact a qualified healthcare or mental-health professional.
