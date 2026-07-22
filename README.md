# MindHaven

A deployment-friendly mental-health RAG chatbot with a polished website UI,
FastAPI backend, OpenAI generation, strict similarity routing, local chat
history, and lightweight profile memory.

## What was preserved from the notebook

- `all-MiniLM-L6-v2` query embeddings
- Existing `combined_chunks.pkl`
- Existing `embeddings.npy`
- Cosine-similarity retrieval
- Top-4 retrieved chunks
- A configurable relevance threshold
- Mental-health-only scope

The notebook's TinyLlama response generator is replaced with the OpenAI API.
The heavy emotion-classification pipeline is intentionally removed for easier
CPU deployment. The OpenAI model receives the retrieved context, recent
conversation, and explicit user memory.

## Important routing behavior

1. Crisis language receives an immediate static safety response.
2. The query embedding is compared with the stored embeddings.
3. When the best score is below `SIMILARITY_THRESHOLD`, the backend returns the
   fixed out-of-context message.
4. In that low-similarity branch, **the OpenAI API is not called**.
5. Only relevant messages reach OpenAI with retrieved context.

## Chat memory without sign-in

Chats and explicit profile details are stored in browser `localStorage`.

For example:

- `My name is Sneha`
- `I am a girl`
- `My pronouns are she/her`
- `Please remember that I prefer short exercises`

After refresh, the same browser can restore chats and memory. This is not an
account system: data does not sync to another browser/device and is lost if the
user clears browser storage. The UI includes buttons to clear chats and memory.

## Project structure

```text
mindhaven-chatbot/
├── app/
│   ├── main.py
│   ├── rag.py
│   ├── llm.py
│   ├── memory.py
│   ├── safety.py
│   ├── config.py
│   ├── schemas.py
│   └── static/
│       ├── index.html
│       ├── styles.css
│       └── app.js
├── data/
│   ├── combined_chunks.pkl
│   └── embeddings.npy
├── scripts/
│   └── validate_artifacts.py
├── .env.example
├── Dockerfile
├── render.yaml
└── requirements.txt
```

## Local setup on Windows Command Prompt

### 1. Copy notebook artifacts

Copy your existing files into `data/`:

```text
data/combined_chunks.pkl
data/embeddings.npy
```

### 2. Create and activate an environment

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Create `.env`

```cmd
copy .env.example .env
```

Edit `.env` and add the real key:

```env
OPENAI_API_KEY=your_real_key
```

### 4. Validate the artifacts

```cmd
python scripts\validate_artifacts.py
```

### 5. Run

```cmd
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Threshold

Default:

```env
SIMILARITY_THRESHOLD=0.55
```

Test the threshold against known in-scope and out-of-scope questions before
final deployment. A higher threshold rejects more questions. A lower threshold
allows more questions through.

## Render deployment

1. Push this project to a private GitHub repository.
2. Keep `combined_chunks.pkl` and `embeddings.npy` in the repository if their
   size and licensing permit it.
3. In Render, create a new Blueprint from the repository. `render.yaml` will be
   detected.
4. Add `OPENAI_API_KEY` as a secret environment variable.
5. Deploy.

The Docker image downloads `all-MiniLM-L6-v2` while building, reducing cold
start time. One Gunicorn worker is intentional so the embedding model and
matrix are not duplicated in memory.

## Security

- Never put the OpenAI API key in frontend JavaScript.
- Never commit `.env`.
- The backend uses `store=False` for OpenAI responses.
- Browser chat history remains local to the user's browser.
- The chatbot clearly states that it is not emergency or professional care.
