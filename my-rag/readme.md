# Dependencies:
```bash
pip install sentence-transformers faiss-cpu numpy requests
```

# Key Advantages with Ollama

- **Better responses**: Much higher quality than DialoGPT  
- **Local privacy**: Everything runs on your machine  
- **Model flexibility**: Easy to switch between models  
- **Performance tracking**: Shows search/generation times  

---

## Quick Setup

1. **Make sure Ollama is running**:  
   ```bash
   ollama serve
   ```

2. **Install a model**:  
   ```bash
   ollama pull llama3.2  # or mistral, codellama
   ```

3. **Run the script**.

---

## Popular Ollama Models for RAG

- **llama3.2**: Good balance of speed/quality  
- **mistral**: Fast and efficient  
- **codellama**: Great for code-related questions  
- **phi3**: Small but capable  

To check your available models:  
```bash
ollama list
```

The system automatically detects available models and offers to use them.

----

## Dependencies for the Web App

Install the following dependencies for the web app:  
```bash
pip install fastapi uvicorn PyPDF2 sentence-transformers faiss-cpu requests python-multipart
```

---

## Features

- **Web Interface**: Clean, responsive design  
- **PDF Upload**: Drag & drop PDF files up to 300+ pages  
- **Progress Tracking**: Real-time progress bar during processing  
- **Smart Chunking**: Sentence-aware chunking with overlap  
- **Interactive Chat**: Ask questions and see sources  
- **Performance Metrics**: Shows search/generation times  

---

## How to Run

1. **Make sure Ollama is running**:  
   ```bash
   ollama serve
   ```

2. **Install a model if needed**:  
   ```bash
   ollama pull llama3.2
   ```

3. **Run the web app**:  
   ```bash
   python your_script.py
   ```
   Access the app at: [http://localhost:8000](http://localhost:8000)

---

## System Capabilities

- Handles large PDFs (300+ pages) efficiently  
- Batch processing for embeddings  
- Text cleaning and preprocessing  
- Ignores images (text-only extraction)  
- Background processing with status updates  
- Persistent document storage  

---

## Production Considerations

- Add authentication  
- Use PostgreSQL + pgvector instead of FAISS  
- Add rate limiting  
- Deploy with Docker