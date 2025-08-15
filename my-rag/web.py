from sentence_transformers import SentenceTransformer
from typing import Optional, List, Dict

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from ollama_rag import OllamaRAG

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    search_time: float
    generation_time: float
    total_time: float


# Initialize RAG system
rag_system = None

# FastAPI app
app = FastAPI(title="RAG PDF Assistant", description="Upload PDF and ask questions")

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG PDF Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .upload-section, .chat-section { border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
        .progress-bar { width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .progress-fill { height: 100%; background-color: #4CAF50; transition: width 0.3s ease; }
        .chat-messages { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0; background-color: #f9f9f9; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user-message { background-color: #e3f2fd; text-align: right; }
        .bot-message { background-color: #f1f8e9; }
        .sources { margin-top: 10px; padding: 10px; background-color: #fff3e0; border-left: 3px solid #ff9800; font-size: 0.9em; }
        input[type="file"], input[type="text"] { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 3px; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 3px; cursor: pointer; }
        button:hover { background-color: #45a049; }
        button:disabled { background-color: #cccccc; cursor: not-allowed; }
        .status { padding: 10px; margin: 10px 0; border-radius: 3px; }
        .status.success { background-color: #d4edda; color: #155724; }
        .status.error { background-color: #f8d7da; color: #721c24; }
        .status.info { background-color: #d1ecf1; color: #0c5460; }
    </style>
</head>
<body>
    <h1>🤖 RAG PDF Assistant</h1>
    <p>Upload a PDF document and ask questions about its content</p>
    
    <div class="container">
        <div class="upload-section">
            <h3>📄 Document Upload</h3>
            <input type="file" id="fileInput" accept=".pdf">
            <button onclick="uploadPDF()">Upload PDF</button>
            
            <div id="uploadStatus" class="status" style="display: none;"></div>
            <div class="progress-bar" id="progressBar" style="display: none;">
                <div class="progress-fill" id="progressFill" style="width: 0%;"></div>
            </div>
            <div id="progressText"></div>
            
            <h4>Document Stats</h4>
            <div id="docStats">No document loaded</div>
        </div>
        
        <div class="chat-section">
            <h3>💬 Ask Questions</h3>
            <div id="chatMessages" class="chat-messages"></div>
            <input type="text" id="queryInput" placeholder="Ask a question about the document..." onkeypress="handleKeyPress(event)">
            <button onclick="askQuestion()" id="askButton">Ask</button>
        </div>
    </div>

    <script>
        let uploadInProgress = false;
        
        async function uploadPDF() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a PDF file');
                return;
            }
            
            if (!file.type.includes('pdf')) {
                alert('Please select a PDF file');
                return;
            }
            
            uploadInProgress = true;
            document.getElementById('uploadStatus').style.display = 'block';
            document.getElementById('progressBar').style.display = 'block';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                // Start upload
                const response = await fetch('/upload-pdf', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    // Start polling for progress
                    pollProgress();
                } else {
                    throw new Error('Upload failed');
                }
            } catch (error) {
                showStatus('Upload failed: ' + error.message, 'error');
                uploadInProgress = false;
            }
        }
        
        async function pollProgress() {
            try {
                const response = await fetch('/upload-status');
                const status = await response.json();
                
                updateProgress(status.progress, status.message);
                
                if (status.status === 'complete') {
                    showStatus('PDF processed successfully!', 'success');
                    updateDocStats();
                    uploadInProgress = false;
                } else if (status.status === 'error') {
                    showStatus('Error: ' + status.message, 'error');
                    uploadInProgress = false;
                } else {
                    setTimeout(pollProgress, 1000);
                }
            } catch (error) {
                showStatus('Error checking progress: ' + error.message, 'error');
                uploadInProgress = false;
            }
        }
        
        function updateProgress(progress, message) {
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressText').textContent = message;
        }
        
        function showStatus(message, type) {
            const statusEl = document.getElementById('uploadStatus');
            statusEl.textContent = message;
            statusEl.className = 'status ' + type;
        }
        
        async function updateDocStats() {
            try {
                const response = await fetch('/stats');
                const stats = await response.json();
                document.getElementById('docStats').innerHTML = `
                    <strong>Documents:</strong> ${stats.total_documents}<br>
                    <strong>Model:</strong> ${stats.model_name}
                `;
            } catch (error) {
                console.error('Error fetching stats:', error);
            }
        }
        
        async function askQuestion() {
            const queryInput = document.getElementById('queryInput');
            const query = queryInput.value.trim();
            
            if (!query) return;
            
            const askButton = document.getElementById('askButton');
            askButton.disabled = true;
            askButton.textContent = 'Thinking...';
            
            addMessage(query, 'user');
            queryInput.value = '';
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                const result = await response.json();
                addMessage(result.answer, 'bot', result);
                
            } catch (error) {
                addMessage('Error: ' + error.message, 'bot');
            } finally {
                askButton.disabled = false;
                askButton.textContent = 'Ask';
            }
        }
        
        function addMessage(message, type, result = null) {
            const messagesEl = document.getElementById('chatMessages');
            const messageEl = document.createElement('div');
            messageEl.className = 'message ' + type + '-message';
            
            if (type === 'bot' && result) {
                let sourcesHtml = '';
                if (result.sources && result.sources.length > 0) {
                    sourcesHtml = '<div class="sources"><strong>Sources:</strong>';
                    result.sources.forEach((source, i) => {
                        sourcesHtml += `<br>${i+1}. Score: ${source.score.toFixed(3)} - ${source.text.substring(0, 100)}...`;
                    });
                    sourcesHtml += `<br><small>Search: ${result.search_time.toFixed(2)}s, Generation: ${result.generation_time.toFixed(2)}s</small></div>`;
                }
                messageEl.innerHTML = message + sourcesHtml;
            } else {
                messageEl.textContent = message;
            }
            
            messagesEl.appendChild(messageEl);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                askQuestion();
            }
        }
        
        // Initialize
        updateDocStats();
    </script>
</body>
</html>
"""

@app.on_event("startup")
async def startup_event():
    global rag_system
    try:
        rag_system = OllamaRAG()
        print("✓ RAG system started successfully")
    except Exception as e:
        print(f"❌ Failed to start RAG system: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_TEMPLATE

@app.post("/upload-pdf")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Read file content
    content = await file.read()
    
    # Process in background
    background_tasks.add_task(rag_system.process_pdf_file, content)
    
    return {"message": "PDF upload started, processing in background"}

@app.get("/upload-status")
async def get_upload_status():
    return rag_system.processing_status

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    result = rag_system.generate_answer(request.query, request.top_k)
    return QueryResponse(**result)

@app.get("/stats")
async def get_stats():
    return {
        "total_documents": len(rag_system.documents),
        "model_name": rag_system.model_name,
        "index_size": rag_system.index.ntotal
    }

@app.post("/save-knowledge-base")
async def save_knowledge_base():
    try:
        rag_system.save_knowledge_base()
        return {"message": "Knowledge base saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save knowledge base: {e}")

@app.post("/load-knowledge-base")
async def load_knowledge_base():
    try:
        rag_system.load_knowledge_base()
        return {"message": "Knowledge base loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load knowledge base: {e}")

if __name__ == "__main__":
    print("Starting RAG Web Application...")
    print("Make sure Ollama is running: ollama serve")
    print("Access the web interface at: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)