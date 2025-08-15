import re
import os
import io
import requests
import pickle
from typing import List, Dict, Optional
import time

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2

from openai import AzureOpenAI

class OllamaRAG:

    def __init__(self, model_name: str = "llama3.2", ollama_url: str = "http://localhost:11434"):
        
        # Load Azure OpenAI credentials from environment variables
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
        azure_openai_model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")
        
        if not azure_openai_endpoint or not azure_openai_key:
            raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY environment variables must be set")
        
        self.azure_openai_model = azure_openai_model
        
        self.azureOpenAiClient = AzureOpenAI(
            api_version="2025-01-01-preview",
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key
        )
        
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.processing_status = {"status": "idle", "progress": 0, "message": ""}
        
        print(f"Initializing RAG with Ollama model: {model_name}")
        
        # Check Ollama connection
        self._check_ollama_connection()
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Embedding model loaded")
        
        # Initialize FAISS index
        self.dimension = 384
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Document storage
        self.documents = []
        self.document_metadata = []
        
        print("✓ RAG system initialized")

        self.load_knowledge_base()

    def _check_ollama_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama not responding")
            
            models = response.json().get('models', [])
            available_models = [model['name'].split(':')[0] for model in models]
            
            if self.model_name not in available_models:
                print(f"⚠️  Model '{self.model_name}' not found.")
                print(f"Available models: {', '.join(available_models)}")
                
                if available_models:
                    self.model_name = available_models[0]
                    print(f"Using available model: {self.model_name}")
                else:
                    raise Exception("No models available")
            
            print(f"✓ Ollama connected, using model: {self.model_name}")
            
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = []
            total_pages = len(pdf_reader.pages)
            
            print(f"Processing PDF with {total_pages} pages...")
            self.processing_status = {
                "status": "extracting",
                "progress": 0,
                "message": f"Extracting text from {total_pages} pages..."
            }
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        # Clean up the text
                        cleaned_text = self._clean_text(page_text)
                        if len(cleaned_text) > 50:  # Only meaningful content
                            text_content.append(cleaned_text)
                    
                    # Update progress
                    progress = int((page_num + 1) / total_pages * 50)  # 50% for extraction
                    self.processing_status["progress"] = progress
                    
                    if page_num % 10 == 0:  # Log every 10 pages
                        print(f"Processed page {page_num + 1}/{total_pages}")
                        
                except Exception as e:
                    print(f"Error processing page {page_num + 1}: {e}")
                    continue
            
            full_text = "\n\n".join(text_content)
            print(f"✓ Extracted text from {len(text_content)} pages")
            
            return full_text
            
        except Exception as e:
            raise Exception(f"Failed to extract PDF text: {e}")
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\n\d+\n', '\n', text)  # Page numbers on separate lines
        text = re.sub(r'Page \d+', '', text)   # "Page X" patterns
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def chunk_text_advanced(self, text: str, chunk_size: int = 2000, overlap: int = 250) -> List[str]:
        """Advanced text chunking with sentence awareness"""
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.strip()) > 100:  # Only meaningful chunks
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) > 100:
                chunks.append(chunk_text)
        
        return chunks
    
    def process_pdf_file(self, pdf_content: bytes) -> Dict[str, any]:
        """Process PDF file and add to knowledge base"""
        try:
            # Extract text
            full_text = self.extract_text_from_pdf(pdf_content)
            
            if not full_text.strip():
                raise Exception("No text content found in PDF")
            
            # Chunk the text
            self.processing_status = {
                "status": "chunking",
                "progress": 50,
                "message": "Chunking text..."
            }
            
            chunks = self.chunk_text_advanced(full_text)
            print(f"✓ Created {len(chunks)} text chunks")
            
            # Create metadata for chunks
            metadata = []
            for i, chunk in enumerate(chunks):
                metadata.append({
                    "chunk_id": i,
                    "source": "pdf_upload",
                    "preview": chunk[:150] + "..." if len(chunk) > 150 else chunk,
                    "word_count": len(chunk.split())
                })
            
            # Generate embeddings and add to index
            self.processing_status = {
                "status": "embedding",
                "progress": 75,
                "message": "Generating embeddings..."
            }
            
            self.add_documents(chunks, metadata)
            
            self.processing_status = {
                "status": "complete",
                "progress": 100,
                "message": f"Successfully processed PDF with {len(chunks)} chunks"
            }

            self.save_knowledge_base()

            return {
                "success": True,
                "chunks_created": len(chunks),
                "total_documents": len(self.documents),
                "text_length": len(full_text)
            }
            
        except Exception as e:
            self.processing_status = {
                "status": "error",
                "progress": 0,
                "message": str(e)
            }
            raise Exception(f"PDF processing failed: {e}")
    
    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Add documents to the knowledge base"""
        if metadata is None:
            metadata = [{"id": i} for i in enumerate(texts)]
        
        # Generate embeddings in batches for large documents
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False  # Disable for web interface
            )
            all_embeddings.append(batch_embeddings)
            
            # Update progress
            if hasattr(self, 'processing_status'):
                progress = 75 + int((i + batch_size) / len(texts) * 25)
                self.processing_status["progress"] = min(progress, 99)
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.document_metadata.extend(metadata)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        if len(self.documents) == 0:
            return []
        
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.document_metadata[idx],
                    "score": float(score)
                })
        
        return results
    
    def _call_ollama(self, prompt: str, max_tokens: int = 500) -> str:
        """Call Ollama API for text generation"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"Error calling Ollama: {response.status_code}"
                
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"
    
    def _call_azure_openai(self, prompt: str, max_tokens: int = 4000) -> str:
        """Call Azure OpenAI API for text generation"""
        try:
            response = self.azureOpenAiClient.chat.completions.create(
                model=self.azure_openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9
            )
            result = response.choices[0].message.content
            return result

        except Exception as e:
            return f"Error calling Azure OpenAI: {str(e)}"

    def generate_answer(self, query: str, top_k: int = 3) -> Dict[str, any]:
        """Generate answer using RAG with Ollama"""
        start_time = time.time()
        
        search_results = self.search(query, top_k)
        
        if not search_results:
            return {
                "answer": "No relevant information found in the knowledge base. Please upload a PDF document first.",
                "sources": [],
                "search_time": time.time() - start_time,
                "generation_time": 0,
                "total_time": time.time() - start_time
            }
        
        # Prepare context
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"Source {i}: {result['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the provided context from the uploaded document, answer the user's question accurately and comprehensively. If the context doesn't contain sufficient information, clearly state what information is missing.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        generation_start = time.time()
        answer = self._call_azure_openai(prompt)
        generation_time = time.time() - generation_start
        
        return {
            "answer": answer,
            "sources": search_results,
            "search_time": generation_start - start_time,
            "generation_time": generation_time,
            "total_time": time.time() - start_time
        }
    
    def save_knowledge_base(self, file_path: str = "knowledge_base.pkl"):
        """Save the knowledge base (documents, metadata, and FAISS index) to a file."""
        try:
            with open(file_path, "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "metadata": self.document_metadata,
                    "index": self.index
                }, f)
            print(f"✓ Knowledge base saved to {file_path}")
        except Exception as e:
            print(f"❌ Failed to save knowledge base: {e}")
            raise

    def load_knowledge_base(self, file_path: str = "knowledge_base.pkl"):
        """Load the knowledge base (documents, metadata, and FAISS index) from a file."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Knowledge base file '{file_path}' not found")
            
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.document_metadata = data["metadata"]
                self.index = data["index"]
            print(f"✓ Knowledge base loaded from {file_path}")
        except Exception as e:
            print(f"❌ Failed to load knowledge base: {e}")
            # raise

