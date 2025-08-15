import requests
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Optional
import time

class OllamaRAG:
    def __init__(self, model_name: str = "llama3.2", ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        
        print(f"Initializing RAG with Ollama model: {model_name}")
        
        # Check if Ollama is running and model is available
        self._check_ollama_connection()
        
        # Initialize embedding model (local)
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
    
    def _check_ollama_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama not responding")
            
            # Check if model is available
            models = response.json().get('models', [])
            available_models = [model['name'].split(':')[0] for model in models]
            
            if self.model_name not in available_models:
                print(f"⚠️  Model '{self.model_name}' not found.")
                print(f"Available models: {', '.join(available_models)}")
                print(f"To install the model, run: ollama pull {self.model_name}")
                
                if available_models:
                    self.model_name = available_models[0]
                    print(f"Using available model: {self.model_name}")
                else:
                    raise Exception("No models available in Ollama")
            
            print(f"✓ Ollama connected, using model: {self.model_name}")
            
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            print("Make sure Ollama is running: ollama serve")
            raise
    
    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Add documents to the knowledge base"""
        if metadata is None:
            metadata = [{"id": i, "text_preview": text[:100]} for i, text in enumerate(texts)]
        
        print(f"Processing {len(texts)} documents...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts, 
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.document_metadata.extend(metadata)
        
        print(f"✓ Added {len(texts)} documents. Total: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        if len(self.documents) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # Search in FAISS
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
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
    
    def generate_answer(self, query: str, top_k: int = 3) -> Dict[str, any]:
        """Generate answer using RAG with Ollama"""
        start_time = time.time()
        
        # Retrieve relevant documents
        search_results = self.search(query, top_k)
        
        if not search_results:
            return {
                "answer": "No relevant information found in the knowledge base.",
                "sources": [],
                "search_time": time.time() - start_time
            }
        
        # Prepare context
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"Source {i}: {result['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt optimized for Ollama
        prompt = f"""Based on the provided context, answer the user's question accurately and concisely. If the context doesn't contain sufficient information, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        generation_start = time.time()
        answer = self._call_ollama(prompt)
        generation_time = time.time() - generation_start
        
        return {
            "answer": answer,
            "sources": search_results,
            "search_time": generation_start - start_time,
            "generation_time": generation_time,
            "total_time": time.time() - start_time
        }
    
    def chat(self, query: str, top_k: int = 3) -> str:
        """Simple chat interface"""
        result = self.generate_answer(query, top_k)
        return result["answer"]
    
    def save_index(self, filepath: str):
        """Save the FAISS index and documents"""
        faiss.write_index(self.index, f"{filepath}.index")
        
        data = {
            "documents": self.documents,
            "metadata": self.document_metadata,
            "model_name": self.model_name
        }
        
        with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load the FAISS index and documents"""
        self.index = faiss.read_index(f"{filepath}.index")
        
        with open(f"{filepath}.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.documents = data["documents"]
        self.document_metadata = data["metadata"]
        
        print(f"✓ Index loaded from {filepath}")


# Document processing utilities
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) > 10:  # Only keep meaningful chunks
            chunks.append(' '.join(chunk_words))
    
    return chunks

def load_text_files(filepaths: List[str]) -> List[str]:
    """Load multiple text files"""
    documents = []
    for filepath in filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Chunk large files
                if len(content) > 2000:
                    chunks = chunk_text(content)
                    documents.extend(chunks)
                else:
                    documents.append(content)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return documents


# Main execution
if __name__ == "__main__":
    # Available models you might have:
    # "llama3.2", "llama2", "codellama", "mistral", "phi3", "gemma2"
    
    model_choice = input("Enter Ollama model name (default: llama3.2): ").strip()
    if not model_choice:
        model_choice = "llama3.2"
    
    try:
        # Initialize RAG system
        rag = OllamaRAG(model_name=model_choice)
        
        # Sample technical documents
        sample_docs = [
            """Python is a high-level, interpreted programming language with dynamic semantics. 
            Its high-level built-in data structures, combined with dynamic typing and dynamic binding, 
            make it very attractive for Rapid Application Development, as well as for use as a scripting 
            or glue language to connect existing components together.""",
            
            """RAG (Retrieval-Augmented Generation) is a natural language processing technique that combines 
            information retrieval with text generation. It works by first retrieving relevant documents from 
            a knowledge base, then using those documents as context for generating responses.""",
            
            """Vector databases are specialized databases designed to store and query high-dimensional vectors 
            efficiently. They use algorithms like approximate nearest neighbor (ANN) search to find similar 
            vectors quickly, making them essential for AI applications like semantic search and recommendation systems.""",
            
            """Ollama is a tool that allows you to run large language models locally on your machine. 
            It supports various models like Llama 2, Code Llama, Mistral, and others. It provides a simple 
            API for interacting with these models without needing external services.""",
            
            """FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and 
            clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, 
            up to ones that possibly do not fit in RAM."""
        ]
        
        # Add documents
        rag.add_documents(sample_docs)
        
        # Interactive session
        print("\n" + "="*60)
        print("🤖 RAG System with Ollama Ready!")
        print("Ask questions about the loaded documents")
        print("Commands: 'quit' to exit, 'save' to save index")
        print("="*60)
        
        while True:
            query = input("\n💭 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            elif query.lower() == 'save':
                rag.save_index("ollama_rag_index")
                continue
            elif not query:
                continue
            
            print(f"\n🔍 Processing: {query}")
            
            # Get detailed response
            result = rag.generate_answer(query)
            
            print(f"\n💬 Answer: {result['answer']}")
            print(f"\n📊 Performance:")
            print(f"   Search: {result['search_time']:.2f}s")
            print(f"   Generation: {result['generation_time']:.2f}s")
            print(f"   Total: {result['total_time']:.2f}s")
            
            if result['sources']:
                print(f"\n📚 Sources used:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"   {i}. Score: {source['score']:.3f}")
                    print(f"      {source['text'][:100]}...")
        
        # Save before exit
        print("\n💾 Saving index...")
        rag.save_index("ollama_rag_index")
        print("✓ Goodbye!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("1. Ollama is running: ollama serve")
        print("2. Model is installed: ollama pull llama3.2")
        print("3. Check available models: ollama list")