"""
RAG (Retrieval-Augmented Generation) Detailed Explanation
=========================================================

RAG is a technique that combines:
1. RETRIEVAL: Finding relevant information from a knowledge base
2. AUGMENTED: Adding this information as context 
3. GENERATION: Using an LLM to generate answers based on the context

Components of RAG:
------------------
1. Knowledge Base: Your documents (PDF, text files, etc.)
2. Embedding Model: Converts text to vectors for similarity search
3. Vector Database: Stores and searches document embeddings (FAISS, Pinecone, etc.)
4. LLM: Generates answers using retrieved context (GPT, Ollama, etc.)

How RAG Works:
--------------
User Query → Embed Query → Search Vector DB → Retrieve Relevant Docs → 
Generate Answer with Context → Return Enhanced Response

Benefits:
---------
- Uses YOUR data (not just training data)
- Reduces hallucination 
- Provides source attribution
- Updates without retraining
- Cost-effective vs fine-tuning
"""

import openai
import requests
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Optional, Literal
import time
from abc import ABC, abstractmethod
import os
from dataclasses import dataclass

@dataclass
class RAGResult:
    answer: str
    sources: List[Dict]
    search_time: float
    generation_time: float
    total_time: float
    model_used: str

class LLMProvider(ABC):
    """Abstract base class for different LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT models (GPT-3.5, GPT-4, etc.)"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model
        self._test_connection()
    
    def _test_connection(self):
        """Test OpenAI API connection"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            print(f"✓ OpenAI {self.model} connected successfully")
        except Exception as e:
            print(f"❌ OpenAI connection failed: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error with OpenAI: {str(e)}"
    
    def get_model_name(self) -> str:
        return f"OpenAI {self.model}"

class OllamaProvider(LLMProvider):
    """Local Ollama models (Llama, Mistral, etc.)"""
    
    def __init__(self, model: str = "llama3.2", url: str = "http://localhost:11434"):
        self.model = model
        self.url = url
        self._test_connection()
    
    def _test_connection(self):
        """Test Ollama connection and model availability"""
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama not responding")
            
            models = response.json().get('models', [])
            available_models = [model['name'].split(':')[0] for model in models]
            
            if self.model not in available_models:
                if available_models:
                    print(f"Model '{self.model}' not found, using: {available_models[0]}")
                    self.model = available_models[0]
                else:
                    raise Exception("No models available in Ollama")
            
            print(f"✓ Ollama {self.model} connected successfully")
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"Error calling Ollama: {response.status_code}"
                
        except Exception as e:
            return f"Error with Ollama: {str(e)}"
    
    def get_model_name(self) -> str:
        return f"Ollama {self.model}"

class AnthropicProvider(LLMProvider):
    """Anthropic Claude models"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        # Note: This is a placeholder - actual implementation would use anthropic library
        self.api_key = api_key
        self.model = model
        print(f"✓ Anthropic {model} provider initialized (placeholder)")
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        # Placeholder implementation
        return "Anthropic Claude implementation would go here"
    
    def get_model_name(self) -> str:
        return f"Anthropic {self.model}"

class MultiModelRAG:
    """
    RAG system that can work with multiple LLM providers
    """
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        print("Initializing Multi-Model RAG System...")
        
        # Initialize embedding model (always local for efficiency)
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✓ Embedding model loaded: {embedding_model}")
        
        # Initialize vector database (FAISS)
        self.dimension = 384  # for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Document storage
        self.documents = []
        self.document_metadata = []
        
        # LLM providers
        self.providers = {}
        self.current_provider = None
        
        print("✓ RAG system initialized")
    
    def add_llm_provider(self, name: str, provider: LLMProvider):
        """Add an LLM provider to the system"""
        self.providers[name] = provider
        print(f"✓ Added LLM provider: {name} ({provider.get_model_name()})")
        
        if self.current_provider is None:
            self.current_provider = name
            print(f"✓ Set default provider: {name}")
    
    def set_provider(self, provider_name: str):
        """Switch to a different LLM provider"""
        if provider_name not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(f"Provider '{provider_name}' not found. Available: {available}")
        
        self.current_provider = provider_name
        print(f"✓ Switched to provider: {provider_name}")
    
    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Add documents to the knowledge base"""
        if metadata is None:
            metadata = [{"id": i, "preview": text[:100]} for i, text in enumerate(texts)]
        
        print(f"Processing {len(texts)} documents...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents
        self.documents.extend(texts)
        self.document_metadata.extend(metadata)
        
        print(f"✓ Added {len(texts)} documents. Total: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents - THE RETRIEVAL PART OF RAG"""
        if len(self.documents) == 0:
            return []
        
        # Convert query to embedding vector
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # Search in vector database
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return relevant documents
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.document_metadata[idx],
                    "score": float(score)
                })
        
        return results
    
    def generate_answer(self, query: str, top_k: int = 3, provider: str = None) -> RAGResult:
        """
        Generate answer using RAG - THE COMPLETE RAG PROCESS
        
        Steps:
        1. RETRIEVE: Search for relevant documents
        2. AUGMENT: Add retrieved docs as context to prompt
        3. GENERATE: Use LLM to generate answer with context
        """
        start_time = time.time()
        
        # Use specified provider or current default
        provider_name = provider or self.current_provider
        if not provider_name or provider_name not in self.providers:
            raise ValueError(f"No valid provider available. Current: {provider_name}")
        
        # STEP 1: RETRIEVE relevant documents
        search_start = time.time()
        search_results = self.search(query, top_k)
        search_time = time.time() - search_start
        
        if not search_results:
            return RAGResult(
                answer="No relevant information found in the knowledge base.",
                sources=[],
                search_time=search_time,
                generation_time=0,
                total_time=time.time() - start_time,
                model_used=self.providers[provider_name].get_model_name()
            )
        
        # STEP 2: AUGMENT - Prepare context from retrieved documents
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"Source {i}: {result['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Create RAG prompt
        rag_prompt = f"""Based on the provided context, answer the user's question accurately and comprehensively. 
If the context doesn't contain sufficient information, clearly state what information is missing.

Context from knowledge base:
{context}

User Question: {query}

Answer:"""
        
        # STEP 3: GENERATE answer using LLM
        generation_start = time.time()
        llm_provider = self.providers[provider_name]
        answer = llm_provider.generate(rag_prompt)
        generation_time = time.time() - generation_start
        
        return RAGResult(
            answer=answer,
            sources=search_results,
            search_time=search_time,
            generation_time=generation_time,
            total_time=time.time() - start_time,
            model_used=llm_provider.get_model_name()
        )
    
    def compare_models(self, query: str, providers: List[str] = None) -> Dict[str, RAGResult]:
        """Compare answers from different LLM providers"""
        providers_to_test = providers or list(self.providers.keys())
        results = {}
        
        print(f"\n🔍 Comparing models for query: '{query}'")
        print("=" * 60)
        
        for provider_name in providers_to_test:
            if provider_name in self.providers:
                print(f"\n🤖 Testing {provider_name}...")
                try:
                    result = self.generate_answer(query, provider=provider_name)
                    results[provider_name] = result
                    
                    print(f"✓ {provider_name}: {result.total_time:.2f}s total")
                except Exception as e:
                    print(f"❌ {provider_name} failed: {e}")
        
        return results
    
    def get_status(self) -> Dict:
        """Get system status"""
        return {
            "total_documents": len(self.documents),
            "vector_index_size": self.index.ntotal,
            "available_providers": list(self.providers.keys()),
            "current_provider": self.current_provider,
            "embedding_model": "all-MiniLM-L6-v2"
        }

# Example usage and demonstration
def demonstrate_rag():
    """Demonstrate how RAG works step by step"""
    
    print("🚀 RAG Demonstration")
    print("=" * 50)
    
    # Initialize RAG system
    rag = MultiModelRAG()
    
    # Add sample documents (this is your knowledge base)
    sample_documents = [
        """Python is a high-level, interpreted programming language with dynamic semantics. 
        Its high-level built-in data structures, combined with dynamic typing and dynamic binding, 
        make it very attractive for Rapid Application Development, as well as for use as a scripting 
        or glue language to connect existing components together. Python's simple, easy to learn 
        syntax emphasizes readability and therefore reduces the cost of program maintenance.""",
        
        """Machine Learning is a subset of artificial intelligence (AI) that provides systems 
        the ability to automatically learn and improve from experience without being explicitly 
        programmed. Machine learning focuses on the development of computer programs that can 
        access data and use it to learn for themselves. The process of learning begins with 
        observations or data, such as examples, direct experience, or instruction.""",
        
        """RAG (Retrieval-Augmented Generation) combines the benefits of retrieval-based and 
        generative approaches to question answering. Instead of relying solely on the parametric 
        knowledge stored in model parameters, RAG retrieves relevant documents from an external 
        knowledge source and uses them to inform the generation process. This approach helps 
        reduce hallucination and provides more factual, grounded responses.""",
        
        """Vector databases are specialized databases designed to handle high-dimensional vector 
        data efficiently. They excel at similarity search operations, which are crucial for AI 
        applications. These databases use techniques like approximate nearest neighbor (ANN) 
        algorithms to quickly find similar vectors in large datasets, making them ideal for 
        applications like recommendation systems, semantic search, and RAG implementations."""
    ]
    
    # Add documents to RAG system
    print("\n📚 Adding documents to knowledge base...")
    rag.add_documents(sample_documents)
    
    # Try to add different LLM providers (you would uncomment the ones you have access to)
    
    # Add Ollama if available
    try:
        ollama_provider = OllamaProvider("llama3.2")
        rag.add_llm_provider("ollama", ollama_provider)
    except Exception as e:
        print(f"⚠️  Ollama not available: {e}")
    
    # Add OpenAI if API key is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            openai_provider = OpenAIProvider(openai_api_key, "gpt-3.5-turbo")
            rag.add_llm_provider("openai-3.5", openai_provider)
            
            # You can also add GPT-4
            gpt4_provider = OpenAIProvider(openai_api_key, "gpt-4")
            rag.add_llm_provider("openai-4", gpt4_provider)
        except Exception as e:
            print(f"⚠️  OpenAI not available: {e}")
    else:
        print("⚠️  OpenAI API key not found in environment")
    
    # Check system status
    status = rag.get_status()
    print(f"\n📊 System Status:")
    print(f"Documents: {status['total_documents']}")
    print(f"Providers: {status['available_providers']}")
    print(f"Current: {status['current_provider']}")
    
    if not rag.providers:
        print("❌ No LLM providers available. Please set up Ollama or OpenAI API key.")
        return
    
    # Test queries
    test_queries = [
        "What is Python used for?",
        "How does RAG work?",
        "What are vector databases?",
        "Explain machine learning"
    ]
    
    print(f"\n💬 Testing RAG with different queries...")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        
        try:
            result = rag.generate_answer(query)
            
            print(f"\n🤖 Answer ({result.model_used}):")
            print(result.answer)
            
            print(f"\n📚 Sources used ({len(result.sources)}):")
            for i, source in enumerate(result.sources, 1):
                print(f"{i}. Score: {source['score']:.3f}")
                print(f"   {source['text'][:150]}...")
            
            print(f"\n⏱️  Performance:")
            print(f"Search: {result.search_time:.3f}s | Generation: {result.generation_time:.3f}s | Total: {result.total_time:.3f}s")
            
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Compare multiple models if available
    if len(rag.providers) > 1:
        print(f"\n🆚 Comparing different models...")
        comparison_query = "What is the main benefit of using RAG?"
        results = rag.compare_models(comparison_query)
        
        for provider, result in results.items():
            print(f"\n{provider.upper()}:")
            print(f"Answer: {result.answer[:200]}...")
            print(f"Time: {result.total_time:.2f}s")

if __name__ == "__main__":
    """
    HOW TO USE THIS RAG SYSTEM:
    
    1. SETUP:
       - Install dependencies: pip install sentence-transformers faiss-cpu openai requests
       - For Ollama: Install Ollama and run 'ollama serve'
       - For OpenAI: Set OPENAI_API_KEY environment variable
    
    2. INITIALIZE:
       rag = MultiModelRAG()
    
    3. ADD LLM PROVIDERS:
       rag.add_llm_provider("ollama", OllamaProvider("llama3.2"))
       rag.add_llm_provider("openai", OpenAIProvider(api_key, "gpt-3.5-turbo"))
    
    4. ADD YOUR DOCUMENTS:
       rag.add_documents(your_documents)
    
    5. ASK QUESTIONS:
       result = rag.generate_answer("Your question here")
    
    6. GET ANSWERS WITH SOURCES:
       print(result.answer)
       print("Sources:", result.sources)
    """
    
    demonstrate_rag()