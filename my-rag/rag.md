# RAG: Retrieval-Augmented Generation

## What is RAG?

**RAG** = **Retrieval + Augmented + Generation**

- **Retrieval**: Find relevant documents from your knowledge base.
- **Augmented**: Add retrieved documents as context to the prompt.
- **Generation**: LLM generates an answer using both its training and your context.

---

## RAG in the Code

### Retrieval
```python
# RETRIEVAL: Search vector database for relevant docs
search_results = self.search(query, top_k)
```

### Augmented
```python
# AUGMENTED: Add context to prompt
context = "\n\n".join([result['text'] for result in search_results])
prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
```

### Generation
```python
# GENERATION: LLM generates answer with context
answer = llm_provider.generate(prompt)
```

---

## RAG Components

1. **Knowledge Base**: Your PDF documents (300+ pages).
2. **Embedding Model**: Converts text → vectors (e.g., SentenceTransformers).
3. **Vector Database**: Stores & searches vectors (e.g., FAISS).
4. **LLM**: Generates answers (e.g., GPT, Ollama, Claude).

---

## How to Use RAG with GPT Models

### Setup with OpenAI GPT
```python
rag = MultiModelRAG()
openai_provider = OpenAIProvider("your-api-key", "gpt-4")
rag.add_llm_provider("gpt4", openai_provider)
```

### Add Your Documents
```python
rag.add_documents(your_pdf_chunks)
```

### Ask Questions
```python
result = rag.generate_answer("Your question about the PDF")
```

---

## Benefits of RAG

- **Uses YOUR data**: Not just GPT's training data.
- **Reduces hallucination**: Provides factual context.
- **Source attribution**: Shows which parts of the PDF were used.
- **Cost-effective**: Cheaper than fine-tuning GPT.
- **Always up-to-date**: Add new documents anytime.

---

## RAG vs. Regular LLM

### Without RAG
**User**: "What did the CEO say about Q4 revenue?"  
**GPT**: "I don't have access to your company's financial data."

### With RAG
- System retrieves relevant sections from your financial PDF.
- **GPT**: "According to the Q4 earnings report, the CEO mentioned revenue increased 15%..."

---

## Compatibility with GPT Models

RAG works seamlessly with:

- **OpenAI GPT-3.5/GPT-4**: Cloud-based, high quality.
- **Ollama Models**: Local, private.
- **Any LLM Provider**: Extensible architecture.

You can even compare answers from different models using the same retrieved context.

---

## Quick Start

### Install Dependencies
```bash
pip install sentence-transformers faiss-cpu openai
```

### Set OpenAI API Key
```bash
export OPENAI_API_KEY="your-key-here"
```

### Use RAG with GPT
```python
rag = MultiModelRAG()
rag.add_llm_provider("gpt4", OpenAIProvider(api_key, "gpt-4"))
rag.add_documents(your_documents)
result = rag.generate_answer("Your question")
```

---

The RAG system makes GPT smarter about YOUR specific documents while maintaining all of GPT's language capabilities.