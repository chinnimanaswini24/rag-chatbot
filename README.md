# 📄 RAG Document Chatbot

Ask questions about your documents and get answers backed by the source text.

🔗 **Live demo:** https://rag-chatbot-pcrwjkwrcypeqcqmxzrkhn.streamlit.app

## ✨ Features
- Upload PDF, TXT or Markdown files
- TF-IDF retrieval with adjustable top-k and chunk size
- Answers show their source passages with match scores
- Extractive mode by default, with optional LLM answers via an Anthropic API key
- Download the chat as a Markdown file

## 🧠 How it works
1. Documents are split into chunks.
2. Chunks are indexed with TF-IDF (scikit-learn).
3. Your question is matched against the chunks using cosine similarity.
4. The top-k passages are returned, or passed to an LLM to write the answer.

## 🛠️ Tech stack
Python · Streamlit · scikit-learn · pypdf · Anthropic API (optional)

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

To enable LLM answers, set your key first:
```bash
set ANTHROPIC_API_KEY=your-key      # Windows
export ANTHROPIC_API_KEY=your-key   # Mac/Linux
```
