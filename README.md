# 📚 RAG PDF Chatbot with Ollama (Gemma3:4b)

This project builds a chatbot that answers questions based on the content of **multiple PDFs** using a **Retrieval-Augmented Generation (RAG)** pipeline with **Ollama’s Gemma3:4b** model.

---

## 🚀 Features

- 🔍 Supports multiple PDF documents
- 🧠 Uses a local LLM (Ollama - gemma3:4b)
- 🧾 Embedding-based vector retrieval using `Chroma`
- 💬 Clean chatbot interface using Streamlit

---

## 🧱 Project Structure

📁 Project Structure ``` rag-pdf-chatbot-ollama/ 
├── app.py # Streamlit chatbot interface 
├── my_docs/ # Folder to store input PDFs 
├── rag_chroma/ # Auto-generated Chroma DB (do not commit) 
├── requirements.txt # Dependencies 

