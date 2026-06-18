Medical_FirstAid_RAG
# 🩺 Medical First Aid Manual using RAG AI

An AI-powered Medical First Aid Manual built with Retrieval-Augmented Generation (RAG), Streamlit, Pinecone, Ollama, and Sentence Transformers.
🚑 **Built an AI-Powered Medical First Aid Manual using RAG + Local LLMs!**
Over the past few weeks, I've been working on a project that combines **Artificial Intelligence** with **medical knowledge retrieval** to create an intelligent First Aid Assistant.
Instead of relying on generic chatbot responses, this application uses **Retrieval-Augmented Generation (RAG)** to answer questions directly from trusted medical manuals.

### 🔥 Tech Stack

✅ Python
✅ Streamlit
✅ Pinecone Vector Database
✅ Ollama (Local LLM)
✅ Sentence Transformers
✅ PyMuPDF
✅ Retrieval-Augmented Generation (RAG)


One of the biggest lessons from this project was realizing that **LLMs become significantly more reliable when grounded in domain-specific knowledge rather than relying solely on pretrained information.**

This application allows healthcare professionals, students, and emergency responders to ask natural language questions from first aid manuals and receive accurate, context-aware answers with source references.

## Features

* 📄 Multi-PDF ingestion
* 🧠 Retrieval-Augmented Generation (RAG)
* 🤖 Local LLM using Ollama
* 🔍 Semantic Search with Pinecone
* 📚 Source Citation
* 📖 Intelligent Chunking
* ⚡ Fast Vector Retrieval
* 🏥 Medical First Aid Knowledge Base
* 📋 Hierarchical Summarization
* 🎯 Metadata-based Filtering
* 💻 Interactive Streamlit Interface

## Workflow

1. Upload one or more First Aid manuals.
2. PDFs are cleaned and split into semantic chunks.
3. Chunks are embedded using Sentence Transformers.
4. Embeddings are stored in Pinecone.
5. User asks a medical question.
6. Relevant passages are retrieved using semantic search.
7. Ollama generates an answer grounded in retrieved evidence.
8. Sources are displayed alongside the response.

## Example Questions

* How should severe bleeding be controlled?
* What are the signs of heat stroke?
* How do you perform CPR?
* What should be done for snake bites?
* What are the symptoms of shock?
* How should burns be treated?

## Future Improvements

* Voice Assistant
* OCR Support
* Medical Image Understanding
* Drug Interaction Lookup
* Emergency Decision Trees
* Hospital Guidelines Integration
* Hybrid Search (BM25 + Vector)
* Citation Confidence Scores

## Disclaimer

This project is intended for educational and research purposes only. It is **not** a substitute for professional medical advice, diagnosis, or emergency healthcare. Always consult qualified medical professionals for clinical decisions.

Developed as an AI-powered Medical First Aid Manual demonstrating Retrieval-Augmented Generation (RAG), semantic search, and local Large Language Models.

#ArtificialIntelligence #RAG #LLM #Python #MachineLearning #MedicalAI #Healthcare #Streamlit #Pinecone #Ollama #GenerativeAI #OpenSource #AIProjects #DataScience #SoftwareEngineering




