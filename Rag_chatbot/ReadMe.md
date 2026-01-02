# 🌱 Agri - RAG Chatbot
### Multilingual, Hallucination-Free Agriculture Knowledge Assistant using RAG

---

## 📌 Overview

Agri-RAGBot is an AI-powered **Retrieval-Augmented Generation (RAG)** chatbot designed to deliver **accurate, document-grounded agricultural guidance** using curated PDF knowledge sources. Unlike generic chatbots, this system strictly answers questions based on its internal agriculture-specific knowledge base and avoids hallucinated or unsupported responses.

The bot supports **multilingual interaction (English, Hindi, Hinglish)**, provides **text and audio responses**, and is deployed using an intuitive **Streamlit interface**, making expert agricultural knowledge accessible to farmers, students, and researchers.

---

## 🎯 Objectives

- Provide reliable agriculture-related answers grounded in verified PDF documents  
- Prevent hallucinations by restricting responses to retrieved document context  
- Enable multilingual question–answer interaction  
- Support speech-based responses for improved accessibility  
- Offer a user-friendly interface with chat export and FAQ caching  

---

## 🧠 Key Features

- ✅ Retrieval-Augmented Generation (RAG) architecture  
- ✅ Vector-based semantic search using FAISS  
- ✅ LLM-driven query decomposition, multi-query retrieval, and reranking  
- ✅ Hallucination-free answers for out-of-scope queries  
- ✅ Multilingual support (English, Hindi, Hinglish)  
- ✅ Text-to-Speech (TTS) audio output  
- ✅ Conversation memory for contextual follow-ups  
- ✅ FAQ caching for faster repeated responses  
- ✅ Downloadable chat history in `.txt` and `.json` formats  
- ✅ Clean and interactive Streamlit UI  

---

## 🏗️ System Architecture (High-Level)

1. PDF Knowledge Base  
2. Text Extraction & Chunking  
3. Embedding Generation (Sentence Transformers)  
4. FAISS Vector Database  
5. LLM-Based Query Reformulation  
6. Retrieval + Reciprocal Rank Fusion (RRF)  
7. LLM Reranking  
8. RAG-Based Answer Generation  
9. Translation & Audio Output  
10. Streamlit Interface  

---

## 📂 Data Sources

- Curated agriculture-related PDF documents (crop cultivation, irrigation, pests, fertilizers, soil management, etc.)
- PDF processing pipeline:
  - PyMuPDF for text extraction
  - Recursive text chunking with overlap
  - Metadata tagging (source file, crop/topic)

---

## 🔄 Detailed Workflow & Methodology

The Agri-RAGBot workflow is divided into two main phases: **Knowledge Base Preparation** and **Query Processing & Response Generation**.

---

### Phase 1: Knowledge Base Preparation (Offline / One-Time Process)

#### Step 1: PDF Ingestion
- Agriculture-related PDFs are scanned from a data directory.
- Text is extracted page-by-page using **PyMuPDF (fitz)**.

#### Step 2: Text Cleaning & Metadata Attachment
- Extracted text is cleaned while preserving contextual meaning.
- Metadata is attached to each document:
  - Source filename
  - Crop or topic name

#### Step 3: Text Chunking
- Documents are split into overlapping chunks using a **Recursive Character Text Splitter**.
- Overlapping ensures continuity of context.

#### Step 4: Embedding Generation
- Each chunk is converted into a dense vector using:
