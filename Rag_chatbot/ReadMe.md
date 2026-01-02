# 🌱 Agri - RAG Chatbot
### Multilingual, Hallucination-Free Agriculture Knowledge Assistant using RAG

---

## 📌 Overview

Agri-RAGBot is an AI-powered **Retrieval-Augmented Generation (RAG)** chatbot designed to deliver **accurate, document-grounded agricultural guidance** using curated PDF knowledge sources. Unlike generic chatbots, this system strictly answers questions based on its internal agriculture-specific knowledge base and avoids hallucinated or unsupported responses.

The bot supports **multilingual interaction (English, Hindi, Gujarati, Tamil, etc)**, provides **text and audio responses**, and is deployed using an intuitive **Streamlit interface**, making expert agricultural knowledge accessible to farmers, students, and researchers.

---

## 🎯 Objectives

- Provide reliable agriculture-related answers grounded in verified PDF documents  
- Prevent hallucinations by restricting responses to retrieved document context  
- Enable multilingual question–answer interaction  
- Support speech-based responses for improved accessibility  
- Offer a user-friendly interface with chat export and FAQ caching  

---

## 🧠 Key Features

-  Retrieval-Augmented Generation (RAG) architecture  
-  Vector-based semantic search using FAISS  
-  LLM-driven query decomposition, multi-query retrieval, and reranking  
-  Hallucination-free answers for out-of-scope queries  
-  Multilingual support (English, Hindi, Hinglish)  
-  Text-to-Speech (TTS) audio output  
-  Conversation memory for contextual follow-ups  
-  FAQ caching for faster repeated responses  
-  Downloadable chat history in `.txt` and `.json` formats  
-  Clean and interactive Streamlit UI  

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

### Phase 1: Knowledge Base Preparation (One-Time Process)

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
```bash
sentence-transformers/all-MiniLM-L12-v2
```
- Embeddings capture semantic meaning rather than keyword matching.

#### Step 5: Vector Index Creation
- Embeddings are stored in a **FAISS vector database**.
- The index is persisted locally to avoid rebuilding.

---

### Phase 2: Query Processing & Response Generation (Runtime Process)
This phase runs every time a user interacts with the chatbot.

#### Step 6: User Query Input
- User enters a question via the Streamlit chat interface.
- Input can be in English, Hindi, or Hinglish.

#### Step 7: Language Detection & Translation
- Query language is detected.
- Non-English queries are translated to English for internal processing.
- Agricultural terminology is preserved.

#### Step 8: Intelligent Query Reformulation
- To improve retrieval accuracy, the original query is expanded using LLM-based techniques:
  - **Sub-question generation**: breaks complex questions into simpler components.
  - **Step-back generalization**: creates broader versions of the query.
  - **Multi-query paraphrasing**: generates multiple alternative phrasings.

#### Step 9: Semantic Retrieval
- Each query variation searches the FAISS vector store.
- Metadata filtering is applied when crop names are detected.

#### Step 10: Reciprocal Rank Fusion (RRF)
- Multiple ranked result lists are merged.
- Documents consistently retrieved across queries receive higher priority.

#### Step 11: LLM-Based Reranking
- Retrieved chunks are re-ranked by the LLM for contextual relevance.

#### Step 12: RAG-Based Answer Generation
- Top-ranked chunks are injected into a structured RAG prompt.
- **GPT-4.1 via OpenRouter API** generates a grounded response.
- If no relevant context exists, the bot safely declines.

#### Step 13: Response Post-Processing
- Output is translated back to the user’s language.
- **gTTS** converts text responses into speech.

#### Step 14: Memory & Caching
- Conversation history is stored for multi-turn coherence.
- Frequently asked questions are cached to reduce latency.

#### Step 15: Output & Export
- Response is displayed in Streamlit with optional audio playback.
- Chat history can be downloaded in `.txt` or `.json` format.

---

## 🧑‍💻 Tech Stack

### Core Technologies
- Python
- Retrieval-Augmented Generation (RAG)
- GPT-4.1 (OpenRouter API)
- FAISS Vector Database
- Sentence Transformers
- LangChain

### Supporting Tools
- Streamlit (UI)
- PyMuPDF (PDF processing)
- GoogleTrans (Multilingual translation)
- gTTS (Text-to-Speech)
- JSON (Caching & Export)
- asyncio (Async processing)

---

## 📊 Results & Validation

- Accurate, document-grounded responses  
- No hallucination for out-of-domain queries  
- Successful multilingual interaction  
- Faster responses due to FAQ caching  
- Stable multi-turn conversations with memory  

---

## 🚫 Hallucination Control

The bot strictly answers only when relevant information exists in the knowledge base. Queries outside the agriculture domain are politely declined, ensuring reliability and trustworthiness.

---

## 📈 Use Cases

- Farmers seeking crop and irrigation guidance  
- Agriculture students and researchers  
- Educational and advisory institutions  
- Precision farming support systems  

---

## 🔮 Future Enhancements

- Image-based plant disease detection    
- Mobile application deployment  
- Integration with IoT sensors and weather APIs  

---

## 🚀 How to Run the Project

```bash
# Clone the repository
git clone https://github.com/your-username/agri-ragbot.git

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_openrouter_key

# Run the application
streamlit run app.py
```
