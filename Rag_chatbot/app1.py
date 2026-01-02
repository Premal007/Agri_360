import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import streamlit as st

load_dotenv()
import asyncio
import io
import time
import re
import json
from datetime import datetime
from googletrans import Translator
from gtts import gTTS
from langdetect import detect
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from operator import itemgetter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableMap
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
import streamlit as st
# Translation class
class HinglishTranslator:
    def __init__(self):
        self.trans = Translator()
        self.technical_terms = {
            'soil': 'मिट्टी',
            'cultivation': 'खेती',
            'fertilizer': 'उर्वरक',
            'strawberry': 'स्ट्रॉबेरी',
            'tomato': 'टमाटर',
            'wheat': 'गेहूँ',
            'onion': 'प्याज'
        }

    async def detect_lang(self, text: str) -> str:
        try:
            detected = await self.trans.detect(text)
            return 'hi' if detected.lang == 'hi' else 'en'
        except:
            return 'en'

    async def to_english(self, text: str) -> str:
        try:
            for hin, eng in {v: k for k, v in self.technical_terms.items()}.items():
                text = text.replace(hin, eng)
            return (await self.trans.translate(text, src='hi', dest='en')).text
        except:
            return text

    async def to_hindi(self, text: str) -> str:
        try:
            result = (await self.trans.translate(text, src='en', dest='hi')).text
            for eng, hin in self.technical_terms.items():
                result = result.replace(eng, hin)
            return result
        except:
            return text

translator = HinglishTranslator()

# RAG Pipeline Functions
def extract_plant_name(question, known_plants):
    """Extracts plant name from question using a list of known plants."""
    question_lower = question.lower()
    for plant in known_plants:
        if re.search(rf'\b{plant}\b', question_lower):
            return plant
    return None

def load_pdf_files(data_path):
    print("📄 Loading PDF documents...")
    documents = []
    for filename in os.listdir(data_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(data_path, filename)
            print(f"🔍 Processing: {filename}")
            try:
                with fitz.open(file_path) as pdf:
                    text = ""
                    for page in pdf:
                        try:
                            page_text = page.get_text()
                            text += f"\n{page_text}\n"
                        except Exception as page_error:
                            print(f"⚠️ Error reading page: {page_error}")
                            continue
                    if text.strip():
                        plant_name = os.path.splitext(filename)[0].lower() #file na name mathi extension kadhie etle khali plant_name malse
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": filename, "plant": plant_name}
                        ))
            except Exception as e:
                print(f"❌ Failed to process {filename}: {str(e)}")
    return documents

def reciprocal_rank_fusion(results: list[list], k=60):
    """Reciprocal Rank Fusion that takes multiple lists of ranked documents."""
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [
        loads(doc)
        for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

def filtered_retriever(question, known_plants, retriever):
    plant_name = extract_plant_name(question, known_plants)
    if plant_name:
        docs = retriever.invoke(question, filter={"plant": plant_name})
        if docs:
            return docs
    docs = retriever.invoke(question, filter={"plant": "general"})
    return docs

# Vectorstore setup
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'device': 'cpu', 'batch_size': 32, 'normalize_embeddings': True}
)
db_path = "faiss_index"
known_plants = ['melon', 'potato', 'okra', 'lettuce', 'peanut', 'onion', 'strawberry', 'sugarcane']

if os.path.exists(db_path):
    print(" Loading FAISS index from disk...")
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
else:
    print("🚀 Building FAISS index from PDFs...")
    documents = load_pdf_files("data")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
    chunks = splitter.split_documents(documents) #chunks create thaya
    db = FAISS.from_documents(chunks, embedding_model) # temathi embeddings generate karisu
    db.save_local(db_path)
    print(f" FAISS vector store created with {len(chunks)} chunks.")

retriever = db.as_retriever(search_kwargs={"k": 3})
# Initialize Google AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    max_output_tokens=1024,
    convert_system_message_to_human=True,
)

 
# Multi-query prompt
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

# Decomposition Prompt
decomposition_msg = ChatPromptTemplate.from_template(
    """You are a helpful assistant that generates multiple sub-questions related to an input question. 
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. 
Generate exactly three search queries related to: {question}
Output (3 queries, each on a new line):"""
)

# Step-Back Prompt
examples = [
    {"input": "Could the members of The Police perform lawful arrests?", "output": "What can the members of The Police do?"},
    {"input": "Jan Sindel’s was born in what country?", "output": "What is Jan Sindel’s personal history?"}
]
example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)
step_back_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:"""),
    few_shot_prompt,
    ("user", "{question}")
])

# Helper functions
def generate_step_back(sub_question):
    return step_back_prompt | llm | StrOutputParser()

def generate_multiquery(sub_question):
    return prompt_perspectives | llm | StrOutputParser() | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])

def combine_questions(input_dict):
    question = input_dict["question"]
    sub_questions = (decomposition_msg | llm | StrOutputParser() | (lambda x: [q.strip() for q in x.split("\n") if q.strip()][:3])).invoke({"question": question})
    step_back_questions = [generate_step_back(sub_q).invoke({"question": sub_q}) for sub_q in sub_questions]
    combined_questions = []
    for i in range(3):
        combined_questions.append(sub_questions[i])
        combined_questions.append(step_back_questions[i])
    multiquery_questions = []
    for sub_q in combined_questions:
        multiquery_ques = generate_multiquery(sub_q).invoke({"question": sub_q})
        multiquery_questions.extend(multiquery_ques)
    return multiquery_questions

# Create the decomposition chain
final_decomposition_stepback_chain = RunnableLambda(combine_questions)

# Create the retrieval chain
def retrieval_chain_fn(input_dict):
    questions = final_decomposition_stepback_chain.invoke(input_dict)
    doc_lists = [filtered_retriever(q, known_plants, retriever) for q in questions]
    return reciprocal_rank_fusion(doc_lists)

# Rerank prompt
rerank_prompt = ChatPromptTemplate.from_template("""
You are an assistant that ranks documents by their relevance to the question.
Question: {question}
Documents:
{documents}

Output:
Provide a ranked list of document indices from most relevant to least relevant, separated by commas.
""")

def rerank_documents_llm(question, documents):
    formatted_docs = "\n\n".join([f"[{i}]: {doc.page_content[:300]}..." for i, doc in enumerate(documents)])
    ranked_indices = (
        rerank_prompt
        | llm
        | StrOutputParser()
        | (lambda x: [int(idx.strip()) for idx in x.split(",") if idx.strip().isdigit()])
    ).invoke({"question": question, "documents": formatted_docs})
    reranked_docs = [documents[i] for i in ranked_indices]
    return reranked_docs

# RAG prompt
rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the following question based on the context provided only,don't you use your knowledge.

Context:
{context}

Question: {question}
""")


# Conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)

# Final pipeline with memory
pipeline = (
    RunnableMap({
        "question": itemgetter("question"),
        "docs": RunnableLambda(retrieval_chain_fn),
        "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]
    })
    | RunnableMap({
        "question": itemgetter("question"),
        "docs": RunnableLambda(lambda inputs: rerank_documents_llm(inputs["question"], inputs["docs"])),
        "chat_history": itemgetter("chat_history")
    })
    | {
        "context": lambda inputs: "\n\n".join([doc.page_content for doc in inputs["docs"][:5]]),
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history")
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Async response function
async def get_response(user_query: str):
    try:
        # Detect language and translate
        lang = await translator.detect_lang(user_query)
        translated_query = await translator.to_english(user_query) if lang == 'hi' else user_query

        # Run pipeline
        inputs = {"question": translated_query}
        response = pipeline.invoke(inputs)

        # Save to memory
        memory.save_context({"question": translated_query}, {"answer": response})

        # Translate back if needed
        if lang == 'hi':
            response = await translator.to_hindi(response)

        return response, lang
    except Exception as e:
        error_msg = f"Error: {str(e)}. Please try again with a different question."
        if lang == 'hi':
            error_msg = await translator.to_hindi(error_msg)
        return error_msg, lang

# ---------------------- FAQ cache and helpers ----------------------
FAQ_CACHE_FILE = "faq_cache.json"

# A simple set of common farmer questions (editable)
faq_questions = [
    "How often should I water tomato plants?",
    "What fertilizer is best for onion?",
    "How to prevent pests on strawberry?",
    "When is the right time to harvest wheat?",
    "What is the ideal soil pH for lettuce?",
    "How to increase sugarcane yield?",
    "How to store harvested potatoes?",
    "How to propagate okra?",
    "How much sunlight does peanut need?",
    "How to deal with wilting in melon plants?"
]

def load_faq_cache():
    try:
        if os.path.exists(FAQ_CACHE_FILE):
            with open(FAQ_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Could not load FAQ cache: {e}")
    return {}

def save_faq_cache(cache: dict):
    try:
        with open(FAQ_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Could not save FAQ cache: {e}")

def _append_assistant_message(answer: str, lang: str):
    """Append assistant message to session and generate audio (non-blocking UI friendly)."""
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "lang": lang
    })
    # generate audio and cache it in session state
    new_audio_key = f"audio_{len(st.session_state.messages)-1}"
    try:
        audio_data = text_to_speech(answer, lang)
        st.session_state.audio_cache[new_audio_key] = audio_data
    except Exception as e:
        print(f"Audio generation failed: {e}")

def handle_faq_click(question: str):
    """When a FAQ is clicked: return cached answer or compute, cache and return."""
    # ensure cache in session
    if "faq_cache" not in st.session_state:
        st.session_state.faq_cache = load_faq_cache()

    cache = st.session_state.faq_cache

    # If cached, display immediately
    if question in cache:
        answer = cache[question]["answer"]
        lang = cache[question].get("lang", "en")
        # append user and assistant messages
        st.session_state.messages.append({"role": "user", "content": question})
        _append_assistant_message(answer, lang)
        return

    # Not cached: compute answer, store in cache and persist
    st.session_state.messages.append({"role": "user", "content": question})
    placeholder = st.empty()
    placeholder.markdown("✍ Thinking (FAQ)...")
    try:
        answer, lang = asyncio.run(get_response(question))
    except Exception as e:
        answer = f"Error computing FAQ answer: {e}"
        lang = "en"

    # Save to cache and persist
    cache[question] = {"answer": answer, "lang": lang}
    st.session_state.faq_cache = cache
    try:
        save_faq_cache(cache)
    except Exception:
        pass

    placeholder.markdown(answer)
    _append_assistant_message(answer, lang)

# Streamlit UI
custom_css = """
<style>
    .stChatMessage { padding: 1rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.05); }
    .stMarkdown { font-size: 16px; line-height: 1.7; }
    .stChatInput input { padding: 12px; font-size: 16px; border-radius: 12px !important; border: 1px solid #ccc; }
    .stButton > button { border-radius: 10px; padding: 6px 12px; font-weight: bold; }
    audio { width: 100%; margin-top: 10px; }
</style>
"""

def text_to_speech(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

def main():
    st.set_page_config(page_title="Plant Chatbot", layout="centered")
    st.markdown(custom_css, unsafe_allow_html=True)
    st.title("🌱 Plant Sustainable Chatbot – Ask Your Questions!")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "audio_cache" not in st.session_state:
        st.session_state.audio_cache = {}
    if "faq_cache" not in st.session_state:
        # load persisted faq cache into session (answers are just text + lang; audio will be generated on demand)
        st.session_state.faq_cache = load_faq_cache()

    # Frequently Asked Questions (tap a question to get a fast cached answer)
    st.subheader("Common Questions")
    faq_cols = st.columns(2)
    for i, q in enumerate(faq_questions):
        col = faq_cols[i % 2]
        if col.button(q, key=f"faq_btn_{i}"):
            handle_faq_click(q)

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "🧑"):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                audio_key = f"audio_{idx}"
                if audio_key in st.session_state.audio_cache:
                    st.audio(st.session_state.audio_cache[audio_key], format="audio/mp3")

    # Chat input
    prompt = st.chat_input("Hey, ask something about your plant")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            placeholder.markdown("✍ Thinking...")
            answer, lang = asyncio.run(get_response(prompt))
            displayed_text = ""
            for char in answer:
                displayed_text += char
                placeholder.markdown(displayed_text + "▌")
                time.sleep(0.008)
            placeholder.markdown(displayed_text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "lang": lang
            })

            new_audio_key = f"audio_{len(st.session_state.messages)-1}"
            with st.spinner("Generating audio..."):
                audio_data = text_to_speech(answer, lang)
                st.session_state.audio_cache[new_audio_key] = audio_data

            st.audio(st.session_state.audio_cache[new_audio_key], format="audio/mp3")


    # Sidebar tools
    with st.sidebar:
        st.header("🛠 Tools")
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.session_state.audio_cache = {}
            memory.clear()
            st.rerun()
        if st.button("⚡ Cache all FAQs"):
            uncached = [q for q in faq_questions if q not in st.session_state.get('faq_cache', {})]
            if uncached:
                with st.spinner(f"Caching {len(uncached)} FAQ answers..."):
                    for q in uncached:
                        try:
                            answer, lang = asyncio.run(get_response(q))
                        except Exception as e:
                            answer = f"Error: {e}"
                            lang = "en"
                        st.session_state.faq_cache[q] = {"answer": answer, "lang": lang}
                    try:
                        save_faq_cache(st.session_state.faq_cache)
                    except Exception:
                        pass
                st.success("Cached FAQ answers.")
            else:
                st.info("All FAQs already cached.")
        st.download_button(
            label="📄 Download .txt",
            data="\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages]),
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        st.download_button(
            label="🧾 Download .json",
            data=json.dumps(st.session_state.messages, indent=2),
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()