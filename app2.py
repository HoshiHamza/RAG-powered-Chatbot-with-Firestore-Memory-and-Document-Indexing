import os
import shutil
from typing import List, Dict, Optional

import gradio as gr
from dotenv import load_dotenv
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import Runnable
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_community.document_loaders import FireCrawlLoader
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory

load_dotenv()

# =========================
# Config / Globals
# =========================
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID") or "langchain-chatbot-aad4e"
COLLECTION_NAME = os.getenv("FIRESTORE_COLLECTION", "chat_history")

# LLM & embeddings
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Paths (per-session vector store to keep things separate)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_ROOT = os.path.join(BASE_DIR, "db") 
os.makedirs(DB_ROOT, exist_ok=True)

# In-memory registries keyed by session_id
_rag_chain: Dict[str, Runnable] = {}
_retrievers: Dict[str, object] = {}
_vectorstores: Dict[str, Chroma] = {}
# Firestore message history wrappers keyed by session_id
_histories: Dict[str, FirestoreChatMessageHistory] = {}

# Firestore client
client = firestore.Client(project=PROJECT_ID)

# =========================
# Helpers
# =========================
def session_chroma_dir(session_id: str) -> str:
    safe = "".join(c for c in (session_id or "default") if c.isalnum() or c in "-_")
    path = os.path.join(DB_ROOT, "chroma", safe)
    os.makedirs(path, exist_ok=True)
    return path

def _fix_metadata_lists(docs):
    for d in docs:
        for k, v in list(d.metadata.items()):
            if isinstance(v, list):
                d.metadata[k] = ", ".join(map(str, v))
    return docs

def load_from_url(url: str, firecrawl_mode: str = "scrape"):
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY not set in env/Secrets (needed for URL mode).")
    loader = FireCrawlLoader(api_key=api_key, url=url.strip(), mode='scrape')
    docs = loader.load()
    return _fix_metadata_lists(docs)

def load_from_files(files: List[str]):
    docs = []
    for fpath in files:
        fpath = str(fpath)
        ext = os.path.splitext(fpath.lower())[-1]
        if ext == ".pdf":
            docs.extend(PyPDFLoader(fpath).load())
        else:
            docs.extend(UnstructuredFileLoader(fpath).load())
    return _fix_metadata_lists(docs)

def init_history(session_id: str) -> FirestoreChatMessageHistory:
    """Create or load Firestore-backed message history wrapper."""
    if session_id in _histories:
        return _histories[session_id]
    hist = FirestoreChatMessageHistory(
        session_id=session_id,
        collection=COLLECTION_NAME,
        client=client
    )
    # Ensure doc exists 
    client.collection(COLLECTION_NAME).document(session_id).set(
        {"created": firestore.SERVER_TIMESTAMP}, merge=True
    )
    _histories[session_id] = hist
    return hist

def get_session_ids():
    collection_ref = client.collection(COLLECTION_NAME)
    docs = collection_ref.stream()
    return sorted(set(d.id for d in docs))

def reset_index_for_session(session_id: str):
    _rag_chain.pop(session_id, None)
    _retrievers.pop(session_id, None)
    _vectorstores.pop(session_id, None)
    sdir = session_chroma_dir(session_id)
    if os.path.exists(sdir):
        shutil.rmtree(sdir)
    os.makedirs(sdir, exist_ok=True)

def _init_retriever_and_chain_for_session(session_id: str, vectorstore: Chroma):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = get_llm()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which may refer to context in the history, "
        "rewrite it as a standalone question. Do NOT answer it."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_system_prompt = (
        "You are an assistant for question-answering. Use the retrieved context to answer. "
        "If unknown, say you don't know. Keep answers concise (≤3 sentences).\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    _retrievers[session_id] = retriever
    _rag_chain[session_id] = rag_chain
    _vectorstores[session_id] = vectorstore



def to_text(content: Any) -> str:
    """Normalize LC/OpenAI content (str | list[parts] | other) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if "text" in p:
                    parts.append(str(p["text"]))
                elif p.get("type") == "text" and "text" in p:
                    parts.append(str(p["text"]))
                elif p.get("type") == "image_url":
                    parts.append("[image]")
                else:
                    parts.append(str(p))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    # Fallback for BaseMessage or other objects
    try:
        return to_text(getattr(content, "content"))
    except Exception:
        return str(content)


# =========================
# Build index (URL/Files) or choose "No Source"
# =========================
def build_index(session_id: str, mode: str, url: str, up_files, chunk_size: int, chunk_overlap: int):
    if not session_id or not session_id.strip():
        return "Please enter/select a Session ID first."

    # Ensure Firestore chat history exists
    init_history(session_id)

    if mode == "No Source":
        # Clear any previous retriever/index for this session
        reset_index_for_session(session_id)
        return "No Source selected. Chat will use only the LLM with Firestore memory."

    # Ingest content
    reset_index_for_session(session_id)
    if mode == "URL":
        if not url or not url.strip():
            return "Please enter a URL."
        docs = load_from_url(url.strip(), firecrawl_mode="scrape")
    else:  
        if not up_files:
            return "Please upload at least one file."
        docs = load_from_files(up_files)

    if not docs:
        return "No documents found to index."

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)

    # Build vector store under session-specific dir
    sdir = session_chroma_dir(session_id)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=sdir
    )
    vectorstore.persist()

    _init_retriever_and_chain_for_session(session_id, vectorstore)
    return f"[{session_id}] Indexed {len(splits)} chunks. Chat will use RAG for this session."

# =========================
# Chat handler
# =========================
def chat(user_input: str, history_ui, session_id: str):
    if not session_id or not session_id.strip():
        return history_ui + [["System", "Please enter/select a Session ID first."]], ""

    hist = init_history(session_id)
    llm = get_llm()

    if session_id in _rag_chain:
        rag = _rag_chain[session_id]  
        result = rag.invoke({
            "input": user_input,
            "chat_history": hist.messages
        })
        answer_raw = result.get("answer", "")
        answer = to_text(answer_raw)
    else:
        resp_msg = llm.invoke(hist.messages + [HumanMessage(content=user_input)])
        answer = to_text(resp_msg.content)

    # Persist to Firestore 
    hist.add_user_message(user_input)
    hist.add_ai_message(answer)

    history_ui.append([user_input, answer])
    return history_ui, ""

# =========================
# Load an existing chat session
# =========================
def load_chat(session_id: str):
    if not session_id:
        return []
    hist = init_history(session_id)

    ui_pairs = []
    last_human: Optional[str] = None

    for msg in hist.messages:
        role = getattr(msg, "type", None)  # "human" | "ai" | ...
        content = to_text(getattr(msg, "content", ""))  
        if role == "human":
            last_human = content
            ui_pairs.append([last_human, None])
        elif role == "ai":
            if ui_pairs and ui_pairs[-1][1] is None:
                ui_pairs[-1][1] = content
            else:
                # Edge case: AI message without preceding human
                ui_pairs.append([None, content])

    return ui_pairs
# =========================
# Gradio UI (single page)
# =========================
with gr.Blocks(title="Ask Your Data · Chat with Websites, Documents & Saved Sessions") as demo:
    gr.Markdown("## RAG + Firestore Chat · URL/Files optional · Firestore memory")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Sessions")
            session_id_box = gr.Textbox(label="Session ID (new or existing)", placeholder="e.g., demo-001")
            new_chat_btn = gr.Button("➕ Start New Chat")
            refresh_btn = gr.Button("🔄 Refresh Sessions")
            chat_sessions = gr.Dropdown(choices=get_session_ids(), label="Previous Chats", interactive=True)
            use_selected_btn = gr.Button("✅ Use Selected Session")

            gr.Markdown("### Ingest (Optional)")
            mode = gr.Radio(choices=["No Source", "URL", "Files"], value="No Source", label="Source Mode")
            url_in = gr.Textbox(label="Website URL (for URL mode)")
            files_in = gr.File(label="Upload documents", file_count="multiple", type="filepath")
            chunk_size = gr.Slider(256, 2000, value=1000, step=16, label="Chunk size")
            chunk_overlap = gr.Slider(0, 400, value=100, step=10, label="Chunk overlap")
            build_btn = gr.Button("🧱 Build / Reset Index for Session")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="Chat", height=520)
            msg = gr.Textbox(placeholder="Type your message...", show_label=False)
            send = gr.Button("Send")

    # --- Wiring ---
    # Refresh sessions dropdown
    def _refresh_sessions():
        return gr.update(choices=get_session_ids())
    refresh_btn.click(_refresh_sessions, outputs=[chat_sessions])

    # Use selected session -> copy into textbox and load chat
    def _use_selected(sess):
        if not sess:
            return "", []
        return sess, load_chat(sess)
    use_selected_btn.click(_use_selected, inputs=[chat_sessions], outputs=[session_id_box, chatbot])

    def start_new_chat(session_id: str):
        if not session_id or not session_id.strip():
            return "", [], gr.update()  
        # Init Firestore
        init_history(session_id)
        # Reset vector store for this session
        reset_index_for_session(session_id)

        # Refresh dropdown with new sessions list
        updated = get_session_ids()
        return session_id, [], gr.update(choices=updated, value=session_id)

    # Build index / or set No Source for this session
    build_btn.click(
        fn=build_index,
        inputs=[session_id_box, mode, url_in, files_in, chunk_size, chunk_overlap],
        outputs=[status]
    )
    new_chat_btn.click(
    start_new_chat,
    inputs=[session_id_box],
    outputs=[session_id_box, chatbot, chat_sessions]
)

    # Send message (must carry session id)
    send.click(chat, inputs=[msg, chatbot, session_id_box], outputs=[chatbot, msg])
    msg.submit(chat, inputs=[msg, chatbot, session_id_box], outputs=[chatbot, msg])

# Local run
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
