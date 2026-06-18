import os, gc, uuid, fitz, streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import chat

# ── ENV ────────────────────────────────────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("INDEX_NAME", "pdf-rag")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "llama3.2")

# FIX #4 — Score threshold was 0.55, far too high for all-MiniLM-L6-v2.
# This model typically returns cosine scores in the 0.25–0.65 range for
# genuinely relevant chunks. 0.55 was silently discarding real matches,
# leading to the "could not find sufficient evidence" response even when
# chunks HAD been retrieved and passed to the LLM.
SCORE_THRESHOLD  = 0.25   # was 0.55 — that was the root cause of issue #4
TOP_K_FETCH      = 8      # fetch a few more candidates before filtering
TOP_K_USE        = 4      # use the top 4 after filtering (was 3)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚", layout="wide")

# ── SESSION STATE ──────────────────────────────────────────────────────────────
defaults = {
    "messages":  [],
    "full_text": "",
    "kb_ready":  False,
    # FIX #1 — Remove "question" from session state entirely.
    # The original code stored the question in session_state and cleared it
    # at the top of the answer block. But Streamlit reruns the entire script
    # on every widget interaction, so the question was sometimes cleared before
    # the answer block ran. The fix: keep the answer block inside the same
    # `if user_input:` branch — no session_state relay needed.
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── API KEY GUARD ──────────────────────────────────────────────────────────────
if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY missing in .env")
    st.stop()

# ── CACHED RESOURCES ───────────────────────────────────────────────────────────
# FIX #3 (speed) — both are @st.cache_resource, so they load once and are
# reused across reruns. No change needed here; this was already correct.
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME, dimension=384, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)

embedder = load_embedder()
index    = init_pinecone()

# FIX #3 (speed) — Slightly larger chunks reduce the total number of vectors
# and speed up both indexing and retrieval without hurting recall much.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # was 400
    chunk_overlap=100,    # was 80
    separators=["\n\n", "\n", ". ", " ", ""]
)

# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())

def extract_pdf(file) -> list[dict]:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        t = clean_text(page.get_text())
        if t.strip():
            pages.append({"page": i + 1, "text": t})
    doc.close()
    return pages

def page_num(meta: dict) -> int:
    return int(meta["page"])

def retrieve(question: str) -> list[dict]:
    """Embed → query Pinecone → filter by lowered threshold → return best matches."""
    emb  = embedder.encode(question, normalize_embeddings=True).tolist()
    res  = index.query(vector=emb, top_k=TOP_K_FETCH, include_metadata=True)
    hits = [m for m in res["matches"] if m["score"] >= SCORE_THRESHOLD]
    return sorted(hits, key=lambda x: x["score"], reverse=True)[:TOP_K_USE]

def ask_ollama(prompt: str) -> str:
    """Call Ollama synchronously (no streaming — avoids Streamlit websocket errors)."""
    response = chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response.message.content

def build_prompt(question: str, context_text: str) -> str:
    # FIX #4 — The old prompt told the model to respond with a fixed refusal
    # string if it was unsure. That made it too conservative — it would refuse
    # even when the context contained partial but useful information.
    # New prompt asks the model to answer from context as best it can, and
    # only say it doesn't know if there is truly NO relevant information.
    return (
        "You are a medical document Q&A assistant. Answer the user's question "
        "using ONLY the context provided below. Be thorough and use all relevant "
        "details from the context. If the context contains partial information, "
        "share what you found rather than refusing. Only say you cannot find the "
        "answer if the context contains absolutely no relevant information.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    DEBUG_MODE = st.checkbox("Debug Retrieval")
    st.divider()

    # FIX #2 — Added an explanation so the purpose of this button is clear.
    st.markdown(
        "**📥 Build Knowledge Base**\n\n"
        "Splits your PDF into text chunks, creates vector embeddings for each "
        "chunk, and stores them in Pinecone. This is required before you can ask "
        "questions. You only need to do it once per PDF."
    )
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file and st.button("📥 Build Knowledge Base"):
        with st.spinner("Building knowledge base…"):
            try:
                stats = index.describe_index_stats()
                if stats.get("total_vector_count", 0) > 0:
                    index.delete(delete_all=True)

                pages = extract_pdf(uploaded_file)
                if not pages:
                    st.error("No text found in PDF.")
                    st.stop()

                st.session_state.full_text = "\n".join(p["text"] for p in pages)
                seen, vectors = set(), []

                for page in pages:
                    chunks = splitter.split_text(page["text"])
                    unique = [c for c in chunks if c not in seen and not seen.add(c)]
                    if not unique:
                        continue
                    # FIX #3 (speed) — batch_size=128 is faster than 64 on CPU
                    embs = embedder.encode(
                        unique,
                        batch_size=128,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    for chunk, emb in zip(unique, embs):
                        vectors.append((
                            str(uuid.uuid4()),
                            emb.tolist(),
                            {"text": chunk, "page": page["page"]}
                        ))

                # FIX #3 (speed) — upsert in batches of 200 instead of 100
                for i in range(0, len(vectors), 200):
                    index.upsert(vectors=vectors[i:i + 200])

                st.session_state.kb_ready = True
                st.success(f"✅ Ready — {len(vectors)} chunks indexed.")

            except Exception as e:
                st.error(f"Build error: {e}")

    st.divider()

    if st.button("📄 Summarize Document"):
        if not st.session_state.full_text:
            st.warning("Build knowledge base first.")
        else:
            with st.spinner("Summarizing…"):
                try:
                    prompt = (
                        "Summarize the document concisely:\n"
                        "- Key themes\n"
                        "- Important findings\n"
                        "- Important people or entities\n\n"
                        f"Document:\n{st.session_state.full_text[:12000]}"
                    )
                    st.write(ask_ollama(prompt))
                except Exception as e:
                    st.error(f"Summary error: {e}")

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔴 Kill Session"):
        for k, v in defaults.items():
            st.session_state[k] = v
        gc.collect()
        st.success("Session cleared.")

# ── TITLE + CHAT HISTORY ───────────────────────────────────────────────────────
st.title("📚 PDF RAG Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── CHAT INPUT + ANSWER ────────────────────────────────────────────────────────
# FIX #1 — The original code stored the question in session_state so that a
# separate block below could pick it up after a rerun. This caused the first
# submission to be silently dropped because the answer block ran on a stale
# empty value before the state was properly committed.
#
# The fix is simple: handle the question directly inside the `if user_input:`
# branch in the SAME script execution. No session_state relay, no rerun needed.
# chat_input() is guaranteed to return a non-empty string exactly once per
# submission — on the very next script run after the user presses Enter.
user_input = st.chat_input("Ask a question about the document…")

if user_input:
    question = user_input.strip()

    # Show user bubble and record in history
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Knowledge base not ready
    if not st.session_state.kb_ready:
        reply = "⚠️ Please upload a PDF and click **Build Knowledge Base** first."
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching document…"):
                try:
                    matches = retrieve(question)

                    if DEBUG_MODE:
                        with st.expander("🔍 Retrieval Debug"):
                            st.write(f"Query: `{question}`")
                            st.write(f"Threshold: `{SCORE_THRESHOLD}` | Matches found: `{len(matches)}`")
                            for i, m in enumerate(matches, 1):
                                st.write(
                                    f"**#{i}** Score: `{m['score']:.4f}` | "
                                    f"Page: `{page_num(m['metadata'])}`"
                                )
                                st.caption(m["metadata"]["text"][:400])
                                st.divider()

                    if not matches:
                        reply = (
                            "I could not find relevant information in the document "
                            "for your question. Try rephrasing or check that the "
                            "knowledge base has been built from the correct PDF."
                        )
                    else:
                        context_text = "\n\n---\n\n".join(
                            m["metadata"]["text"] for m in matches
                        )
                        citations = ", ".join(
                            sorted(
                                {f"Page {page_num(m['metadata'])}" for m in matches},
                                key=lambda s: int(s.split()[1])
                            )
                        )

                        answer = ask_ollama(build_prompt(question, context_text))
                        reply  = f"{answer}\n\n**Source:** {citations}"

                    st.markdown(reply)

                except Exception as e:
                    reply = f"❌ Error: {e}"
                    st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
