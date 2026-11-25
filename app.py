"""RAG Document Chatbot - Streamlit + TF-IDF retrieval (+ optional LLM answers)."""
import io
import os
import re
from datetime import datetime

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RAG Document Chatbot", page_icon="📄", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1100px; }

/* Hero banner */
.hero {
  background: linear-gradient(120deg, #2438d8 0%, #6a2fe0 50%, #e0245e 100%);
  background-size: 200% 200%; animation: shift 10s ease infinite;
  border-radius: 20px; padding: 32px 36px; color: #fff;
  box-shadow: 0 14px 40px rgba(106, 47, 224, .35); margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; color: #fff; }
.hero p  { margin: 8px 0 0; font-size: 1.02rem; opacity: .92; }
@keyframes shift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

/* Tag chips */
.chips { display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0 22px; }
.chip {
  background: #eef0ff; color: #3b3fc9; border: 1px solid #d9dcff;
  padding: 5px 14px; border-radius: 999px; font-size: .82rem; font-weight: 600;
}

/* Metric cards */
.metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 26px; }
.metric {
  background: #fff; border: 1px solid #ececf6; border-radius: 16px; padding: 16px 20px;
  box-shadow: 0 4px 16px rgba(30, 30, 90, .06); transition: transform .15s ease;
}
.metric:hover { transform: translateY(-3px); }
.metric .label { color: #7a7a95; font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.metric .value { color: #1f1f3a; font-size: 2.2rem; font-weight: 800; line-height: 1.2; }

/* Section titles */
.section-title { font-size: 1.35rem; font-weight: 700; margin: 8px 0 10px; color: #1f1f3a; }

/* Buttons */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: 12px; border: 1px solid #d9dcff; background: #fff; color: #3b3fc9;
  font-weight: 600; padding: .5rem 1rem; transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
  background: linear-gradient(120deg, #2438d8, #6a2fe0); color: #fff; border-color: transparent;
  transform: translateY(-2px); box-shadow: 0 8px 20px rgba(60, 60, 220, .3);
}
.stFormSubmitButton > button { background: linear-gradient(120deg, #2438d8, #6a2fe0); color: #fff; border: none; }

/* Inputs */
.stTextInput input { border-radius: 12px; border: 1.5px solid #d9dcff; padding: 12px 14px; }
.stTextInput input:focus { border-color: #6a2fe0; box-shadow: 0 0 0 3px rgba(106, 47, 224, .15); }

/* Chat bubbles */
.q-bubble {
  background: linear-gradient(120deg, #2438d8, #6a2fe0); color: #fff; padding: 12px 18px;
  border-radius: 18px 18px 4px 18px; margin: 18px 0 8px auto; width: fit-content; max-width: 85%;
  box-shadow: 0 6px 18px rgba(60, 60, 220, .25); font-weight: 500;
}
.a-bubble {
  background: #f6f7ff; border: 1px solid #e3e5ff; border-left: 5px solid #6a2fe0;
  padding: 16px 20px; border-radius: 6px 18px 18px 18px; margin-bottom: 8px; color: #23233f;
  line-height: 1.6;
}
.badge {
  display: inline-block; font-size: .72rem; font-weight: 700; padding: 3px 10px; border-radius: 999px;
  margin-right: 6px; background: #e6f7ee; color: #14804a;
}
.badge.llm { background: #fff1e0; color: #b45f06; }
.badge.src { background: #eef0ff; color: #3b3fc9; }
.score-bar { height: 6px; border-radius: 6px; background: #e9eaf7; overflow: hidden; margin: 4px 0 10px; }
.score-fill { height: 100%; background: linear-gradient(90deg, #2438d8, #e0245e); }

/* Sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #f1f3ff 0%, #fbf1f6 100%); }
section[data-testid="stSidebar"] h2 { font-weight: 800; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sample documents (used when nothing is uploaded)
# ----------------------------------------------------------------------------
SAMPLE_DOCS = {
    "ml_basics.txt": (
        "Overfitting happens when a model learns the training data too well, including its noise, "
        "so it performs poorly on new unseen data. Signs of overfitting are very high training "
        "accuracy but much lower validation accuracy.\n\n"
        "Ways to reduce overfitting include getting more training data, using regularization such as "
        "L1 or L2 penalties, applying dropout in neural networks, early stopping, and simplifying the model."
    ),
    "rag_overview.txt": (
        "Retrieval-Augmented Generation (RAG) combines a retriever with a language model. The retriever "
        "searches a document collection and returns the passages most relevant to the user's question.\n\n"
        "The retriever in RAG converts the question and the document chunks into vectors, for example using "
        "TF-IDF or embeddings, and ranks chunks by similarity. The top-k passages are then passed to the "
        "generator as context so the answer is grounded in the source text.\n\n"
        "RAG reduces hallucination because the model answers from retrieved evidence instead of memory alone."
    ),
    "prompting.txt": (
        "Few-shot prompting means giving a language model a small number of worked examples inside the "
        "prompt so it can infer the task and the output format without any fine-tuning.\n\n"
        "Zero-shot prompting gives no examples, only an instruction. Few-shot prompts usually improve "
        "accuracy and consistency, especially for formatting and classification tasks."
    ),
}

EXAMPLE_QUESTIONS = [
    "What is overfitting?",
    "What does the retriever do in RAG?",
    "What is few-shot prompting?",
]

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def read_upload(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            st.sidebar.error("Install pypdf to read PDFs:  pip install pypdf")
            return ""
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    return data.decode("utf-8", errors="ignore")


def chunk_text(text: str, max_words: int):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur, count = [], [], 0
    for p in paras:
        w = len(p.split())
        if cur and count + w > max_words:
            chunks.append("\n\n".join(cur))
            cur, count = [], 0
        cur.append(p)
        count += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


@st.cache_data(show_spinner=False)
def build_index(docs: tuple, max_words: int):
    chunks = []
    for source, text in docs:
        for i, c in enumerate(chunk_text(text, max_words), 1):
            chunks.append({"source": source, "id": i, "text": c})
    if not chunks:
        return chunks, None, None
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform([c["text"] for c in chunks])
    return chunks, vec, matrix


def retrieve(query, chunks, vec, matrix, k):
    scores = cosine_similarity(vec.transform([query]), matrix).ravel()
    order = scores.argsort()[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def best_sentences(query, text, n=2):
    q_terms = set(re.findall(r"\w+", query.lower()))
    sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    ranked = sorted(sents, key=lambda s: len(q_terms & set(re.findall(r"\w+", s.lower()))), reverse=True)
    keep = ranked[:n]
    return " ".join(s for s in sents if s in keep)


def llm_answer(query, hits, api_key):
    import anthropic

    context = "\n\n".join(f"[{i}] ({c['source']}) {c['text']}" for i, (c, _) in enumerate(hits, 1))
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=600,
        system="Answer using ONLY the numbered context passages. Cite passages like [1]. "
        "If the answer is not in the context, say you don't know.",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    )
    return msg.content[0].text


def answer_question(query: str):
    hits = retrieve(query, chunks, vec, matrix, st.session_state.top_k) if vec is not None else []
    if not hits:
        st.session_state.history.append(
            {"q": query, "a": "I couldn't find anything relevant in the documents.", "hits": [], "mode": "none",
             "time": datetime.now().strftime("%H:%M")}
        )
        return
    mode, text = "extractive", None
    if api_key:
        try:
            with st.spinner("Asking the LLM…"):
                text, mode = llm_answer(query, hits, api_key), "llm"
        except Exception as e:  # fall back gracefully
            st.sidebar.warning(f"LLM call failed, using extractive mode.\n\n{e}")
    if text is None:
        text = best_sentences(query, hits[0][0]["text"])
    st.session_state.history.append(
        {"q": query, "a": text, "hits": hits, "mode": mode, "time": datetime.now().strftime("%H:%M")}
    )


# ----------------------------------------------------------------------------
# Session state + callbacks (all buttons use callbacks so they respond first click)
# ----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("pending", None)
ss.setdefault("top_k", 3)


def queue_question(q):
    ss.pending = q


def submit_form():
    q = ss.get("question_box", "").strip()
    if q:
        ss.pending = q
    ss.question_box = ""


def clear_chat():
    ss.history = []
    ss.pending = None


def reset_all():
    ss.history = []
    ss.pending = None
    ss.top_k = 3
    ss.chunk_words = 120
    ss.uploader_key = ss.get("uploader_key", 0) + 1
    build_index.clear()


ss.setdefault("chunk_words", 120)
ss.setdefault("uploader_key", 0)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    files = st.file_uploader(
        "Upload documents", type=["pdf", "txt", "md"], accept_multiple_files=True,
        key=f"uploader_{ss.uploader_key}",
    )
    st.slider("Passages to retrieve (top-k)", 1, 8, key="top_k")
    st.slider("Chunk size (words)", 40, 400, step=20, key="chunk_words")
    api_key = st.text_input(
        "Anthropic API key (optional)", type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Leave empty to use extractive mode.",
    )
    if api_key:
        st.success("LLM mode: answers are written by the model from retrieved passages.")
    else:
        st.info("Extractive mode: showing the best matching passage (no API key set).")

    st.markdown("---")
    st.button("🧹 Clear chat", on_click=clear_chat, use_container_width=True)
    st.button("🔄 Reset everything", on_click=reset_all, use_container_width=True)
    if not files:
        st.caption("No files uploaded? The built-in sample documents are used.")

# ----------------------------------------------------------------------------
# Index
# ----------------------------------------------------------------------------
docs = [(f.name, read_upload(f)) for f in files] if files else list(SAMPLE_DOCS.items())
docs = [(n, t) for n, t in docs if t.strip()]
chunks, vec, matrix = build_index(tuple(docs), ss.chunk_words)

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>📄 RAG Document Chatbot</h1>
  <p>Ask questions about your documents and get answers backed by the source text.</p>
</div>
<div class="chips">
  <span class="chip">Retrieval</span><span class="chip">TF-IDF</span>
  <span class="chip">LLM</span><span class="chip">Python</span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="metrics">
  <div class="metric"><div class="label">Documents</div><div class="value">{len(docs)}</div></div>
  <div class="metric"><div class="label">Chunks indexed</div><div class="value">{len(chunks)}</div></div>
  <div class="metric"><div class="label">Top-k</div><div class="value">{ss.top_k}</div></div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">💬 Ask a question</div>', unsafe_allow_html=True)

cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, q in zip(cols, EXAMPLE_QUESTIONS):
    col.button(q, key=f"ex_{q}", on_click=queue_question, args=(q,), use_container_width=True)

with st.form("ask_form", clear_on_submit=False):
    c1, c2 = st.columns([5, 1])
    c1.text_input("Your question", key="question_box", label_visibility="collapsed",
                  placeholder="Type your question and press Enter…")
    c2.form_submit_button("Ask ➤", on_click=submit_form, use_container_width=True)

# Run any queued question (from example buttons or the form)
if ss.pending:
    q, ss.pending = ss.pending, None
    answer_question(q)

# ----------------------------------------------------------------------------
# Conversation
# ----------------------------------------------------------------------------
if not ss.history:
    st.caption("Pick an example above or type your own question to get started.")
else:
    top = st.columns([1, 1, 4])
    md = "\n\n".join(f"**Q:** {h['q']}\n\n**A:** {h['a']}" for h in ss.history)
    top[0].download_button("⬇️ Download chat", md, file_name="chat.md", mime="text/markdown")
    top[1].button("🗑️ Clear", key="clear_main", on_click=clear_chat)

    for n, h in enumerate(reversed(ss.history)):
        st.markdown(f'<div class="q-bubble">🙋 {h["q"]}</div>', unsafe_allow_html=True)
        badge = '<span class="badge llm">LLM</span>' if h["mode"] == "llm" else '<span class="badge">Extractive</span>'
        st.markdown(
            f'<div class="a-bubble">{badge}<span class="badge src">{len(h["hits"])} sources</span>'
            f'<span style="float:right;color:#9a9ab5;font-size:.75rem">{h["time"]}</span><br><br>{h["a"]}</div>',
            unsafe_allow_html=True,
        )
        for i, (c, score) in enumerate(h["hits"], 1):
            with st.expander(f"📚 Source {i}: {c['source']} · chunk {c['id']} · match {score:.0%}", expanded=False):
                st.markdown(
                    f'<div class="score-bar"><div class="score-fill" style="width:{min(score,1)*100:.0f}%"></div></div>',
                    unsafe_allow_html=True,
                )
                st.write(c["text"])