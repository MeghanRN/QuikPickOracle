# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# src/app.py
"""QuikPick Oracle Streamlit application.

This file glues together several moving parts to provide a question‑and‑answer
experience for field technicians.  A newcomer to large language models (LLMs)
should think of this script as a recipe:

1.  Collect the user’s question from a simple web interface.
2.  Look up related troubleshooting information in a vector database.
3.  Send both the question and the related context to a locally hosted
    LLM for a final answer.
4.  Record feedback so the system can be improved later.

Every major block below is annotated to explain *why* it exists and *how* it
fits into that recipe.
"""

# The `pysqlite3` package bundles a modern SQLite build.  We import it and
# then replace Python's default `sqlite3` module so the rest of the code can use
# SQLite features transparently.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import re
import csv
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

import streamlit as st
from chromadb import PersistentClient
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
from llama_cpp import Llama

from prompt_templates import SYSTEM_TEMPLATE, build_prompt
from feedback_db import save as save_feedback          
from feedback_db import _append_positive, _append_negative  
from rapidfuzz import fuzz, process 
from streamlit_feedback import streamlit_feedback

# ── SESSION SETUP ──────────────────────────────────────────────────────────
# `st.session_state` stores variables that survive across user interactions.
# These defaults ensure the keys exist even on the very first page load.
st.session_state.setdefault("to_log", [])        # feedback to store later
st.session_state.setdefault("assistant_meta", {})# assistant messages metadata
st.session_state.setdefault("pending_q", None)   # question waiting to be asked
st.session_state.setdefault("is_thinking", False)# flag while the LLM works

# Load small avatar images so chat bubbles look friendly.
bot_avatar  = Image.open("images/sphere.png")
user_avatar = Image.open("images/image.png")

# Configure the Streamlit page: title, browser tab icon and layout.
st.set_page_config(
    page_title="QuikPick Oracle",
    page_icon="images/logo.png",
    layout="centered",
)

# Display the project logo centered on the page.
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("images/logo.png", width=350)   # adjust width to taste


# ── CONFIG ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("model/Llama-3-13B-Instruct-Q4_K_M.gguf")
LORA_PATH  = "models/lora-adapter"
VECTOR_DIR = "vectorstore"
ERROR_RE   = re.compile(r"^\d+_\d+$")
MAX_TOKENS = 512
MEM_TURNS  = 8

# ── CACHES ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading local Llama model…")
def load_llm() -> Llama:
    """Load the quantized Llama model from disk.

    The ``llama_cpp`` library runs the GGUF model completely locally.  We
    configure a context window large enough for our prompts and return the
    initialized :class:`~llama_cpp.Llama` instance.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run 'python download_model.py' "
            "to fetch it."
        )
    return Llama(model_path=str(MODEL_PATH), n_ctx=4096)
def run_llm(prompt: str) -> str:
    """Send a prompt to the local LLM and return only its text reply.

    ``prompt`` is a long string constructed elsewhere that contains system
    instructions, any relevant context documents and the user's question.  This
    function hides the slightly verbose API call and extracts just the text of
    the assistant's answer.
    """
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are QuikPick Oracle."},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.2,
        top_p=0.95,
    )
    # extract the assistant’s reply
    return (resp["choices"][0]["message"]["content"] or "").strip()

@st.cache_resource(show_spinner="Opening vector store…")
def load_store():
    """Connect to the on-disk vector database.

    We use `ChromaDB`'s ``PersistentClient`` which stores embeddings and
    metadata on disk.  The collection named ``"errors"`` contains troubleshooting
    information indexed by :mod:`ingest.py`.
    """
    return PersistentClient(path=VECTOR_DIR).get_collection("errors")

@st.cache_resource(show_spinner="Loading embedder…")
def load_embedder():
    """Load the sentence transformer used to embed text.

    The model converts short pieces of text into numeric vectors.  These vectors
    let us compare questions with stored documents using cosine similarity.
    """
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

llm      = load_llm()
store    = load_store()
embedder = load_embedder()

# ── HELPERS ─────────────────────────────────────────────────────────────────
# helpers.py  
def click_fup(q: str):
    """Handle a user clicking on a suggested follow‑up question.

    The app presents the user with both pre‑scripted and model‑generated follow‑up
    questions.  When one is clicked we may need to advance an internal step
    counter (so the next canned question can appear) and we stage the selected
    question so it will be asked on the next rerun of the Streamlit script.
    """
    # Walk through all documents loaded for the current error code.  If the
    # clicked question matches one of our hand‑written follow‑ups we increment
    # ``step_counter`` so that the subsequent step becomes available.
    for d in st.session_state.docs:
        meta = d["meta"]
        if meta.get("IsFollowUp") and meta.get("Question") == q:
            st.session_state.step_counter += 1
            break
    # Store the question so that on the next page refresh it will be treated as
    # the user's input.
    st.session_state.next_q = q

def get_answer(prompt: str, *, max_retry: int = 1) -> tuple[str, str | None, str | None]:
    """Ask the LLM for an answer and interpret optional image directives.

    The model occasionally returns a special ``<SHOW>`` tag instructing the app
    to display an image from our knowledge base.  This function extracts that
    tag, locates the matching image, and removes the directive from the answer
    text.  It also guards against a common failure mode where the model only
    outputs follow‑up questions without an actual answer; in that case we retry
    once with a gentle reminder.
    """
    raw = run_llm(prompt)

    show_pat = re.compile(r"<SHOW>\s*(?:<([^>]+)>|(\S+))", re.I)
    m = show_pat.search(raw)
    img_path: str | None = None
    img_caption: str | None = None

    if m:
        fname = m.group(1)                                   # e.g. "anatomy_part2.png"

        img_doc = next(
            (
                d for d in st.session_state.docs
                if d["meta"].get("IsImage")
                and Path(d["meta"]["filepath"]).name == fname
            ),
            None,
        )

        if img_doc:
            img_path    = img_doc["meta"]["filepath"]          # still the full string path
            img_caption = img_doc["meta"]["Caption"]

        raw = show_pat.sub("", raw).strip()  # strip the <SHOW> line


    tries = 0
    while raw.lstrip().lower().startswith("<Follow‑Up>") and tries < max_retry:
        tries += 1
        raw = run_llm(
            prompt
            + "\n\n### Oracle Note\n"
            + "Your previous reply started with the Follow‑Up header and did not "
            + "contain an answer. Please begin with a complete answer first, then "
            + "add the Follow‑Up block."
        )

    return raw, img_path, img_caption


def count_similar_questions(q: str, questions: List[str], threshold: int = 90) -> int:
    """Count how often a new question repeats something already asked.

    ``rapidfuzz`` compares two strings and returns a similarity score from
    ``0`` (completely different) to ``100`` (identical).  If the user's new
    question is too similar to previous ones we can escalate to a human.
    ``threshold`` controls how close two questions must be to be considered a
    repeat.
    """
    return sum(
        1
        for prev in questions
        if fuzz.token_set_ratio(q, prev) >= threshold
    )

QA_PATH = Path("data/sample_qa.csv")

def append_qa_if_new(code: str, question: str, answer: str) -> None:
    """Store new Q&A pairs so they can be reused as scripted answers.

    When the assistant gives a particularly good answer we may want to include
    it as a canned response in future versions of the tool.  This helper writes
    the trio (error code, question, answer) to ``data/sample_qa.csv`` unless an
    identical row already exists.
    """
    QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = QA_PATH.exists()
    answer = answer.replace("\n", r"\n")
    # Read existing rows to avoid duplicates.
    seen = set()
    if exists:
        with QA_PATH.open(newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    seen.add(tuple(row[:3]))

    row = (code.strip(), question.strip(), answer.strip())
    if row not in seen:
        with QA_PATH.open("a", newline="") as f:
            writer = csv.writer(f)
            # Write header once if the file was just created
            if not exists:
                writer.writerow(["ErrorCode", "Question", "Answer"])
            writer.writerow(row)

def _rerun():
    """Trigger a fresh execution of the Streamlit script.

    Streamlit reruns the entire file from top to bottom whenever the user
    interacts with a widget.  This helper hides the slight version difference
    between ``st.rerun`` (newer API) and the older ``experimental_rerun`` so the
    rest of the code can simply call ``_rerun()``.
    """
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# ── HELPERS ─────────────────────────────────────────────────────────────────
QA_PATH = Path("data/sample_qa.csv")

@st.cache_resource(show_spinner="Loading docs…")
def docs_for_code(code: str) -> List[Dict[str, Any]]:
    """Fetch all documents related to a specific error code.

    The vector store contains several types of entries:
    * **Solutions** for each error code.
    * **Follow‑up questions** that form a step‑by‑step decision tree.
    * **Reference images** that may be shown alongside answers.
    * **Canned Q&A pairs** collected from previous interactions.

    This function gathers all of those pieces so that later steps can perform
    similarity search against them.
    """
    docs: list[dict] = []

    # 1) Retrieve rows that belong to the requested error code.
    res_code = store.get(
        where={"ErrorCode": code},
        include=["documents", "metadatas", "embeddings"],
    )

    # 2) Retrieve any global images (they have IsImage=True).
    res_img = store.get(
        where={"IsImage": True},
        include=["documents", "metadatas", "embeddings"],
    )

    # Helper to push results from a Chroma query into ``docs``.
    def _append(res):
        for d, m, e in zip(res["documents"], res["metadatas"], res["embeddings"]):
            if e is None:                       # In rare cases embeddings may be missing
                e = embedder.encode([d])[0]     # so generate them on the fly.
            docs.append({"content": d, "meta": m, "embedding": e})

    _append(res_code)
    _append(res_img)

    # 3) Append any canned QA stored on disk.
    if QA_PATH.exists():
        reader = csv.DictReader(QA_PATH.open())
        for row in reader:
            if row["ErrorCode"] == code or row["ErrorCode"].strip() == "*":
                docs.append({
                    "content": f"Q: {row['Question']}\nA: {row['Answer']}",
                    "meta": {"IsQA": True, "Question": row["Question"]},
                    "embedding": embedder.encode([row["Question"]])[0],
                })

    return docs


def retrieve_similar(q: str, docs: list[dict], k: int = 3) -> list[dict]:
    """Return the ``k`` documents most relevant to question ``q``.

    The function first tries to find an exact match among the canned Q&A pairs
    using fuzzy string matching.  If a high‑scoring match is found we return it
    immediately because those answers are usually authoritative.  Otherwise we
    compute cosine similarity between the question and every document's
    embedding and return the top ``k`` items.
    """
    # 1) Attempt to match against any pre-written Q&A entries.
    qa_hit = best_qa_match(q, docs, min_score=90)
    if qa_hit:
        return [qa_hit]  # give the “gold” answer only

    # 2) Otherwise fall back to embedding similarity.
    q_vec = embedder.encode([q])[0]
    scored = sorted(
        docs,
        key=lambda d: dot(d["embedding"], q_vec) /
                      (norm(d["embedding"]) * norm(q_vec)),
        reverse=True,
    )
    return scored[:k]

def best_qa_match(user_q: str, docs: list[dict], min_score: int = 90):
    """Find the canned question most similar to ``user_q``.

    ``rapidfuzz`` provides efficient fuzzy string matching.  We build a list of
    all stored questions and let the library score each against ``user_q``.  If
    the best score is above ``min_score`` we return the corresponding document;
    otherwise ``None`` indicates that no close match exists.
    """
    # Build a list of (question_text, index) for only the QA rows.
    choices = [
        (d["meta"]["Question"], i)
        for i, d in enumerate(docs)
        if d["meta"].get("Question")
    ]
    if not choices:
        return None

    # Rapidfuzz finds the best match quickly.
    match, score, idx = process.extractOne(
        user_q,
        choices,
        scorer=fuzz.token_set_ratio  # good for re-ordering / extra words
    )
    return docs[idx] if score >= min_score else None

def qa_answer(doc: dict) -> str:
    """Pull just the answer text from a canned Q&A document.

    The :func:`ingest` script stores Q&A pairs in the simple format
    ``"Q: ...\nA: ..."``.  This helper splits on ``"A:"`` and returns the answer
    portion, falling back to the full content if the pattern is unexpected.
    """
    parts = doc["content"].split("A:", 1)
    return parts[1].strip() if len(parts) > 1 else doc["content"].strip()

# ── SESSION STATE ──────────────────────────────────────────────────────────
# These keys track the overall conversation.  ``code`` stores the currently
# selected error code, ``docs`` holds the knowledge-base entries for that code
# and ``history`` is the list of chat messages.  ``step_counter`` remembers how
# many canned follow‑up steps the user has already taken.
st.session_state.setdefault("code",   None)
st.session_state.setdefault("docs",   [])
st.session_state.setdefault("history", [])
st.session_state.setdefault("step_counter", 0) # canned follow ups (steps the user has clicked so far)

# ── UI ───────────────────────────────────────────────────────────────────────

# 1) Select error code
# The conversation always begins by identifying which error code the technician
# is dealing with.  Once a valid code is entered we load the related documents
# and restart the script so the main chat interface appears.
if st.session_state.code is None:
    code = st.text_input("Enter an error code (e.g. 2_4) to begin:")
    if code and ERROR_RE.match(code):
        bundle = docs_for_code(code)
        if bundle:
            st.session_state.code = code
            st.session_state.docs = bundle
            _rerun()
        else:
            st.error("Error code not found.")
    st.stop()

# 2) Show a collapsible banner describing the error code.
main = next((d for d in st.session_state.docs if "Message" in d["meta"]), None)
if main is None:
    st.error("Error code not found.")
    st.stop()

with st.expander("Error-code details", expanded=False):
    st.markdown(
        f"**Error Code:** {main['meta']['ErrorCode']}  \n"
        f"**Message:** {main['meta']['Message']}  \n"
        f"**Solution:** {main['meta']['Solution']}"
    )

# Offer the first scripted follow-up question (Step 1) when the conversation
# is just starting.
if st.session_state.step_counter == 0:
    first_fu = [
        d for d in st.session_state.docs
        if d["meta"].get("IsFollowUp")
        and d["meta"]["ErrorCode"] == st.session_state.code
        and d["meta"]["StepIndex"] == 1
    ]
    if first_fu:
        q1 = first_fu[0]["meta"]["Question"]
        st.button(
            q1,
            key="canned-init",
            on_click=click_fup,
            args=(q1,),
        )

# 3) Replay chat history AND inject feedback widgets
# Streamlit reruns this script on every user interaction.  To give the illusion
# of a persistent chat we redraw every previous message stored in
# ``st.session_state.history``.

# Identify the index of the last assistant message; we only show follow‑up
# buttons under the most recent reply.
last_ai_idx = max(
    (i for i, m in enumerate(st.session_state.history) if isinstance(m, AIMessage)),
    default=None
)
for i, msg in enumerate(st.session_state.history):
    if isinstance(msg, HumanMessage):
        # Render user messages with the user avatar.
        st.chat_message("user", avatar=user_avatar).markdown(msg.content)

    else:
        # Assistant messages can include images, follow‑up questions and a
        # feedback widget, so we handle them in a dedicated block.
        with st.chat_message("assistant", avatar=bot_avatar):
            st.markdown(msg.content)
            meta = st.session_state["assistant_meta"].get(i, {})
            if meta.get("img"):
                st.image(meta["img"], width=300)

            # Ensure metadata exists even if this entry came from an older run.
            if i not in st.session_state["assistant_meta"]:
                st.session_state["assistant_meta"][i] = {
                    "q": st.session_state.history[i-1].content,
                    "a": msg.content,
                    "fups": "",
                }

            meta      = st.session_state["assistant_meta"][i]
            llm_fups  = meta.get("fups", "")

            # a) Step‑specific canned follow‑up
            # Determine the next step number and search for matching docs.
            next_step = st.session_state.step_counter + 1
            canned = [
                d for d in st.session_state.docs
                if d["meta"].get("IsFollowUp")
                and d["meta"]["ErrorCode"] == st.session_state.code
                and d["meta"]["StepIndex"] == next_step
            ]
            if i == last_ai_idx and canned:
                st.divider()
                for j, doc in enumerate(canned):
                    q = doc["meta"]["Question"]
                    st.button(
                        q,
                        key=f"canned-{i}-{j}",
                        on_click=click_fup,
                        args=(q,),
                    )

            # b) LLM‑generated follow‑ups
            if i == last_ai_idx and llm_fups:
                st.divider()
                lines = [l for l in llm_fups.splitlines() if l.strip()][:3]
                for j, line in enumerate(lines):
                    q = line.lstrip("0123456789.- ").strip()
                    if q:
                        st.button(
                            q,
                            key=f"fup-{j}-{i}",
                            on_click=click_fup,
                            args=(q,),
                        )

            # c) canned Q&A follow‑ups
            #canned = [
            #    d["meta"]["Question"]
            #    for d in st.session_state.docs
            #    if d["meta"].get("IsQA")
            #]
            #if canned:
            #    st.divider()
            #    st.markdown("**Canned follow‑ups:**")
            #    for j, q in enumerate(canned):
            #        st.button(
            #            q,
            #            key=f"qa-{j}-{i}",
            #            on_click=click_fup,
            #            args=(q,),
            #        )            
            
            # ── Unified feedback widget ──
            # Present thumbs-up/down buttons so users can rate this response.
            widget_key   = f"fb_{i}"
            persist_key  = f"fb_score_{i}"
            prev_score   = st.session_state.get(persist_key)
            disable_icon = None if prev_score is None else ("👍" if prev_score == 1 else "👎")

            # Render the component and capture raw response dict
            resp = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Tell us more",
                disable_with_score=disable_icon,
                key=widget_key,
                align="flex-end",
            )

            # Only handle the first feedback submission per message.
            if resp is not None and persist_key not in st.session_state:
                raw_score = resp["score"]
                # 1) try to cast directly (handles "1" or 0/1)
                try:
                    score = int(raw_score)
                except (ValueError, TypeError):
                    # 2) fallback: look for "+1" or "👍" in string
                    s = str(raw_score)
                    score = 1 if ("+1" in s or "👍" in s) else 0

                text   = (resp.get("text") or "").strip()
                rating = 5 if score == 1 else 1

                # Persist structured feedback to the database.
                save_feedback(
                    st.session_state.code,
                    st.session_state["assistant_meta"][i]["q"],
                    st.session_state["assistant_meta"][i]["a"],
                    rating=rating,
                    comment=text,
                )
                # For 👍 also append the canned QA for future reuse and log both
                # positive and negative examples separately.
                if score == 1:
                    append_qa_if_new(
                        st.session_state.code,
                        st.session_state["assistant_meta"][i]["q"],
                        st.session_state["assistant_meta"][i]["a"],
                    )
                    _append_negative(
                        st.session_state.code,
                        st.session_state["assistant_meta"][i]["q"],
                        st.session_state["assistant_meta"][i]["a"],
                        text or "<no comment>",
                    )

                # Remember the numeric score so disable_with_score works on the
                # next rerun and thank the user.
                st.session_state[persist_key] = score
                st.toast("Thanks for the feedback!")


# 4) New question or follow‑up input (OUTSIDE the for‑loop)

# a) If a follow‑up button was clicked, stage it and lock input
fup = st.session_state.pop("next_q", None)
if fup:
    st.session_state.pending_q   = fup
    st.session_state.is_thinking = True
    _rerun()

# b) Show the input box, disabled if we’re “thinking”
typed = st.chat_input(
    "Ask a question …",
    key="main_input",
    disabled=st.session_state.is_thinking,
)
# If the user typed while not already pending, stage it
if typed and st.session_state.pending_q is None:
    st.session_state.pending_q   = typed
    st.session_state.is_thinking = True
    _rerun()

# c) If there’s a staged prompt, process it
if st.session_state.pending_q:
    user_q = st.session_state.pending_q
    st.session_state.pending_q = None

    # 1) ALWAYS show the user’s question as a chat bubble
    st.chat_message("user", avatar=user_avatar).write(user_q)
    st.session_state.history.append(HumanMessage(content=user_q))

    # 2) Check if it’s one of our canned follow‑ups
    canned = next(
        (
            d for d in st.session_state.docs
            if d["meta"].get("IsFollowUp")
            and d["meta"]["Question"] == user_q
        ),
        None
    )
    # 3) If it’s canned, show it and return immediately
    if canned:
        answer = canned["meta"]["Answer"].strip('"').strip("'")
        st.chat_message("assistant", avatar=bot_avatar).markdown(answer)
        st.session_state.history.append(AIMessage(content=answer))
        st.session_state.is_thinking = False
        _rerun()

    # ── fuzzy-repeat check goes here ──
    past_qs = [
        m.content
        for m in st.session_state.history
        if isinstance(m, HumanMessage)
    ]
    # count how many past questions look like this one
    repeat_count = count_similar_questions(user_q, past_qs, threshold=90)
    # if they’ve asked “the same” >3 times, escalate
    if repeat_count > 2:
        st.session_state.history.append(
            AIMessage(content="Thank you for your patience. \n\nPlease contact the QuikPick team. \nYou will need: \n1. A photo of the Service UI BasicInfo section \n2. the Jupiter PCSN \n\nEnsure you mention the error code to the team")
        )
        # clear the “thinking” flag so we don’t lock the input
        st.session_state.is_thinking = False
        _rerun()

    # ── pull context, call LLM ──
    with st.spinner("Thinking…"):
        # 1) always grab the main error-code doc
        main_doc = next(
            d for d in st.session_state.docs
            if d["meta"].get("ErrorCode") == st.session_state.code
               and "Message" in d["meta"]
        )

        # 2) get the top-(k-1) most similar others
        sim_docs = retrieve_similar(user_q, st.session_state.docs, k=5)
        # drop main_doc if it snuck in
        sim_docs = [d for d in sim_docs if d is not main_doc]

        # 3) assemble final ctx_docs list
        ctx_docs = [d for d in ([main_doc] + sim_docs[:2]) if not d["meta"].get("IsImage")]

        # 4) build the prompt from exactly those
        ctx_block = "\n\n".join(d["content"] for d in ctx_docs)
        hist_txt  = "\n".join(
            m.content for m in st.session_state.history[-MEM_TURNS:]
        )
        img_doc = next((d for d in sim_docs if d["meta"].get("IsImage")), None)
        img_path = img_doc["meta"]["filepath"] if img_doc else None
        img_caption = img_doc["meta"]["Caption"] if img_doc else ""
        # collect every image doc for the *current* error‑code
        image_catalog = "\n".join(
            f"- {d['meta']['Caption']}  •  <{Path(d['meta']['filepath']).name}>"
            for d in st.session_state.docs
            if d["meta"].get("IsImage")
        )

        prompt = build_prompt(
            SYSTEM_TEMPLATE.format(image_catalog=image_catalog),   # <-- new
            ctx_block,
            image_catalog,
            hist_txt,
            user_q,
        ) + " "

        # call HF Inference API instead of local llama
        raw, img_path, img_caption = get_answer(prompt)

        if "<END>" in raw:
            raw = raw.split("<END>", 1)[0].rstrip()

    parts = re.split(r"(?i)<\s*follow[\-\u2010-\u2015\s]?up\s*>", raw, maxsplit=1)
    main_ans, llm_fups = (parts + [""])[:2]
    main_ans, llm_fups = main_ans.strip(), llm_fups.strip()

    if not main_ans:
        main_ans, llm_fups = raw.strip(), ""

    # stash Q/A + follow‑ups for replay
    mid = len(st.session_state.history)
    st.session_state["assistant_meta"][mid] = {
        "q":      user_q,
        "a":      main_ans,
        "fups":   llm_fups,
        "img":    img_path,
        "cap":    img_caption,
    }

    # append AI reply
    st.session_state.history.append(AIMessage(content=main_ans))

    # unlock input & rerun to redraw the enabled chat box
    st.session_state.is_thinking = False
    _rerun()

for kind, payload in st.session_state.pop("to_log", []):
    if kind == "pos":
        log_positive(*payload)   # type: ignore[arg-type]
    else:
        log_negative(*payload)   # type: ignore[arg-type]

# 5) Reset
if st.button("↻ Restart"):
    st.session_state.clear()
    _rerun()
