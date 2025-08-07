"""Prepare the vector database used by the Streamlit app.

The Streamlit interface relies on a *vector store* (a fancy database for
machine‑learning embeddings) to quickly retrieve relevant snippets of
troubleshooting information.  This script reads several CSV files exported from
internal documentation and indexes them into that store.

Usage
-----
    python src/ingest.py data/error_codes.csv [data/steps.csv] [data/sample_qa.csv]

The command line arguments point to CSV files with the following columns:

* ``error_codes.csv`` – **required**.  Columns: ``ErrorCode``, ``Message``, ``Solution``.
  Each row represents a high‑level problem and its suggested remedy.
* ``steps.csv`` – optional.  Columns: ``ErrorCode``, ``step``, ``Question``, ``Answer``.
  These rows represent the step‑by‑step follow‑up questions the app can ask.
* ``sample_qa.csv`` – optional.  Columns: ``ErrorCode``, ``Question``, ``Answer``.
  These are additional canned Q&A pairs, possibly shared across multiple error
  codes.
"""
import sys
import pathlib
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from PIL import Image

# Directory containing reference images that may be shown alongside answers.
IMG_DIR     = pathlib.Path("data/images")
# Name of the text embedding model; we use a small CPU‑friendly transformer.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Location on disk where ChromaDB stores its data.
VECTOR_DIR  = "vectorstore"
# Name of the collection within ChromaDB.
COLL_NAME   = "errors"
# Optional CSV mapping image file names to human‑friendly captions.
CAP_CSV   = pathlib.Path("data/image_captions.csv")


txt_model = SentenceTransformer(EMBED_MODEL, device="cpu")
# If you wanted to embed images with CLIP you could load a model here too.
#img_model = SentenceTransformer(IMG_MODEL,  device="cpu")

def _embed_text(texts: List[str]) -> np.ndarray:
    """Convert a list of strings into embedding vectors.

    We call ``encode`` on the sentence‑transformer model which returns a list of
    384‑dimensional float vectors.  ChromaDB expects ``float32`` arrays, so we
    convert the result accordingly.
    """
    return np.asarray(
        txt_model.encode(texts, batch_size=64, show_progress_bar=True),
        dtype=np.float32,
    )

#def _embed_imgs(paths):
#    # CLIP takes image paths or PIL objects
#    return np.asarray(img_model.encode([str(p) for p in paths]), dtype=np.float32)



def ingest(code_csv: pathlib.Path, steps_csv: pathlib.Path | None = None, qa_csv: pathlib.Path | None = None):
    """Read CSV files and populate the Chroma vector store.

    ``code_csv``  – file with core error codes and solutions.
    ``steps_csv`` – optional follow‑up questions.
    ``qa_csv``    – optional canned QA pairs.
    """
    # 1) Load full “Solution” entries -------------------------------------------------
    df_code = pd.read_csv(code_csv).fillna("")
    if not {"ErrorCode", "Message", "Solution"}.issubset(df_code.columns):
        sys.exit(f"{code_csv} must contain ErrorCode, Message, Solution columns")

    code_texts = df_code["Solution"].tolist()
    code_embs  =  _embed_text(code_texts)
    code_meta  = df_code[["ErrorCode", "Message", "Solution"]].to_dict("records")

    # 2) Load hand‑written steps.csv (Q/A follow‑ups) -------------------------------
    step_texts: List[str] = []
    step_meta:  List[Dict[str, Any]] = []
    step_embs  = None
    if steps_csv and pathlib.Path(steps_csv).exists():
        df_steps = pd.read_csv(steps_csv).fillna("")
        if not {"ErrorCode", "step", "Question", "Answer"}.issubset(df_steps.columns):
            sys.exit(f"{steps_csv} must contain ErrorCode, step, Question, Answer columns")

        for row in df_steps.itertuples(index=False):
            q  = row.Question
            a  = row.Answer
            step_texts.append(f"Q: {q}\nA: {a}")
            step_meta.append({
                "ErrorCode": row.ErrorCode,
                "StepIndex": int(row.step),  # which step number this is
                "Question":  q,
                "Answer":    a,
                "IsFollowUp": True,
            })

        step_embs =  _embed_text(step_texts)

    # 3) Load optional sample_qa.csv with wildcard support -----------------------
    qa_texts: List[str] = []
    qa_meta:  List[Dict[str, Any]] = []
    qa_embs  = None
    if qa_csv and pathlib.Path(qa_csv).exists():
        df_qa = pd.read_csv(qa_csv).fillna("")
        if not {"ErrorCode", "Question", "Answer"}.issubset(df_qa.columns):
            sys.exit(f"{qa_csv} must contain ErrorCode, Question, Answer columns")

        # ``all_error_codes`` is used when a row should apply to every code.
        all_error_codes = df_code["ErrorCode"].unique()

        for row in df_qa.itertuples(index=False):
            q = row.Question
            a = row.Answer

            error_code_field = row.ErrorCode.strip()  # e.g. "2_4" or "*"

            if error_code_field == "*":
                # Apply to all error codes
                target_codes = all_error_codes
            else:
                # Support comma-separated list of error codes
                target_codes = [ec.strip() for ec in error_code_field.split(",")]  # type: ignore

            for ec in target_codes:
                qa_texts.append(f"Q: {q}\nA: {a}")
                qa_meta.append({
                    "ErrorCode": ec,
                    "Question":  q,
                    "Answer":    a,
                    "IsQA":      True,
                    "IsGlobal":  error_code_field == "*",
                })

        qa_embs =  _embed_text(qa_texts)

    # 3‑b) Optional: bulk‑load reference images -------------------------------
    # Some troubleshooting steps reference diagrams or photos.  We index the
    # *captions* of those images so the LLM can decide when to show them.
    AP_CSV = pathlib.Path("data/image_captions.csv")
    ap_map: dict[str, str] = {}
    if CAP_CSV.exists():
       cap_map = dict(pd.read_csv(CAP_CSV).values)


    img_files = list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png"))
    img_docs  = []
    img_meta  = []
    img_embs  = None

    if img_files:
        for p in img_files:
            caption = cap_map.get(p.name, p.stem.replace("_", " ")) # type: ignore
            img_docs.append(caption)
            img_meta.append({
                "IsImage": True,
                "filepath": str(p),
                "Caption":  caption,
            })

        # embed CAPTIONS with the same 384‑dim model
        img_embs = _embed_text(img_docs)

    # 4) Push everything into Chroma -----------------------------------------
    client = PersistentClient(path=VECTOR_DIR)
    col    = client.get_or_create_collection(COLL_NAME)

    # a) full Solutions
    col.add(
        ids        = [f"code-{i}" for i in range(len(code_texts))],
        documents  = code_texts,
        embeddings = code_embs,    # type: ignore[arg-type]
        metadatas  = code_meta,    # type: ignore[arg-type]
    )

    # b) steps follow‑ups
    if step_texts:
        col.add(
            ids        = [f"step-{i}" for i in range(len(step_texts))],
            documents  = step_texts,
            embeddings = step_embs,   # type: ignore[arg-type]
            metadatas  = step_meta,   # type: ignore[arg-type]
        )

    # c) sample QA
    if qa_texts:
        col.add(
            ids        = [f"qa-{i}" for i in range(len(qa_texts))],
            documents  = qa_texts,
            embeddings = qa_embs,     # type: ignore[arg-type]
            metadatas  = qa_meta,     # type: ignore[arg-type]
        )
    # d) reference images
    print("DEBUG  imgs found:", len(img_files), [p.name for p in img_files][:5])
    if img_files:
        col.add(
            ids        =[f"img-{i}" for i in range(len(img_files))],
            documents  = img_docs,
            embeddings = img_embs,        # type: ignore[arg-type]
            metadatas  = img_meta,        # type: ignore[arg-type]
        )
    # Summarize what was indexed for a quick sanity check.
    print(
        f"✅ Indexed {len(code_texts)} Solutions"
        + (f", {len(step_texts)} follow‑ups" if step_texts else "")
        + (f", {len(qa_texts)} QA pairs" if qa_texts else "")
        + (f", {len(img_files)} images"    if img_files else "")

    )


if __name__ == "__main__":
    # When run as a script, parse command line arguments and kick off ingestion.
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    code_csv  = pathlib.Path(sys.argv[1])
    steps_csv = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
    qa_csv    = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None
    ingest(code_csv, steps_csv, qa_csv)

