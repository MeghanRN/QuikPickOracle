# QuikPick Oracle

QuikPick Oracle is a question-answering assistant that helps you troubleshoot QuikPick coolers. The app searches a knowledge base of error codes, common questions, and reference images, then asks a large language model to write a friendly answer.

The project has two major parts:

1. **Data ingestion** (`src/ingest.py`) – teaches the app new information by converting CSV files and images into an internal "vector store".
2. **Web application** (`src/app.py`) – a Streamlit web interface you can open in your browser to talk to the Oracle.

Follow the steps below to install, teach, and use the assistant.

---

## Understanding the technology

This project combines two ideas:

1. **Large language models (LLMs)** – neural networks trained on vast text collections. They predict the next word in a sentence and can generate coherent paragraphs. The Oracle uses Meta's [Llama‑4 Scout 17B](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) hosted on Hugging Face. When you ask a question, the LLM turns the request into an answer, limited to about 512 tokens (roughly 400 words). A low *temperature* setting keeps responses focused and less random.
2. **Retrieval** – instead of relying solely on what the LLM already knows, the app searches your own documentation. Everything you ingest is converted into numeric "embeddings" and stored in a small database. Asking a question triggers a similarity search against this database so the LLM can reference the most relevant facts.

LLMs work purely by pattern matching. They do not understand text the way humans do, nor can they browse the internet or remember past runs of the program. Instead, they process text in small chunks called **tokens** (roughly four characters each) and predict what comes next. Because the model has no built‑in memory of your CSV files, retrieval is how we supply it with the latest information.

Putting the two together is known as **Retrieval‑Augmented Generation (RAG)**. Here's the flow:

1. You ingest CSV files and images. The script `src/ingest.py` encodes every piece of text using the `all‑MiniLM‑L6‑v2` SentenceTransformer. Each embedding is a list of 384 numbers representing semantic meaning and is stored in a [Chroma](https://www.trychroma.com/) vector store.
2. When you ask the Oracle something, the question is embedded the same way. Cosine similarity identifies the most relevant rows and image captions.
3. Those results, along with your question, are sent to the LLM as context. The model then **generates** a friendly answer grounded in your data. Because the knowledge comes from your vector store, responses stay up to date and are less likely to hallucinate.

If you're new to LLMs, remember that they are pattern‑matching engines rather than true reasoning systems. Clear questions, well‑formatted CSVs, and high‑quality captions help the model perform better.

---

## 1. Prepare your environment

1. Install [Python 3.10+](https://www.python.org/) if it is not already available.
2. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. The web app connects to a hosted language model on Hugging Face. Create the file `src/.streamlit/secrets.toml` and add your personal API token:

   ```toml
   [hf]
   api_token = "YOUR_HUGGINGFACE_TOKEN"
   ```

   Keep this file private; it should not be committed to version control.

---

## 2. Understand the project layout

- `data/` – raw information that can be ingested.
  - `error_codes.csv` – list of machine error codes and their solutions.
  - `steps.csv` – optional follow-up questions and answers for each code.
  - `sample_qa.csv` – optional additional Q&A pairs.
  - `image_captions.csv` and the `data/images/` folder – optional reference images.
- `vectorstore/` – automatically created database of embeddings. This is the "memory" the app searches.
- `src/ingest.py` – script that builds the vector store.
- `src/app.py` – Streamlit user interface.
- `feedback.db` – stores the thumbs-up/down feedback given in the app.

---

## 3. Teach the Oracle (ingest new data)

1. Place any new CSV files or images in the `data/` directory.  Ensure the CSV columns match the examples above.
2. Run the ingestion script from the project root:

   ```bash
   python src/ingest.py data/error_codes.csv data/steps.csv data/sample_qa.csv
   ```

   - The first path (`error_codes.csv`) is required.
   - `steps.csv` and `sample_qa.csv` are optional; omit them if you do not have those files.
3. The script reads the files, converts the text (and optional image captions) into numerical **embeddings**, and stores them in the `vectorstore/` directory.  Think of this as teaching the Oracle new facts.
4. Re-run the ingestion command whenever you update or add new data.  To rebuild from scratch, delete the `vectorstore/` folder first.

---

## 4. Launch the web app

After the vector store exists, start the Streamlit app:

```bash
streamlit run src/app.py
```

Your browser will open a page where you can type a question or error code. The Oracle will:

1. Search the vector store for the most relevant pieces of information.
2. Ask the language model to craft a helpful answer.
3. Display follow-up questions or related images when available.
4. Let you rate the response with a thumbs up or down. Feedback is saved in `feedback.db` for future improvement.

---

## 5. Updating data later

- Edit or add new CSV files in `data/` and run the ingestion script again.
- Add new images to `data/images/` and list them in `data/image_captions.csv` to give each image a friendly description.
- The web app always uses the latest contents of `vectorstore/`; you only need to restart the app after re-ingesting.

---

## 6. Troubleshooting tips

- **Missing Hugging Face token:** The app will fail to start if the token is not set in `src/.streamlit/secrets.toml`.
- **Stale answers:** Delete the `vectorstore/` folder and ingest again to rebuild the knowledge base.
- **No browser page:** Streamlit prints a local URL in the terminal. Copy the link into your browser if it does not open automatically.

---

You are now ready to use the QuikPick Oracle. Ingest your latest troubleshooting data, launch the app, and start asking questions!
