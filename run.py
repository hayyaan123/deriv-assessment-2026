"""
Grounded RAG assistant pipeline.

Stages:
INIT -> DOCUMENTS_LOADED -> CHUNKS_CREATED -> INDEX_BUILT -> QUESTIONS_LOADED
-> RETRIEVAL_COMPLETE -> ANSWERS_GENERATED -> CITATIONS_VALIDATED
-> RESULTS_EXPORTED -> VALIDATION_COMPLETE

Every stage writes its intermediate output to artifacts/ so each step can be inspected.
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Configuration (defined in code, as required) ----
DOCS_DIR = Path("docs")
QUESTIONS_PATH = Path("questions.json")
ARTIFACTS_DIR = Path("artifacts")

CHUNK_SIZE = 500      # max characters per chunk
CHUNK_OVERLAP = 100   # characters shared between consecutive chunks
TOP_K = 3             # chunks retrieved per question

LLM_PROVIDER = "google-gemini"
LLM_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
UNSUPPORTED_ANSWER = "The knowledge base does not provide enough information to answer this."
# Used only by the offline fallback (no API key): below this retrieval score we refuse to answer.
FALLBACK_MIN_SCORE = 0.15


def log_stage(stage: str) -> None:
    print(f"[STAGE] {stage}")


def save_json(name: str, data) -> Path:
    path = ARTIFACTS_DIR / name
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---- Stage 1: Document ingestion ----
def load_documents(docs_dir: Path) -> list[dict]:
    """Load every .md / .txt file in docs_dir. No filenames are hardcoded."""
    files = sorted(p for p in docs_dir.rglob("*") if p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise FileNotFoundError(f"No .md or .txt files found in {docs_dir}/")
    documents = []
    for p in files:
        documents.append({
            "doc_id": p.relative_to(docs_dir).with_suffix("").as_posix(),
            "path": p.as_posix(),
            "text": p.read_text(encoding="utf-8"),
        })
    return documents


# ---- Stage 2: Chunking ----
def chunk_document(doc: dict, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Fixed-size character windows with overlap. Window ends are nudged back to the
    nearest paragraph/sentence/word boundary so chunks don't cut words in half.
    Chunk IDs are deterministic: <doc_id>::chunk_<n>.
    """
    text = doc["text"]
    chunks, start, n = [], 0, 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for sep in ("\n\n", ". ", "\n", " "):
                cut = window.rfind(sep)
                if cut > size // 2:          # only accept a boundary in the back half
                    end = start + cut + len(sep)
                    break
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "chunk_id": f"{doc['doc_id']}::chunk_{n}",
                "doc_id": doc["doc_id"],
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
            })
            n += 1
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)  # always make forward progress
        # snap the overlap start forward to a word boundary so chunks don't begin mid-word
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return chunks


def create_chunks(documents: list[dict]) -> list[dict]:
    return [c for doc in documents for c in chunk_document(doc)]


# ---- Stage 3: Index + retrieval (TF-IDF, built at runtime from docs) ----
class TfidfIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = scores.argsort()[::-1][:k]
        return [{
            "chunk_id": self.chunks[i]["chunk_id"],
            "doc_id": self.chunks[i]["doc_id"],
            "score": round(float(scores[i]), 4),
            "text": self.chunks[i]["text"],
        } for i in ranked]


def load_questions(path: Path) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    for q in questions:
        if "id" not in q or "question" not in q:
            raise ValueError(f"Malformed question entry: {q}")
    return questions


def retrieve_all(index: TfidfIndex, questions: list[dict]) -> list[dict]:
    return [{
        "id": q["id"],
        "question": q["question"],
        "retrieved_chunks": index.search(q["question"]),
    } for q in questions]


# ---- Stage 4: Answer generation (uses ONLY the retrieved chunks) ----
PROMPT_TEMPLATE = """You are a support assistant for a developer platform.
Answer the user's question using ONLY the context chunks below.

Rules:
1. Use only facts stated in the context. Do not use outside knowledge.
2. Do not invent product features, limits, prices, or policies.
3. If the context does not contain the answer, set "supported" to false, set "answer" to
   exactly "{unsupported}", and return an empty "citations" list.
4. If the answer is supported, cite every chunk you used by its chunk_id
   (copy the IDs exactly as given). Only cite chunk_ids that appear below.
5. Keep the answer short (1-3 sentences).

Return ONLY a JSON object, no markdown, in this exact shape:
{{"answer": "string", "supported": true, "citations": ["chunk_id", "..."]}}

Context:
{context}

Question: {question}
"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[chunk_id: {c['chunk_id']}]\n{c['text']}" for c in chunks)
    return PROMPT_TEMPLATE.format(unsupported=UNSUPPORTED_ANSWER, context=context, question=question)


def parse_llm_json(raw: str) -> dict:
    """Extract the JSON object from the model output (tolerates ```json fences / extra text)."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model output")
    data = json.loads(match.group(0))
    citations = data.get("citations") or []
    if not isinstance(citations, list):
        citations = [citations]
    return {
        "answer": str(data.get("answer", "")).strip(),
        "supported": bool(data.get("supported", False)),
        "citations": [str(c) for c in citations],
    }


def log_llm_call(record: dict) -> None:
    with (ARTIFACTS_DIR / "llm_calls.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def generate_with_gemini(client, item: dict) -> dict:
    prompt = build_prompt(item["question"], item["retrieved_chunks"])
    log = {
        "stage": "answer_generation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "question_id": item["id"],
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "input_artifacts": ["artifacts/retrieval_results.json"],
        "output_artifact": "artifacts/answers.json",
    }
    try:
        interaction = client.interactions.create(model=LLM_MODEL, input=prompt)
        parsed = parse_llm_json(interaction.output_text)
        log["status"] = "ok"
    except Exception as e:  # API error or unparseable output -> refuse rather than guess
        print(f"   ! question {item['id']}: LLM call failed ({type(e).__name__}: {e}); marking unsupported")
        parsed = {"answer": UNSUPPORTED_ANSWER, "supported": False, "citations": []}
        log["status"] = f"error: {type(e).__name__}"
    log_llm_call(log)
    return parsed


STOPWORDS = {"the", "a", "an", "is", "are", "do", "does", "can", "i", "what", "how", "if",
             "of", "to", "for", "on", "in", "there", "my", "it", "be", "and", "or", "long"}


def generate_extractive(item: dict) -> dict:
    """
    Offline fallback when no API key is set (documented in README):
    refuse if the best chunk scores below FALLBACK_MIN_SCORE, otherwise return the
    sentence from the top chunk with the most question-term overlap, citing that chunk.
    """
    chunks = item["retrieved_chunks"]
    if not chunks or chunks[0]["score"] < FALLBACK_MIN_SCORE:
        return {"answer": UNSUPPORTED_ANSWER, "supported": False, "citations": []}
    top = chunks[0]
    q_terms = {w for w in re.findall(r"[a-z0-9]+", item["question"].lower()) if w not in STOPWORDS}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", top["text"]) if len(s.strip()) > 20]
    best = max(sentences, key=lambda s: len(q_terms & set(re.findall(r"[a-z0-9]+", s.lower()))), default=top["text"])
    return {"answer": " ".join(best.split()), "supported": True, "citations": [top["chunk_id"]]}


def generate_answers(retrieval: list[dict]) -> tuple[list[dict], str]:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = None
    if api_key:
        from google import genai
        client = genai.Client(api_key=api_key)
        mode = f"llm ({LLM_MODEL})"
    else:
        print("   ! GEMINI_API_KEY not set -> using offline extractive fallback")
        mode = "extractive-fallback"

    answers = []
    for item in retrieval:
        result = generate_with_gemini(client, item) if client else generate_extractive(item)
        answers.append({"id": item["id"], "question": item["question"], **result})
    return answers, mode


def main() -> None:
    load_dotenv()
    log_stage("INIT")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "llm_calls.jsonl").unlink(missing_ok=True)  # fresh log per run

    documents = load_documents(DOCS_DIR)
    save_json("documents.json", documents)
    log_stage(f"DOCUMENTS_LOADED ({len(documents)} docs)")

    chunks = create_chunks(documents)
    save_json("chunks.json", chunks)
    log_stage(f"CHUNKS_CREATED ({len(chunks)} chunks)")

    index = TfidfIndex(chunks)
    log_stage("INDEX_BUILT")

    questions = load_questions(QUESTIONS_PATH)
    log_stage(f"QUESTIONS_LOADED ({len(questions)} questions)")

    retrieval = retrieve_all(index, questions)
    save_json("retrieval_results.json", retrieval)
    log_stage("RETRIEVAL_COMPLETE")

    answers, mode = generate_answers(retrieval)
    save_json("answers.json", answers)
    log_stage(f"ANSWERS_GENERATED (mode: {mode})")

    # Next: CITATIONS_VALIDATED -> RESULTS_EXPORTED -> VALIDATION_COMPLETE


if __name__ == "__main__":
    main()
