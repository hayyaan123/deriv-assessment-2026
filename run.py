"""
Grounded RAG assistant pipeline.

Stages:
INIT -> DOCUMENTS_LOADED -> CHUNKS_CREATED -> INDEX_BUILT -> QUESTIONS_LOADED
-> RETRIEVAL_COMPLETE -> ANSWERS_GENERATED -> CITATIONS_VALIDATED
-> RESULTS_EXPORTED -> VALIDATION_COMPLETE

Every stage writes its intermediate output to artifacts/ so each step can be inspected.
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Configuration (defined in code, as required) ----
DOCS_DIR = Path("docs")
QUESTIONS_PATH = Path("questions.json")
ARTIFACTS_DIR = Path("artifacts")

CHUNK_SIZE = 500      # max characters per chunk
CHUNK_OVERLAP = 100   # characters shared between consecutive chunks
TOP_K = 3             # chunks retrieved per question


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


def main() -> None:
    log_stage("INIT")
    ARTIFACTS_DIR.mkdir(exist_ok=True)

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

    # Next: ANSWERS_GENERATED -> CITATIONS_VALIDATED -> RESULTS_EXPORTED -> VALIDATION_COMPLETE


if __name__ == "__main__":
    main()
