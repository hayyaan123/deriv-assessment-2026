"""
Standalone artifact checker. Run after `python run.py`:

    python validate.py

Exits 0 if every check passes, 1 otherwise. Needs only the standard library.
"""
import json
import sys
from pathlib import Path

ARTIFACTS = Path("artifacts")
QUESTIONS = Path("questions.json")
MAX_RETRIEVED = 3
REQUIRED = [
    QUESTIONS,
    ARTIFACTS / "documents.json",
    ARTIFACTS / "chunks.json",
    ARTIFACTS / "retrieval_results.json",
    ARTIFACTS / "answers.json",
    ARTIFACTS / "citation_validation.json",
    ARTIFACTS / "final_answers.json",
]


def run_checks() -> list[tuple[str, bool, list[str]]]:
    """Returns (check name, passed, problems) for every check."""
    results = []

    def check(name, problems):
        results.append((name, not problems, problems))

    # 1. required files exist
    missing = [str(p) for p in REQUIRED if not p.exists()]
    check("required artifact files exist", [f"missing: {m}" for m in missing])
    if missing:
        return results

    # 2. JSON is valid
    data, bad = {}, []
    for p in REQUIRED:
        try:
            data[p.name] = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            bad.append(f"{p}: {e}")
    llm_log = ARTIFACTS / "llm_calls.jsonl"
    if llm_log.exists():
        for n, line in enumerate(llm_log.read_text(encoding="utf-8").splitlines(), 1):
            try:
                json.loads(line)
            except json.JSONDecodeError:
                bad.append(f"{llm_log} line {n} is not valid JSON")
    check("JSON is valid", bad)
    if bad:
        return results

    questions = data["questions.json"]
    chunks = data["chunks.json"]
    retrieval = {r["id"]: r for r in data["retrieval_results.json"]}
    answers = data["answers.json"]
    question_ids = [q["id"] for q in questions]
    answer_ids = [a["id"] for a in answers]
    chunk_ids = {c["chunk_id"] for c in chunks}

    # 3. every question has exactly one answer
    problems = [f"question {qid} has {answer_ids.count(qid)} answers" for qid in question_ids
                if answer_ids.count(qid) != 1]
    problems += [f"answer {aid} has no matching question" for aid in set(answer_ids) - set(question_ids)]
    check("every question has exactly one answer", problems)

    # 4. at most 3 retrieved chunks per question
    problems = [f"question {qid}: no retrieval result" for qid in question_ids if qid not in retrieval]
    problems += [f"question {qid}: {len(r['retrieved_chunks'])} chunks retrieved"
                 for qid, r in retrieval.items() if len(r["retrieved_chunks"]) > MAX_RETRIEVED]
    check(f"each question has at most {MAX_RETRIEVED} retrieved chunks", problems)

    # 5. cited chunk IDs are valid (exist, and were retrieved for that question)
    problems = []
    for a in answers:
        retrieved_ids = {c["chunk_id"] for c in retrieval.get(a["id"], {}).get("retrieved_chunks", [])}
        for cid in a.get("citations", []):
            if cid not in chunk_ids:
                problems.append(f"answer {a['id']}: '{cid}' does not exist in chunks.json")
            elif cid not in retrieved_ids:
                problems.append(f"answer {a['id']}: '{cid}' was not retrieved for this question")
    check("cited chunk IDs are valid", problems)

    # 6. unsupported answers have no citations
    check("unsupported answers have no citations",
          [f"answer {a['id']}" for a in answers if not a.get("supported") and a.get("citations")])

    # 7. supported answers have citations
    check("supported answers have citations",
          [f"answer {a['id']}" for a in answers if a.get("supported") and not a.get("citations")])

    return results


def main() -> int:
    results = run_checks()
    for name, passed, problems in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        for p in problems:
            print(f"       - {p}")
    failed = sum(not passed for _, passed, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
