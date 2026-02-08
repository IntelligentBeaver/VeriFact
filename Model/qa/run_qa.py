"""Interactive CLI for the QA system with top-level config."""

from pathlib import Path

from qa_system import QAConfig, QASystem, save_answer


# =========================
# CONFIGURATION (edit here)
# =========================
INDEX_DIR = Path("..") / "retrieval" / "storage"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
TOP_K = 6
MIN_SCORE = 0.4
MAX_CONTEXT_CHARS = 12000
SAVE_DIR = Path("qa_outputs")


def main() -> None:
    cfg = QAConfig(
        index_dir=INDEX_DIR.resolve(),
        top_k=TOP_K,
        min_score=MIN_SCORE,
        ollama_url=OLLAMA_URL,
        ollama_model=OLLAMA_MODEL,
        max_context_chars=MAX_CONTEXT_CHARS,
    )
    qa = QASystem(cfg)

    print("QA system ready. Type a question or 'exit'.")
    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        result = qa.answer(question)
        print("\nAnswer:\n")
        print(result["answer"])

        save = input("\nSave JSON output? (y/n): ").strip().lower()
        if save == "y":
            out_path = SAVE_DIR / "last_answer.json"
            save_answer(out_path, result)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
