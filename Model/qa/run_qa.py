"""Interactive CLI for the QA system with top-level config."""

from pathlib import Path

from config import QAConfigDefaults
from qa_system import QAConfig, QASystem, save_answer


# =========================
# CONFIGURATION (edit here)
# =========================
INDEX_DIR = Path("..") / "storage"
OLLAMA_URL = QAConfigDefaults.ollama_url
OLLAMA_MODEL = QAConfigDefaults.ollama_model
TOP_K = QAConfigDefaults.top_k
MIN_SCORE = QAConfigDefaults.min_score
MAX_CONTEXT_CHARS = QAConfigDefaults.max_context_chars
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
