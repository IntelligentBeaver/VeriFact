"""Interactive CLI for the QA system with top-level config."""

from pathlib import Path

from config import load_qa_config
from qa_system import QAConfig, QASystem, save_answer


# =========================
# CONFIGURATION (edit here)
# =========================
INDEX_DIR = Path("..") / "storage"
defaults = load_qa_config()
OLLAMA_URL = defaults.ollama_url
OLLAMA_MODEL = defaults.ollama_model
TOP_K = defaults.top_k
MIN_SCORE = defaults.min_score
MAX_CONTEXT_CHARS = defaults.max_context_chars
SAVE_DIR = Path("qa_outputs")


def main() -> None:
    cfg = QAConfig.from_index_dir(INDEX_DIR)
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
