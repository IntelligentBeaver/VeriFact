"""Simple CLI for the QA system."""

from pathlib import Path

from qa_system import QASystem, default_config, save_answer


def main() -> None:
    cfg = default_config()
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
            out_path = Path("qa_outputs") / "last_answer.json"
            save_answer(out_path, result)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
