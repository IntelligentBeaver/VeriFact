"""
QA system built on top of the existing retriever.
Uses Ollama for Llama 3.1 8B by default.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import os
import sys

import requests


BASE_DIR = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = BASE_DIR / "retrieval"

if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))

from simple_retriever import MinimalModelManager, SimpleRetriever  # noqa: E402


@dataclass
class QAConfig:
    index_dir: Path
    top_k: int = 6
    min_score: float = 0.4
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    max_context_chars: int = 12000


class OllamaClient:
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = requests.post(self.url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Ollama is not reachable at http://localhost:11434. "
                "Start Ollama and ensure the model is pulled, or set --model/OLLAMA_URL."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                "Ollama request timed out. Try a smaller context or check Ollama server load."
            ) from exc


class QASystem:
    def __init__(self, config: QAConfig):
        self.config = config
        self.model_manager = MinimalModelManager(str(config.index_dir))
        self.retriever = SimpleRetriever(self.model_manager, str(config.index_dir))
        self.llm = OllamaClient(config.ollama_url, config.ollama_model)

    def answer(self, question: str) -> Dict[str, Any]:
        results = self.retriever.search(question)
        filtered = [r for r in results if r.get("final_score", 0) >= self.config.min_score]
        top_results = filtered[: self.config.top_k]

        if not top_results:
            return {
                "question": question,
                "answer": "I do not have enough evidence to answer from the index.",
                "sources": [],
                "raw_results": results,
            }

        context, sources = self._build_context(top_results)
        prompt = self._build_prompt(question, context)
        try:
            answer = self.llm.generate(prompt)
        except RuntimeError as exc:
            return {
                "question": question,
                "answer": f"LLM error: {exc}",
                "sources": sources,
                "raw_results": results,
            }

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "raw_results": results,
        }

    def _build_context(self, results: List[Dict[str, Any]]) -> (str, List[Dict[str, Any]]):
        context_blocks = []
        sources = []
        total_chars = 0

        for i, result in enumerate(results, 1):
            passage = result.get("passage", {})
            title = passage.get("title", "")
            url = passage.get("url", "")
            text = passage.get("text", "")
            score = result.get("final_score", 0)

            block = (
                f"[Source {i}]\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Score: {score:.3f}\n"
                f"Text: {text}\n"
            )

            if total_chars + len(block) > self.config.max_context_chars:
                break

            context_blocks.append(block)
            total_chars += len(block)
            sources.append(
                {
                    "id": i,
                    "title": title,
                    "url": url,
                    "score": score,
                }
            )

        return "\n".join(context_blocks), sources

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            "You are a careful medical QA assistant. Answer using only the provided sources. "
            "If the sources do not support an answer, say you do not have enough evidence. "
            "Cite sources like [Source 1], [Source 2].\n\n"
            f"Question: {question}\n\n"
            f"Sources:\n{context}\n\n"
            "Answer:"
        )


def default_config() -> QAConfig:
    index_dir = BASE_DIR / "retrieval" / "storage"
    return QAConfig(index_dir=index_dir)


def save_answer(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QA over existing retrieval index.")
    parser.add_argument("question", nargs="+", help="Question to answer")
    parser.add_argument("--model", default=None, help="Ollama model name")
    parser.add_argument("--index", default=None, help="Index directory path")
    parser.add_argument("--out", default=None, help="Save JSON output to file")

    args = parser.parse_args()

    cfg = default_config()
    if args.model:
        cfg.ollama_model = args.model
    if args.index:
        cfg.index_dir = Path(args.index).resolve()

    qa = QASystem(cfg)
    question_text = " ".join(args.question)
    result = qa.answer(question_text)

    print("\nAnswer:\n")
    print(result["answer"])

    if args.out:
        save_answer(Path(args.out), result)
        print(f"\nSaved: {args.out}")
