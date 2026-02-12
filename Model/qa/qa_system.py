"""
QA system built on top of the existing retriever.
Uses Ollama for Llama 3.1 8B by default.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import sys

import requests

from config import QAConfigDefaults

BASE_DIR = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = BASE_DIR / "retrieval"

if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))

from simple_retriever import MinimalModelManager, SimpleRetriever  # noqa: E402


@dataclass
class QAConfig:
    index_dir: Path
    top_k: int = QAConfigDefaults.top_k
    min_score: float = QAConfigDefaults.min_score
    ollama_url: str = QAConfigDefaults.ollama_url
    ollama_model: str = QAConfigDefaults.ollama_model
    max_context_chars: int = QAConfigDefaults.max_context_chars


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
        except requests.exceptions.HTTPError as exc:
            details = ""
            try:
                details = response.text.strip()
            except Exception:
                details = ""
            detail_msg = f" Details: {details}" if details else ""
            raise RuntimeError(
                "Ollama returned an HTTP error. Check the model tag and server logs."
                + detail_msg
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Ollama is not reachable at http://localhost:11434. "
                "Start Ollama (local or Docker) and ensure the model is pulled. "
                "Example: docker run -d --name ollama -p 11434:11434 ollama/ollama; "
                "docker exec ollama ollama pull llama3.1:8b. "
                "Or set --model / OLLAMA_URL."
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
            "If at least one source indicates an association, risk, or link, answer with that "
            "association even if causation is not proven. "
            "If sources explicitly state transmission or causation, you may say it causes or transmits. "
            "If sources show association or risk but not causation, say 'associated with increased risk' "
            "and avoid claiming it causes the outcome. "
            "Use this format: 'Conclusion: ...' then 'Evidence: ...'. "
            "Limit to 2 sentences total. "
            "Only cite sources that explicitly mention diabetes or prediabetes risk/link. "
            "Cite sources like [Source 1], [Source 2].\n\n"
            f"Question: {question}\n\n"
            f"Sources:\n{context}\n\n"
            "Answer:"
        )


def save_answer(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    print("Run qa/run_qa.py for interactive usage.")
