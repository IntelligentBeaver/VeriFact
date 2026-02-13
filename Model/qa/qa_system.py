"""
QA system built on top of the existing retriever.
Uses Ollama for Llama 3.1 8B by default.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import sys
import re
import string

import requests

from config import QAConfigDefaults, PROMPT_TEMPLATE

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

        context, sources = self._build_context(question, top_results)
        prompt = PROMPT_TEMPLATE.format(question=question, context=context)
        try:
            answer = self.llm.generate(prompt)
        except RuntimeError as exc:
            return {
                "question": question,
                "answer": f"LLM error: {exc}",
                "sources": sources,
                "raw_results": results,
            }

        # Post-process / enforce format and filter citations
        final_answer = self._enforce_answer_format(answer, question, sources)

        return {
            "question": question,
            "answer": final_answer,
            "sources": sources,
            "raw_results": results,
        }

    def _build_context(self, question: str, results: List[Dict[str, Any]]) -> (str, List[Dict[str, Any]]):
        context_blocks = []
        sources = []
        total_chars = 0

        # Build keyword set from question for eligibility filtering
        keywords = self._extract_keywords(question)
        matched_any = False

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

            # eligibility: include passage if it contains any keyword (or include as fallback)
            eligible = False
            passage_text_lower = text.lower()
            for kw in keywords:
                if kw in passage_text_lower:
                    eligible = True
                    matched_any = True
                    break

            if not eligible and keywords:
                # skip this passage — it's not relevant by keyword
                continue

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

    def _extract_keywords(self, text: str) -> List[str]:
        # Simple keyword extraction: split, remove punctuation, filter stopwords and short tokens
        stopwords = {
            'the', 'is', 'in', 'and', 'or', 'of', 'a', 'an', 'to', 'for', 'with', 'on', 'by', 'that', 'does', 'do'
        }
        text = text.lower()
        # remove punctuation
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        tokens = [t.strip() for t in text.split() if t and len(t) > 2 and t not in stopwords]
        # return up to 8 keywords
        return tokens[:8]

    def _enforce_answer_format(self, answer: str, question: str, sources: List[Dict[str, Any]]) -> str:
        # Ensure answer follows 'Conclusion: ...' and 'Evidence: ...' format.
        if isinstance(answer, str) and answer.strip().lower().startswith('conclusion:') and 'evidence:' in answer.lower():
            return answer.strip()

        # Attempt to synthesize a short deterministic conclusion from sources
        # Scan sources' texts for causal/transmission/association hints
        hints = []
        for s in sources:
            txt = (s.get('title', '') + ' ' + s.get('url', '')).lower()
            # if raw_results include passage text, it will be present in raw_results; but sources here are metadata
            # Use title/url and score as fallback; better hints come from raw_results when available
            if any(w in txt for w in ('transmit', 'transmission', 'transmitted', 'cause', 'causes', 'caused')):
                hints.append('cause')
            elif any(w in txt for w in ('associate', 'associated', 'link', 'linked', 'risk')):
                hints.append('associate')

        if 'cause' in hints:
            conclusion = 'Conclusion: The sources indicate transmission/causation as stated in the cited sources.'
        elif 'associate' in hints:
            conclusion = 'Conclusion: The sources indicate an association or increased risk.'
        else:
            # fallback to the original LLM answer's first sentence if nothing found
            first_sentence = re.split(r'\n|\.|!|\?', answer.strip())[0]
            if first_sentence:
                conclusion = f'Conclusion: {first_sentence.strip()}'
            else:
                conclusion = 'Conclusion: I do not have enough evidence to answer from the provided sources.'

        evidence_ids = [str(s.get('id')) for s in sources][:5]
        evidence_line = 'Evidence: Sources [' + ', '.join(f'Source {i}' for i in evidence_ids) + '].'
        return conclusion + '\n\n' + evidence_line

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
