from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import VerifierModelConfig, load_verifier_model_config


def _softmax(values: Iterable[float]) -> List[float]:
    exp_values = []
    max_val = None
    for val in values:
        if max_val is None or val > max_val:
            max_val = val
    max_val = max_val if max_val is not None else 0.0
    total = 0.0
    for val in values:
        exp_val = pow(2.718281828459045, val - max_val)
        exp_values.append(exp_val)
        total += exp_val
    if total == 0.0:
        return [0.0 for _ in exp_values]
    return [val / total for val in exp_values]


def _parse_labels(raw_labels: Optional[str]) -> List[str]:
    if not raw_labels:
        return ["neutral", "refutes", "supports"]
    return [label.strip() for label in raw_labels.split(",") if label.strip()]


def _format_input(template: str, claim: str, evidence: str) -> str:
    return template.format(claim=claim, evidence=evidence)


def _load_pickle(path: str) -> Any:
    joblib_error = None
    try:
        import joblib

        return joblib.load(path)
    except Exception as exc:
        joblib_error = exc

    torch_error = None
    try:
        import torch

        map_location = None
        if not torch.cuda.is_available():
            map_location = torch.device("cpu")
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        torch_error = exc

    try:
        original_torch_load = None
        try:
            import torch

            original_torch_load = torch.load

            def _torch_load_cpu(*args, **kwargs):
                if original_torch_load is None:
                    raise RuntimeError("torch.load is not available")
                if not torch.cuda.is_available() and "map_location" not in kwargs:
                    kwargs["map_location"] = torch.device("cpu")
                return original_torch_load(*args, **kwargs)

            torch.load = _torch_load_cpu
        except Exception:
            original_torch_load = None

        try:
            with open(path, "rb") as handle:
                try:
                    return pickle.load(handle)
                except ModuleNotFoundError as mod_exc:
                    # Provide actionable guidance for missing dependency during unpickle
                    missing = getattr(mod_exc, "name", None) or str(mod_exc)
                    pkg = missing.split(".")[0]
                    raise RuntimeError(
                        f"Missing Python dependency during unpickle: '{missing}'.\n"
                        f"Install it in your environment and restart the app, e.g.:\n"
                        f"  pip install {pkg}\n"
                        f"If the model requires GPU-specific libs, install matching versions."
                    ) from mod_exc
        finally:
            if original_torch_load is not None:
                torch.load = original_torch_load
    except Exception as exc:
        details = []
        if joblib_error:
            details.append(f"joblib error: {joblib_error}")
        if torch_error:
            details.append(f"torch error: {torch_error}")
        error_detail = "; ".join(details) if details else "joblib unavailable"
        raise RuntimeError(f"Failed to load pickle at {path}. {error_detail}") from exc


@dataclass(frozen=True)
class VerifierResult:
    label: str
    confidence: Optional[float]
    scores: Dict[str, float]
    raw_output: Optional[Any] = None


class PickleVerifier:
    def __init__(
        self,
        model: Any,
        labels: List[str],
        input_template: str,
        tokenizer: Optional[Any] = None,
        tokenizer_path: Optional[str] = None,
        base_model_name: Optional[str] = None,
        tokenizer_max_length: int = 512,
        tokenizer_padding: str = "max_length",
        tokenizer_local_files_only: bool = True,
    ) -> None:
        self.model = model
        self.labels = labels
        self.input_template = input_template
        self.tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path
        self.base_model_name = base_model_name
        self.tokenizer_max_length = tokenizer_max_length
        self.tokenizer_padding = tokenizer_padding
        self.tokenizer_local_files_only = tokenizer_local_files_only

    def verify(self, claim: str, evidence: str) -> VerifierResult:
        if hasattr(self.model, "verify_claim"):
            return self._from_verify_claim(claim, evidence)

        text_input = _format_input(self.input_template, claim, evidence)

        if hasattr(self.model, "predict_proba"):
            return self._from_predict_proba(text_input)

        if hasattr(self.model, "decision_function"):
            return self._from_decision_function(text_input)

        if hasattr(self.model, "predict"):
            return self._from_predict(text_input)

        if callable(self.model):
            return self._from_callable(text_input)

        raise RuntimeError("Unsupported verifier model interface. Provide a model with verify_claim, predict_proba, or predict.")

    def _from_verify_claim(self, claim: str, evidence: str) -> VerifierResult:
        output = self.model.verify_claim(claim, evidence)
        label = str(output.get("prediction") or output.get("label") or "unknown")
        confidence = output.get("confidence")
        confidence_scores = output.get("confidence_scores") or {}
        scores = {
            key.lower(): float(value)
            for key, value in confidence_scores.items()
            if value is not None
        }
        return VerifierResult(label=label, confidence=confidence, scores=scores, raw_output=output)

    def _from_predict_proba(self, text_input: str) -> VerifierResult:
        probs = self.model.predict_proba([text_input])[0]
        labels = self._resolve_labels(len(probs))
        scores = {labels[idx].lower(): float(prob) for idx, prob in enumerate(probs)}
        best_idx = max(range(len(probs)), key=lambda idx: probs[idx])
        return VerifierResult(label=labels[best_idx], confidence=float(probs[best_idx]), scores=scores)

    def _from_decision_function(self, text_input: str) -> VerifierResult:
        scores_raw = self.model.decision_function([text_input])[0]
        probs = _softmax(scores_raw)
        labels = self._resolve_labels(len(probs))
        scores = {labels[idx].lower(): float(prob) for idx, prob in enumerate(probs)}
        best_idx = max(range(len(probs)), key=lambda idx: probs[idx])
        return VerifierResult(label=labels[best_idx], confidence=float(probs[best_idx]), scores=scores)

    def _from_predict(self, text_input: str) -> VerifierResult:
        pred = self.model.predict([text_input])[0]
        label = str(pred)
        return VerifierResult(label=label, confidence=None, scores={})

    def _resolve_local_path(self, raw_path: Optional[str]) -> Optional[Path]:
        if not raw_path:
            return None

        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate

        project_root = Path(__file__).resolve().parent.parent
        return (project_root / candidate).resolve()

    def _get_tokenizer(self):
        if self.tokenizer is not None:
            return self.tokenizer

        model_tokenizer = getattr(self.model, "tokenizer", None)
        if model_tokenizer is not None:
            self.tokenizer = model_tokenizer
            return self.tokenizer

        from transformers import AutoTokenizer

        attempts: List[str] = []

        local_candidates: List[Optional[str]] = [
            self.tokenizer_path,
        ]
        for candidate in local_candidates:
            resolved = self._resolve_local_path(candidate)
            if resolved is not None:
                attempts.append(str(resolved))
                if resolved.exists():
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        str(resolved),
                        local_files_only=self.tokenizer_local_files_only,
                    )
                    return self.tokenizer

        model_candidates: List[str] = []
        if self.base_model_name:
            model_candidates.append(self.base_model_name)

        config_name = getattr(getattr(self.model, "config", None), "_name_or_path", None)
        if config_name:
            model_candidates.append(str(config_name))

        deduped_model_candidates: List[str] = []
        for item in model_candidates:
            if item and item not in deduped_model_candidates:
                deduped_model_candidates.append(item)

        for model_name in deduped_model_candidates:
            try:
                attempts.append(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                return self.tokenizer
            except Exception:
                continue

        raise RuntimeError(
            "Transformer model requires a tokenizer but none could be loaded. "
            "Provide VERIFIER_TOKENIZER_PATH pointing to a local tokenizer folder, "
            "or set VERIFIER_BASE_MODEL. "
            f"Tried: {attempts}"
        )

    def _from_callable(self, text_input: str) -> VerifierResult:
        # If the callable is a PyTorch / Transformers model it will expect tensors
        try:
            import torch
        except Exception:
            raise RuntimeError("PyTorch is required for callable verifier models but is not installed.")

        # If model is a torch.nn.Module, prepare tokenized inputs
        is_torch_module = False
        try:
            if isinstance(self.model, torch.nn.Module):
                is_torch_module = True
        except Exception:
            is_torch_module = False

        if is_torch_module:
            # Attempt to find or load a tokenizer
            tokenizer = self._get_tokenizer()

            # Tokenize the combined input
            try:
                inputs = tokenizer(
                    text_input,
                    add_special_tokens=True,
                    truncation=True,
                    padding=self.tokenizer_padding,
                    max_length=self.tokenizer_max_length,
                    return_tensors="pt",
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to tokenize verifier input: {exc}") from exc

            # Move tensors to model device
            try:
                device = next(self.model.parameters()).device
            except Exception:
                device = torch.device("cpu")

            for k, v in inputs.items():
                if hasattr(v, "to"):
                    inputs[k] = v.to(device)

            # Call model and extract logits
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = None
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                logits = outputs[0]

            if logits is None:
                raise RuntimeError("Model did not return logits when called with tokenized inputs.")

            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            labels = self._resolve_labels(len(probs))
            scores = {labels[idx].lower(): float(probs[idx]) for idx in range(len(probs))}
            best_idx = int(probs.argmax())
            return VerifierResult(label=labels[best_idx], confidence=float(probs[best_idx]), scores=scores, raw_output=None)

        # Not a torch module: expect the callable to return label/score list or similar
        output = self.model(text_input)
        if isinstance(output, list) and output and isinstance(output[0], dict):
            labels = []
            scores = []
            for item in output:
                label = item.get("label")
                score = item.get("score")
                if label is not None and score is not None:
                    labels.append(str(label))
                    scores.append(float(score))
            if labels and scores:
                score_map = {labels[idx].lower(): scores[idx] for idx in range(len(labels))}
                best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
                return VerifierResult(
                    label=labels[best_idx],
                    confidence=scores[best_idx],
                    scores=score_map,
                    raw_output=output,
                )

        raise RuntimeError("Callable verifier output format not supported.")

    def _resolve_labels(self, num_labels: int) -> List[str]:
        if hasattr(self.model, "classes_"):
            classes = list(self.model.classes_)
            if len(classes) == num_labels:
                return [str(label) for label in classes]
        if len(self.labels) == num_labels:
            return self.labels
        return [f"class_{idx}" for idx in range(num_labels)]


def load_verifier_from_pickle(path: str, config: Optional[VerifierModelConfig] = None) -> PickleVerifier:
    verifier_config = config or load_verifier_model_config()
    labels = _parse_labels(verifier_config.labels_csv)
    input_template = verifier_config.input_template
    loaded = _load_pickle(path)

    model = loaded
    tokenizer = None
    tokenizer_path = verifier_config.tokenizer_path
    base_model_name = verifier_config.base_model

    if isinstance(loaded, dict):
        model = (
            loaded.get("model")
            or loaded.get("verifier_model")
            or loaded.get("classifier")
            or loaded
        )
        tokenizer = loaded.get("tokenizer")
        tokenizer_path = tokenizer_path or loaded.get("tokenizer_path") or loaded.get("tokenizer_dir")
        base_model_name = base_model_name or loaded.get("base_model_name") or loaded.get("model_name")
    elif isinstance(loaded, (list, tuple)) and len(loaded) >= 1:
        model = loaded[0]
        if len(loaded) > 1:
            tokenizer = loaded[1]

    return PickleVerifier(
        model=model,
        labels=labels,
        input_template=input_template,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        base_model_name=base_model_name,
        tokenizer_max_length=verifier_config.tokenizer_max_length,
        tokenizer_padding=verifier_config.tokenizer_padding,
        tokenizer_local_files_only=verifier_config.tokenizer_local_files_only,
    )
