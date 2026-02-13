# Verifact Backend

## Centralized Configuration

All runtime configuration is centralized in `config.py` and can be overridden using environment variables.

### App
- `APP_TITLE` (default: `Verifact Retrieval API`)
- `INDEX_DIR` (default: `storage`)
- `VERIFIER_MODEL_PATH` (default: `verifier/retriever_model.pkl`)

### Verifier Model
- `VERIFIER_LABELS` (default: `neutral,refutes,supports`)
- `VERIFIER_INPUT_TEMPLATE` (default: `{claim} [SEP] {evidence}`)
- `VERIFIER_TOKENIZER_PATH` (default: empty)
- `VERIFIER_BASE_MODEL` (default: `dmis-lab/biobert-v1.1`)
- `VERIFIER_TOKENIZER_MAX_LENGTH` (default: `512`)
- `VERIFIER_TOKENIZER_PADDING` (default: `max_length`)
- `VERIFIER_TOKENIZER_LOCAL_FILES_ONLY` (default: `true`)

### Verifier API
- `VERIFIER_MIN_TOP_K` (default: `1`)
- `VERIFIER_MAX_TOP_K` (default: `10`)
- `VERIFIER_DEFAULT_TOP_K` (default: `3`)
- `VERIFIER_DEFAULT_MIN_SCORE` (default: `0.0`)
- `VERIFIER_NO_EVIDENCE_VERDICT` (default: `NOT_ENOUGH_EVIDENCE`)

## Run

```bash
uvicorn main:app --reload
```
