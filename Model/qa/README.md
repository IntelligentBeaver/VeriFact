# QA System

This directory provides a QA layer on top of the existing retriever.

## Requirements

- ElasticSearch running (per retrieval docs)
- Ollama running locally
- Llama 3.1 8B model pulled in Ollama

## Quick Start

From the repo root:

```bash
cd qa
python run_qa.py
```

Or run a single question and save output:

```bash
python qa_system.py "Are vaccines safe during pregnancy?" --out qa_outputs/answer.json
```

## Configuration

Defaults are in `qa_system.py`:

- Index: `retrieval/storage`
- Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Ollama URL: `http://localhost:11434/api/generate`

You can override:

```bash
python qa_system.py "Your question" --model llama3.1:8b --index /path/to/index
```
