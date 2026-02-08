$ErrorActionPreference = "Stop"

Write-Host "Starting services..."
docker compose up -d --build

Write-Host "Pulling Ollama model..."
docker exec ollama ollama pull llama3.1:8b

Write-Host "Indexing passages..."
docker compose run --rm --workdir /app/retrieval retriever python run_retriever.py setup

Write-Host "Done."
