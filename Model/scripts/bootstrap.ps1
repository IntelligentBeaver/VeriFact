$ErrorActionPreference = "Stop"

Write-Host "Starting services..."
docker compose up -d --build

Write-Host "Indexing passages..."
docker compose run --rm --workdir /app/retrieval retriever python run_retriever.py setup

Write-Host "Done."
