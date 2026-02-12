import os
from pathlib import Path

from fastapi import FastAPI

from retrieval.simple_retriever import MinimalModelManager, SimpleRetriever
from routes.qa import router as qa_router
from routes.retrieval import router as retrieval_router

app = FastAPI(title="Verifact Retrieval API")


@app.on_event("startup")
def startup():
    index_dir = os.getenv("INDEX_DIR", "storage")
    index_path = Path(index_dir)

    model_manager = MinimalModelManager(str(index_path))
    retriever = SimpleRetriever(model_manager, str(index_path))

    app.state.retriever = retriever


@app.on_event("shutdown")
def shutdown():
    retriever = getattr(app.state, "retriever", None)
    if retriever is not None:
        retriever.clear_entity_cache()


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.get("/health")
def health():
    retriever = getattr(app.state, "retriever", None)
    return {
        "status": "ok" if retriever is not None else "starting",
        "retriever_loaded": retriever is not None,
        "elasticsearch_connected": bool(getattr(retriever, "es", None)) if retriever else False,
    }


app.include_router(retrieval_router)
app.include_router(qa_router)