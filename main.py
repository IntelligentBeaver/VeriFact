from pathlib import Path

from fastapi import FastAPI

from config import load_app_config
from retrieval.simple_retriever import MinimalModelManager, SimpleRetriever
from routes.qa import router as qa_router
from routes.retrieval import router as retrieval_router
from routes.verifier import router as verifier_router
from verifier.verifier_service import load_verifier_from_pickle

app_config = load_app_config()
app = FastAPI(title=app_config.app_title)


@app.on_event("startup")
def startup():
    index_path = Path(app_config.index_dir)

    model_manager = MinimalModelManager(str(index_path))
    retriever = SimpleRetriever(model_manager, str(index_path))

    app.state.retriever = retriever

    verifier_file = Path(app_config.verifier_model_path)
    if verifier_file.exists():
        app.state.verifier = load_verifier_from_pickle(str(verifier_file))
    else:
        app.state.verifier = None
        print(f"Warning: verifier model not found at {verifier_file}")


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
app.include_router(verifier_router)