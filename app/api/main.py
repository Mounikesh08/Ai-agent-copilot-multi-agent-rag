from pathlib import Path
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

app = FastAPI(title="AI Agent Copilot API")

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


@app.get("/")
def root():
    return {"message": "AI Agent Copilot API is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported right now.")

        file_path = RAW_DATA_DIR / file.filename

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "message": "Upload route is working",
            "file_name": file.filename,
            "indexing_result": {
                "file_path": str(file_path),
                "pages_loaded": 0,
                "chunks_created": 0,
                "documents_stored": 0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query_documents(request: QueryRequest):
    try:
        return {
            "question": request.question,
            "answer": "Backend query route is working. Full RAG will be added next.",
            "sources": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))