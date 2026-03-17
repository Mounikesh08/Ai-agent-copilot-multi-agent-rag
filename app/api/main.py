from pathlib import Path
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.retrieval.vectordb import index_pdf
from app.retrieval.rag_chain import generate_answer


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
    """
    Upload a PDF file and index it into ChromaDB.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now.")

    file_path = RAW_DATA_DIR / file.filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = index_pdf(str(file_path))

        return {
            "message": "File uploaded and indexed successfully",
            "file_name": file.filename,
            "indexing_result": result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query_documents(request: QueryRequest):
    """
    Query indexed documents and return a grounded answer.
    """
    try:
        result = generate_answer(request.question, k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))