from pathlib import Path
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.utils.config import RAW_DATA_DIR

app = FastAPI(title="AI Agent Copilot API")


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

        # Lazy import to avoid heavy startup work during app boot
        from app.retrieval.vectordb import index_pdf

        indexing_result = index_pdf(str(file_path))

        return {
            "message": "File uploaded and indexed successfully",
            "file_name": file.filename,
            "indexing_result": indexing_result,
        }

    except Exception as e:
        import traceback
        print("UPLOAD ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query_documents(request: QueryRequest):
    try:
        # Lazy import to avoid heavy startup work during app boot
        from app.retrieval.rag_chain import generate_answer

        result = generate_answer(request.question, k=request.top_k)
        return result

    except Exception as e:
        import traceback
        print("QUERY ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))