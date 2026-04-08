from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

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
        return {
            "message": "Upload route is working",
            "file_name": file.filename
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