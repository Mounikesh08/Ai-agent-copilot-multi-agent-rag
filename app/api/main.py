from fastapi import FastAPI

app = FastAPI(title="AI Agent Copilot API")


@app.get("/")
def root():
    return {"message": "AI Agent Copilot API is running"}