from fastapi import FastAPI, HTTPException, UploadFile, File
from ollama import Client, ResponseError
from pydantic import BaseModel, Field
import httpx


app = FastAPI(
    title="DocPilot",
    description="AI-powered Document Intelligence Platform",
    version="0.1.0"
)


# Intitlize ollama client
client = Client(
    host="http://localhost:11434",
    timeout=120.0
)
MODEL_NAME = "qwen3:1.7b"

# DEfinr the request structure


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=4000
    )

# Define the response structure


class ChatResponse(BaseModel):
    answer: str
    model: str

# Root endpoint


@app.get("/")
def home():
    return {
        "message": "Welcome to DocPilot",
        "status": "running"
    }

# Health check endpoint


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": request.message
                }
            ],
            think=False,
            options={
                "num_ctx": 2048,
                "num_predict": 500
            }
        )

        return ChatResponse(
            answer=response.message.content or "",
            model=MODEL_NAME
        )

    except (ResponseError, httpx.HTTPError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Unable to get a response from Ollama."
        ) from exc


MAX_FILE_SIZE = 10*1024*1024


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    # 1. Validate the file extension and content type

    if (
        not file.filename
        or not file.filename.endswith(".pdf")
        or file.content_type != "application/pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail="only PDF files are allowed."
        )

    # 2 read upload file
    contents = await file.read(MAX_FILE_SIZE+1)

    # 3 Validate file size

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds 10mb limit"
        )

    # 4 check if the pdf fails the signature
    if not contents.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file signature"
        )

    return {
        "filename": file.filename,
        "file_size_bytes": len(contents),
        "status": "uploaded",
        "message": "PDF recived succesfully"
    }
