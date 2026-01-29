import time
import uuid
import asyncio
import io
import requests
import numpy as np
import faiss
import json
import fitz 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import (
    FastAPI, UploadFile, File,
    BackgroundTasks, HTTPException,
    Request, Depends
)
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Generator

EMBED_URL = "https://abhinavdread-bge-en-ft-optimised.hf.space/embed"
GEN_URL   = "https://abhinavdread-qwen-1-5b-q4-k-m.hf.space/generate"
EMBED_DIM = 128
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


class Ingestor:
    def __init__(self, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk(self, text):
        return self.splitter.split_text(text)

    def embed(self, texts):
        r = requests.post(EMBED_URL, json={"chunks": texts}, timeout=120)
        r.raise_for_status()
        return np.array(r.json()["embeddings"], dtype="float32")

    def ingest_bytes(self, file_bytes: bytes, filename: str):
        doc = fitz.open(stream=file_bytes, filetype=filename.split(".")[-1])
        records = []
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"          
        doc.close()
        
        if not full_text:
            return None, {}
            
        chunks = self.chunk(full_text)
        for i, ch in enumerate(chunks):
             records.append({
                "text": ch,
                "meta": {"id": i}
            })

        if not records:
            return None, {}

        texts = [r["text"] for r in records]
        embeds = self.embed(texts)

        index = faiss.IndexFlatIP(EMBED_DIM)
        index.add(embeds)
        
        meta = {i: records[i] for i in range(len(records))}

        return index, meta

class RAGEngine:
    def __init__(self, index, meta):
        self.index = index
        self.meta = meta

    def embed_query(self, q):
        r = requests.post(EMBED_URL, json={"chunks": [q]}, timeout=30)
        r.raise_for_status()
        return np.array(r.json()["embeddings"], dtype="float32")

    def retrieve(self, query, k=4):
        qv = self.embed_query(query)
        scores, ids = self.index.search(qv, 15)

        chunks = []
        for s, i in zip(scores[0], ids[0]):
            if i == -1: continue
            if s < 0.4: continue 
            chunks.append(self.meta[i]["text"]) 

        return chunks[:k]

    def stream_answer(self, question, chunks):
        context = "\n\n".join(chunks)
        prompt = f"""<|im_start|>system
                                Use ONLY the context.
                                <|im_end|>
                                <|im_start|>user
                                Context:
                                {context}

                                Question:
                                {question}
                                <|im_end|>
                                <|im_start|>assistant
                                """

        
        try:
            with requests.post(
                GEN_URL,
                json={"prompt": prompt, "max_tokens": 512, "temperature": 0.2},
                stream=True,
                timeout=300
            ) as r:
                
                start_time = time.perf_counter()
                
                for chunk in r.iter_content(chunk_size=None):
                    if chunk:
                        text_chunk = chunk.decode("utf-8", errors="ignore")
                        yield json.dumps({"token": text_chunk}).encode() + b"\n"
                        
                total_time = time.perf_counter() - start_time
                metrics = {
                    "latency_ms": int(total_time * 1000),
                    "tokens_generated": 0,
                    "tokens_per_second": 0,
                    "retrieved_chunks": len(chunks),
                    "done": True
                }
                yield json.dumps(metrics).encode() + b"\n"
                
        except Exception as e:
            yield json.dumps({"error": str(e)}).encode() + b"\n"


app = FastAPI(title="RAG")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

sessions: dict = {} 
rate_log: dict = {}
RATE_LIMIT = 30
RATE_WINDOW = 60


def rate_limiter(request: Request):
    ip = request.client.host
    now = time.time()
    window_start = now - RATE_WINDOW
    hits = [t for t in rate_log.get(ip, []) if t > window_start]
    if len(hits) >= RATE_LIMIT:
        raise HTTPException(429, "Rate limit exceeded")
    hits.append(now)
    rate_log[ip] = hits

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def process_upload(session_id: str, content: bytes, filename: str):
    ingestor = Ingestor()
    index, meta = ingestor.ingest_bytes(content, filename)
    
    if index and meta:
        sessions[session_id] = RAGEngine(index, meta)
    else:
        pass

@app.post("/upload", dependencies=[Depends(rate_limiter)])
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    content = await file.read()
    background_tasks.add_task(process_upload, session_id, content, file.filename)
    return {"session_id": session_id, "status": "processing"}

class ChatRequest(BaseModel):
    session_id: str
    question: str

@app.post("/chat", dependencies=[Depends(rate_limiter)])
async def chat(req: ChatRequest):
    if req.session_id not in sessions:
        raise HTTPException(400, "Session expired or processing not complete.")
    
    engine = sessions[req.session_id] 
    

    chunks = engine.retrieve(req.question)
    
    if not chunks:
        async def yield_error():
            msg = "I cannot find this information in the uploaded document."
            for word in msg.split():
                 yield json.dumps({"token": word + " "}).encode() + b"\n"
                 await asyncio.sleep(0.05)
            yield json.dumps({"done": True, "latency_ms": 0, "retrieved_chunks": 0}).encode() + b"\n"
            
        return StreamingResponse(yield_error(), media_type="application/x-ndjson")

    return StreamingResponse(
        engine.stream_answer(req.question, chunks),
        media_type="application/x-ndjson"
    )

@app.post("/reset")
def reset(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared"}
