"""
FastAPI RAG Application
=======================
Ported from reference implementation (Ingestor/RAGEngine).
Uses PyMuPDF (fitz) for parsing and strict prompt templates.
"""

import time
import uuid
import asyncio
import io
import requests
import numpy as np
import faiss
import json
import fitz # PyMuPDF

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

# =============================================================================
# CONFIGURATION
# =============================================================================

EMBED_URL = "https://abhinavdread-bge-en-ft-optimised.hf.space/embed"
GEN_URL   = "https://abhinavdread-qwen-1-5b-q4-k-m.hf.space/generate"
EMBED_DIM = 128

# =============================================================================
# CORE LOGIC (Ported from Reference)
# =============================================================================

class Ingestor:
    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text):
        chunks, i = [], 0
        while i < len(text):
            chunks.append(text[i:i+self.chunk_size])
            i += self.chunk_size - self.overlap
        return chunks

    def embed(self, texts):
        # Using the reference implementation's embed call
        r = requests.post(EMBED_URL, json={"chunks": texts}, timeout=120)
        r.raise_for_status()
        return np.array(r.json()["embeddings"], dtype="float32")

    def ingest_bytes(self, file_bytes: bytes, filename: str):
        """
        Adapted from reference 'ingest' to work with in-memory bytes.
        Returns (index, meta_dict).
        """
        doc = fitz.open(stream=file_bytes, filetype=filename.split(".")[-1])
        records = []

        # Extract and chunk
        for p, page in enumerate(doc):
            txt = page.get_text().strip()
            if not txt: continue
            for ch in self.chunk(txt):
                records.append({
                    "text": ch,
                    "meta": {"page": p+1}
                })
        
        doc.close()
        
        if not records:
            return None, {}

        # Embed all chunks
        texts = [r["text"] for r in records]
        # Batch embedding if needed, but reference did it in one go (assuming small docs)
        # For safety/reliability with larger docs, we might batch, but stick to reference logic first.
        embeds = self.embed(texts)

        # Build Index
        index = faiss.IndexFlatIP(EMBED_DIM)
        index.add(embeds)
        
        # Build Meta (dict instead of file for in-memory)
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
        # Reference used k=15 for search, then filtered
        scores, ids = self.index.search(qv, 15)

        chunks = []
        for s, i in zip(scores[0], ids[0]):
            if i == -1: continue
            if s < 0.4: continue # Reference threshold
            chunks.append(self.meta[i]["text"]) # Note: meta key is int in this dict

        return chunks[:k]

    def stream_answer(self, question, chunks):
        context = "\n\n".join(chunks)
        # Exact prompt from reference
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
        # True streaming request
        # Reference Logic uses 'stream=True' and yields chunks
        # We wrap this to yield bytes for StreamingResponse
        
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
                        # Yield token for UI (NDJSON)
                        yield json.dumps({"token": text_chunk}).encode() + b"\n"
                        
                # Final Metrics for UI (Legacy requirement)
                # We can't easily calculate accurate metrics in this simple loop strictly following reference
                # but we need to send *something* to stop the frontend spinner if it expects 'done'
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

# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(title="Reliable RAG")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

sessions: dict = {} # Stores {session_id: RAGEngine_instance}
rate_log: dict = {}
RATE_LIMIT = 30
RATE_WINDOW = 60

# =============================================================================
# ROUTES
# =============================================================================

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
        # Handle empty/fail case safely
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
        # If session exists but engine not ready, it might be processing
        # Simple check for now
        raise HTTPException(400, "Session expired or processing not complete.")
    
    engine = sessions[req.session_id] # Type: RAGEngine
    
    # 1. Retrieve
    chunks = engine.retrieve(req.question)
    
    if not chunks:
        # Fallback for no context
        async def yield_error():
            msg = "I cannot find this information in the uploaded document."
            # Simulate streaming the fixed message
            for word in msg.split():
                 yield json.dumps({"token": word + " "}).encode() + b"\n"
                 await asyncio.sleep(0.05)
            yield json.dumps({"done": True, "latency_ms": 0, "retrieved_chunks": 0}).encode() + b"\n"
            
        return StreamingResponse(yield_error(), media_type="application/x-ndjson")

    # 2. Stream Answer
    # Returning the generator directly. 
    # FastAPI handles the iteration.
    
    return StreamingResponse(
        engine.stream_answer(req.question, chunks),
        media_type="application/x-ndjson"
    )

@app.post("/reset")
def reset(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared"}
