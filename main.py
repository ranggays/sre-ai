from fastapi import FastAPI, Request
from rag_module import answer_with_rag, ingest_pdf_by_url
from agentic_module import run_agentic_task
#from kg_module import get_knowledge_graph

app = FastAPI()

@app.post("/ingest")
async def ingest_endpoint(request: Request):
    data = await request.json()
    pdf_url = data.get("pdf_url")
    if not pdf_url:
        return {"error": "pdf_url is required"}
    success = ingest_pdf_by_url(pdf_url)
    return {"success": success}

@app.post("/rag")
async def rag_endpoint(request: Request):
    data = await request.json()
    return answer_with_rag(data["question"])

@app.post("/agentic")
async def agentic_endpoint(request: Request):
    data = await request.json()
    return run_agentic_task(data["task"])

"""
@app.get("/kg")
async def kg_endpoint():
    return get_knowledge_graph()
"""