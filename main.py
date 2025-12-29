import logging
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import uvicorn
from fastapi import FastAPI, Request
import medqa
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/ask")
async def ask_question(request: Request):
    json_data = await request.json()
    query = json_data["question"]
    kwargs = {}
    if "thread" in json_data.keys():
        t = json_data["thread"]
        kwargs["thread"] = t
    if "user" in json_data.keys():
        u = json_data["user"]
        kwargs["user"] = u
    answer = medqa.run(query, **kwargs)
    return { "MedQA": answer }

@app.post("/raw")
async def raw_test(request: Request):
    json_data = await request.json()
    query = json_data["question"]
    answer = medqa.raw(query)
    return {"MedQA": answer}

@app.post("/stream")
async def stream_question(request: Request):
    json_data = await request.json()
    q = json_data["question"]
    kwargs = {}
    if "thread" in json_data.keys():
        t = json_data["thread"]
        kwargs["thread"] = t
    if "user" in json_data.keys():
        u = json_data["user"]
        kwargs["user"] = u
    messages = ""
    async def answer(resp, query, **kargs):
        for chunk in medqa.stream(query, **kargs):
            msg = chunk.choices[0].delta.content
            if msg:
                resp += msg
                yield msg
        medqa.save_message(q, messages, **kargs)
    return StreamingResponse(answer(messages, q, **kwargs), media_type="text/plain")

@app.post("/route")
async def route(request: Request):
    json_data = await request.json()
    q = json_data["question"]
    kwargs = {}
    if "thread" in json_data.keys():
        t = json_data["thread"]
        kwargs["thread"] = t
    if "user" in json_data.keys():
        u = json_data["user"]
        kwargs["user"] = u
    answer = medqa.routing(q, **kwargs)
    return {"MedQA": answer}

@app.get("/")
async def root():
    return { "你好！": "朋友" }

if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=2424, reload=True)
