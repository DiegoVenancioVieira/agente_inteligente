"""
Cache semantico + API do agente de FAQ municipal.

Fica NA FRENTE do AnythingLLM:
  pergunta -> embedding (bge-m3) -> busca no cache
    - se similaridade >= THRESHOLD  -> devolve resposta cacheada (~0,2s, sem LLM)
    - senao                         -> chama o AnythingLLM (qwen2.5:7b) e grava no cache

Threshold 0,85 calibrado com dados reais:
  quase-identicas 0,89-1,00 (acerta) | parafrases 0,60-0,71 e diferentes 0,36-0,45 (vao ao LLM).
Recusas (fora de escopo) NAO sao cacheadas: ja sao rapidas e a FAQ pode mudar.
"""
import os
import json
import time
import asyncio
import sqlite3
import re
from contextlib import asynccontextmanager
from collections import deque, defaultdict

import httpx
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------- config (.env)
ANYTHINGLLM_URL = os.getenv("ANYTHINGLLM_URL", "http://192.168.0.118:3001").rstrip("/")
ANYTHINGLLM_API_KEY = os.getenv("ANYTHINGLLM_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.0.115:11434").rstrip("/")
EMBEDDER = os.getenv("OLLAMA_EMBEDDER", "bge-m3")
WORKSPACE = os.getenv("WORKSPACE_SLUG", "semde")
HIT_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85"))
TTL_DAYS = int(os.getenv("CACHE_TTL_DAYS", "30"))
DB_PATH = os.getenv("CACHE_DB_PATH", "./cache.db")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

API_BASE = f"{ANYTHINGLLM_URL}/api/v1"
AUTH = {"Authorization": f"Bearer {ANYTHINGLLM_API_KEY}"}

# --------------------------------------------------------------- estado em RAM
_lock = asyncio.Lock()
_vecs: np.ndarray | None = None          # (n, dim) vetores normalizados
_meta: list[dict] = []                   # [{id, question, answer, created_at}]
_stats = {"hits": 0, "misses": 0, "refusals": 0}


def _norm_text(t: str) -> str:
    """Normalizacao leve: minusculas + espacos colapsados (ajuda a bater repetidos)."""
    return re.sub(r"\s+", " ", t.strip().lower())


def _unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n else a


def _db():
    # garante que a pasta do banco exista (ex.: /data do volume) antes de abrir
    d = os.path.dirname(DB_PATH)
    if d:
        os.makedirs(d, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "CREATE TABLE IF NOT EXISTS cache ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, answer TEXT, "
        "embedding BLOB, created_at REAL)"
    )
    # perguntas que o agente NAO soube responder (recusa) -> insumo p/ as secretarias
    con.execute(
        "CREATE TABLE IF NOT EXISTS unanswered ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, created_at REAL)"
    )
    return con


def _load_cache():
    """Carrega o cache nao-expirado do SQLite para a matriz em memoria."""
    global _vecs, _meta
    cutoff = time.time() - TTL_DAYS * 86400
    con = _db()
    rows = con.execute(
        "SELECT id, question, answer, embedding, created_at FROM cache "
        "WHERE created_at > ? ORDER BY id", (cutoff,)
    ).fetchall()
    con.close()
    _meta = []
    mats = []
    for rid, q, a, emb, ts in rows:
        _meta.append({"id": rid, "question": q, "answer": a, "created_at": ts})
        mats.append(np.frombuffer(emb, dtype=np.float32))
    _vecs = np.vstack(mats) if mats else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_cache()
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    yield
    await app.state.client.aclose()


app = FastAPI(title="Agente FAQ Municipal — API + Cache Semantico", lifespan=lifespan)

# CORS liberado: /ask e um endpoint publico de FAQ (sem dados sensiveis).
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/", include_in_schema=False)
async def index():
    """Serve a interface de chat dos municipes."""
    return FileResponse("web/index.html")


# ------------------------------------------------------------- infra externa
async def _embed(client: httpx.AsyncClient, text: str) -> np.ndarray:
    r = await client.post(f"{OLLAMA_URL}/api/embeddings",
                          json={"model": EMBEDDER, "prompt": text}, timeout=60)
    r.raise_for_status()
    return _unit(r.json()["embedding"])


async def _ask_llm(client: httpx.AsyncClient, question: str) -> tuple[str, int]:
    """Chama o AnythingLLM em thread NOVA (evita contaminacao de historico)."""
    r = await client.post(f"{API_BASE}/workspace/{WORKSPACE}/thread/new",
                          headers=AUTH, json={}, timeout=20)
    r.raise_for_status()
    slug = r.json()["thread"]["slug"]
    try:
        r = await client.post(
            f"{API_BASE}/workspace/{WORKSPACE}/thread/{slug}/chat",
            headers=AUTH, json={"message": question, "mode": "query"}, timeout=180,
        )
        r.raise_for_status()
        d = json.loads(r.content)  # decode UTF-8 explicito (evita mojibake de acentos)
        return d.get("textResponse", ""), len(d.get("sources", []))
    finally:
        try:
            await client.delete(f"{API_BASE}/workspace/{WORKSPACE}/thread/{slug}",
                                headers=AUTH, timeout=15)
        except Exception:
            pass  # thread orfa nao quebra a resposta


async def _store(question: str, answer: str, vec: np.ndarray):
    global _vecs, _meta
    ts = time.time()
    con = _db()
    cur = con.execute(
        "INSERT INTO cache (question, answer, embedding, created_at) VALUES (?,?,?,?)",
        (question, answer, vec.astype(np.float32).tobytes(), ts),
    )
    con.commit()
    rid = cur.lastrowid
    con.close()
    async with _lock:
        _meta.append({"id": rid, "question": question, "answer": answer, "created_at": ts})
        _vecs = vec.reshape(1, -1) if _vecs is None else np.vstack([_vecs, vec])


def _search(vec: np.ndarray) -> tuple[float, int]:
    """Melhor similaridade (cosseno) no cache. Retorna (score, indice) ou (0,-1)."""
    if _vecs is None or len(_meta) == 0:
        return 0.0, -1
    sims = _vecs @ vec            # vetores ja normalizados -> produto = cosseno
    i = int(np.argmax(sims))
    return float(sims[i]), i


# ------------------------------------------------- rate-limit / admin / log
_hits_by_ip: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    """IP real do cidadao (respeita X-Forwarded-For do Traefik/Coolify)."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _rate_ok(ip: str) -> bool:
    """Janela deslizante de 60s por IP. Protege o LLM contra abuso/DoS."""
    now = time.time()
    dq = _hits_by_ip[ip]
    while dq and now - dq[0] > 60:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_PER_MIN:
        return False
    dq.append(now)
    return True


def require_admin(authorization: str = Header(default="")):
    """Protege endpoints administrativos (fecha por padrao se ADMIN_TOKEN vazio)."""
    if not ADMIN_TOKEN:
        raise HTTPException(503, "ADMIN_TOKEN nao configurado no servidor")
    if authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(401, "Nao autorizado")


def _log_unanswered(question: str):
    con = _db()
    con.execute("INSERT INTO unanswered (question, created_at) VALUES (?,?)",
                (question, time.time()))
    con.commit()
    con.close()


# ------------------------------------------------------------------ API
class AskIn(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)


class AskOut(BaseModel):
    answer: str
    cached: bool
    similarity: float
    latency_ms: int
    matched_question: str | None = None


@app.post("/ask", response_model=AskOut)
async def ask(body: AskIn, request: Request):
    if not _rate_ok(_client_ip(request)):
        raise HTTPException(
            429, "Muitas perguntas em pouco tempo. Aguarde alguns segundos e tente novamente.")
    client: httpx.AsyncClient = app.state.client
    t0 = time.time()
    try:
        vec = await _embed(client, _norm_text(body.question))
    except Exception as e:
        raise HTTPException(502, f"Falha ao gerar embedding: {e}")

    score, idx = _search(vec)
    if idx >= 0 and score >= HIT_THRESHOLD:
        _stats["hits"] += 1
        m = _meta[idx]
        return AskOut(answer=m["answer"], cached=True, similarity=round(score, 3),
                      latency_ms=int((time.time() - t0) * 1000),
                      matched_question=m["question"])

    # miss -> LLM
    _stats["misses"] += 1
    try:
        answer, n_sources = await _ask_llm(client, body.question)
    except Exception as e:
        raise HTTPException(502, f"Falha ao consultar o AnythingLLM: {e}")

    if n_sources > 0:                      # so cacheia respostas fundamentadas
        await _store(body.question, answer, vec)
    else:
        _stats["refusals"] += 1            # recusa: nao cacheia (ja e rapida)
        _log_unanswered(body.question)     # registra p/ a secretaria melhorar a FAQ

    return AskOut(answer=answer, cached=False, similarity=round(score, 3),
                  latency_ms=int((time.time() - t0) * 1000))


@app.get("/health")
async def health():
    return {"status": "ok", "workspace": WORKSPACE, "cache_size": len(_meta),
            "threshold": HIT_THRESHOLD, "ttl_days": TTL_DAYS}


@app.get("/stats")
async def stats():
    total = _stats["hits"] + _stats["misses"]
    return {**_stats, "cache_size": len(_meta),
            "hit_rate": round(_stats["hits"] / total, 3) if total else 0.0}


@app.get("/unanswered", dependencies=[Depends(require_admin)])
async def unanswered(limit: int = 100):
    """Perguntas que o agente nao soube responder, agrupadas por frequencia.
    Insumo para as secretarias saberem o que falta na FAQ."""
    con = _db()
    rows = con.execute(
        "SELECT question, COUNT(*) c, MAX(created_at) last FROM unanswered "
        "GROUP BY lower(trim(question)) ORDER BY c DESC, last DESC LIMIT ?",
        (limit,),
    ).fetchall()
    con.close()
    return {"unanswered": [
        {"question": q, "count": c,
         "last_seen": time.strftime("%Y-%m-%d %H:%M", time.localtime(last))}
        for q, c, last in rows
    ]}


@app.post("/cache/clear", dependencies=[Depends(require_admin)])
async def clear_cache():
    """Limpa TODO o cache. Chame quando uma secretaria atualizar um PDF da FAQ."""
    global _vecs, _meta
    con = _db()
    con.execute("DELETE FROM cache")
    con.commit()
    con.close()
    async with _lock:
        _vecs = None
        _meta = []
    return {"status": "cache limpo"}
