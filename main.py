"""
Cache semantico + API do agente de FAQ municipal (MULTI-SECRETARIA).

Fica NA FRENTE do AnythingLLM. Cada secretaria = um workspace do AnythingLLM.
  pergunta (+secretaria) -> embedding (bge-m3) -> busca no cache DAQUELA secretaria
    - similaridade >= THRESHOLD -> devolve resposta cacheada (~0,3s, sem LLM)
    - senao                     -> chama o AnythingLLM (qwen2.5:7b) do workspace e grava

Cache, sugestoes e recusas sao SEPARADOS por secretaria (workspace).
Recusas (fora de escopo) NAO sao cacheadas.
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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------- config (.env)
ANYTHINGLLM_URL = os.getenv("ANYTHINGLLM_URL", "http://192.168.0.118:3001").rstrip("/")
ANYTHINGLLM_API_KEY = os.getenv("ANYTHINGLLM_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.0.115:11434").rstrip("/")
EMBEDDER = os.getenv("OLLAMA_EMBEDDER", "bge-m3")
HIT_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85"))
TTL_DAYS = int(os.getenv("CACHE_TTL_DAYS", "30"))
DB_PATH = os.getenv("CACHE_DB_PATH", "./cache.db")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

API_BASE = f"{ANYTHINGLLM_URL}/api/v1"
AUTH = {"Authorization": f"Bearer {ANYTHINGLLM_API_KEY}"}

# Secretarias atendidas. Cada slug = workspace no AnythingLLM.
# Para adicionar uma nova: crie o workspace no AnythingLLM e acrescente aqui.
SECRETARIAS: dict[str, dict] = {
    "semde": {
        "label": "SEMDE",
        "welcome": ("Olá! 👋 Sou o assistente virtual da SEMDE (Secretaria Municipal do "
                    "Desenvolvimento Econômico e Inovação) de Aracaju. Como posso ajudar?"),
        "chips": ["Qual a alíquota do ISSQN pela Lei 183?",
                  "O MEI pode pedir o benefício?",
                  "O protocolo de intenções é um contrato?"],
    },
    "procon": {
        "label": "PROCON",
        "welcome": ("Olá! 👋 Sou o assistente virtual do PROCON de Aracaju. "
                    "Tire suas dúvidas sobre seus direitos como consumidor. Como posso ajudar?"),
        "chips": ["Comprei um produto com defeito, o que fazer?",
                  "Posso me arrepender de uma compra pela internet?",
                  "Fui cobrado indevidamente, tenho direito a algo?"],
    },
}
DEFAULT_WS = next(iter(SECRETARIAS))


def _valid_ws(ws: str | None) -> str:
    """Resolve a secretaria: default se vazia, 404 se desconhecida."""
    if not ws:
        return DEFAULT_WS
    if ws not in SECRETARIAS:
        raise HTTPException(404, f"Secretaria '{ws}' não encontrada")
    return ws


# --------------------------------------------------------------- estado em RAM
_lock = asyncio.Lock()
_vecs_by_ws: dict[str, np.ndarray] = {}                     # ws -> (n, dim)
_meta_by_ws: dict[str, list] = defaultdict(list)            # ws -> [{id,question,answer,...}]
_stats_by_ws: dict[str, dict] = defaultdict(lambda: {"hits": 0, "misses": 0, "refusals": 0})


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def _unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n else a


def _db():
    d = os.path.dirname(DB_PATH)
    if d:
        os.makedirs(d, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "CREATE TABLE IF NOT EXISTS cache ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, answer TEXT, "
        "embedding BLOB, created_at REAL)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS unanswered ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, created_at REAL)"
    )
    # migracoes idempotentes
    for tbl, col, ddl in [
        ("cache", "hits", "ALTER TABLE cache ADD COLUMN hits INTEGER DEFAULT 0"),
        ("cache", "workspace", f"ALTER TABLE cache ADD COLUMN workspace TEXT DEFAULT '{DEFAULT_WS}'"),
        ("unanswered", "workspace", f"ALTER TABLE unanswered ADD COLUMN workspace TEXT DEFAULT '{DEFAULT_WS}'"),
    ]:
        try:
            con.execute(ddl)
            con.commit()
        except sqlite3.OperationalError:
            pass  # coluna ja existe
    return con


def _load_cache():
    """Carrega o cache nao-expirado, agrupado por secretaria."""
    _vecs_by_ws.clear()
    _meta_by_ws.clear()
    cutoff = time.time() - TTL_DAYS * 86400
    con = _db()
    rows = con.execute(
        "SELECT id, question, answer, embedding, created_at, hits, workspace FROM cache "
        "WHERE created_at > ? ORDER BY id", (cutoff,)
    ).fetchall()
    con.close()
    mats: dict[str, list] = defaultdict(list)
    for rid, q, a, emb, ts, hits, ws in rows:
        ws = ws or DEFAULT_WS
        _meta_by_ws[ws].append({"id": rid, "question": q, "answer": a,
                                "created_at": ts, "hits": hits or 0})
        mats[ws].append(np.frombuffer(emb, dtype=np.float32))
    for ws, m in mats.items():
        _vecs_by_ws[ws] = np.vstack(m)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_cache()
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    yield
    await app.state.client.aclose()


app = FastAPI(title="Agente FAQ Municipal — Multi-secretaria", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("web/index.html")


@app.get("/relatorio", include_in_schema=False)
async def relatorio():
    return FileResponse("web/relatorio.html")


@app.get("/admin", include_in_schema=False)
async def admin_page():
    return FileResponse("web/admin.html")


@app.get("/documentacao", include_in_schema=False)
async def documentacao_page():
    return FileResponse("web/documentacao.html")


@app.get("/tutorial", include_in_schema=False)
async def tutorial_page():
    return FileResponse("web/tutorial.html")


app.mount("/static", StaticFiles(directory="web"), name="static")


# ------------------------------------------------------------- infra externa
async def _embed(client: httpx.AsyncClient, text: str) -> np.ndarray:
    r = await client.post(f"{OLLAMA_URL}/api/embeddings",
                          json={"model": EMBEDDER, "prompt": text}, timeout=60)
    r.raise_for_status()
    return _unit(r.json()["embedding"])


async def _ask_llm(client: httpx.AsyncClient, ws: str, question: str) -> tuple[str, int]:
    """Chama o AnythingLLM do workspace em thread NOVA (sem contaminacao)."""
    r = await client.post(f"{API_BASE}/workspace/{ws}/thread/new",
                          headers=AUTH, json={}, timeout=20)
    r.raise_for_status()
    slug = r.json()["thread"]["slug"]
    try:
        r = await client.post(
            f"{API_BASE}/workspace/{ws}/thread/{slug}/chat",
            headers=AUTH, json={"message": question, "mode": "query"}, timeout=180,
        )
        r.raise_for_status()
        d = json.loads(r.content)
        return d.get("textResponse", ""), len(d.get("sources", []))
    finally:
        try:
            await client.delete(f"{API_BASE}/workspace/{ws}/thread/{slug}",
                                headers=AUTH, timeout=15)
        except Exception:
            pass


async def _add_knowledge(client: httpx.AsyncClient, ws: str, question: str, answer: str) -> str:
    """Injeta um Q&A no workspace (vira conhecimento do bot na hora)."""
    text = f"Pergunta: {question}\nResposta: {answer}"
    r = await client.post(
        f"{API_BASE}/document/raw-text", headers=AUTH,
        json={"textContent": text,
              "metadata": {"title": f"{ws}-{int(time.time())}",
                           "description": "Resposta cadastrada pela secretaria"}},
        timeout=60)
    r.raise_for_status()
    docs = json.loads(r.content).get("documents", [])
    loc = docs[0]["location"] if docs else None
    if loc:
        await client.post(f"{API_BASE}/workspace/{ws}/update-embeddings",
                          headers=AUTH, json={"adds": [loc]}, timeout=120)
    return loc


async def _store(ws: str, question: str, answer: str, vec: np.ndarray):
    ts = time.time()
    con = _db()
    cur = con.execute(
        "INSERT INTO cache (question, answer, embedding, created_at, workspace) VALUES (?,?,?,?,?)",
        (question, answer, vec.astype(np.float32).tobytes(), ts, ws),
    )
    con.commit()
    rid = cur.lastrowid
    con.close()
    async with _lock:
        _meta_by_ws[ws].append({"id": rid, "question": question, "answer": answer,
                                "created_at": ts, "hits": 0})
        v = vec.reshape(1, -1)
        _vecs_by_ws[ws] = v if ws not in _vecs_by_ws else np.vstack([_vecs_by_ws[ws], vec])


def _search(ws: str, vec: np.ndarray) -> tuple[float, int]:
    """Melhor similaridade no cache DA secretaria. Retorna (score, indice) ou (0,-1)."""
    vecs = _vecs_by_ws.get(ws)
    if vecs is None or not _meta_by_ws.get(ws):
        return 0.0, -1
    sims = vecs @ vec
    i = int(np.argmax(sims))
    return float(sims[i]), i


# ------------------------------------------------- rate-limit / admin / log
_hits_by_ip: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _rate_ok(ip: str) -> bool:
    now = time.time()
    dq = _hits_by_ip[ip]
    while dq and now - dq[0] > 60:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_PER_MIN:
        return False
    dq.append(now)
    return True


def require_admin(authorization: str = Header(default="")):
    if not ADMIN_TOKEN:
        raise HTTPException(503, "ADMIN_TOKEN nao configurado no servidor")
    if authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(401, "Nao autorizado")


def _log_unanswered(ws: str, question: str):
    con = _db()
    con.execute("INSERT INTO unanswered (question, created_at, workspace) VALUES (?,?,?)",
                (question, time.time(), ws))
    con.commit()
    con.close()


# ------------------------------------------------------------------ API
class AskIn(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    secretaria: str | None = None


class AskOut(BaseModel):
    answer: str
    cached: bool
    similarity: float
    latency_ms: int
    matched_question: str | None = None


@app.get("/secretarias")
async def secretarias():
    """Lista das secretarias para o seletor da interface."""
    return {"secretarias": [
        {"slug": s, "label": m["label"], "welcome": m["welcome"], "chips": m["chips"]}
        for s, m in SECRETARIAS.items()
    ]}


@app.post("/ask", response_model=AskOut)
async def ask(body: AskIn, request: Request):
    if not _rate_ok(_client_ip(request)):
        raise HTTPException(429, "Muitas perguntas em pouco tempo. Aguarde e tente novamente.")
    ws = _valid_ws(body.secretaria)
    st = _stats_by_ws[ws]
    client: httpx.AsyncClient = app.state.client
    t0 = time.time()
    try:
        vec = await _embed(client, _norm_text(body.question))
    except Exception as e:
        raise HTTPException(502, f"Falha ao gerar embedding: {e}")

    score, idx = _search(ws, vec)
    if idx >= 0 and score >= HIT_THRESHOLD:
        st["hits"] += 1
        m = _meta_by_ws[ws][idx]
        m["hits"] = m.get("hits", 0) + 1
        try:
            con = _db()
            con.execute("UPDATE cache SET hits = hits + 1 WHERE id = ?", (m["id"],))
            con.commit()
            con.close()
        except Exception:
            pass
        return AskOut(answer=m["answer"], cached=True, similarity=round(score, 3),
                      latency_ms=int((time.time() - t0) * 1000), matched_question=m["question"])

    st["misses"] += 1
    try:
        answer, n_sources = await _ask_llm(client, ws, body.question)
    except Exception as e:
        raise HTTPException(502, f"Falha ao consultar o AnythingLLM: {e}")

    if n_sources > 0:
        await _store(ws, body.question, answer, vec)
    else:
        st["refusals"] += 1
        _log_unanswered(ws, body.question)

    return AskOut(answer=answer, cached=False, similarity=round(score, 3),
                  latency_ms=int((time.time() - t0) * 1000))


@app.get("/health")
async def health():
    return {"status": "ok", "threshold": HIT_THRESHOLD, "ttl_days": TTL_DAYS,
            "secretarias": {s: len(_meta_by_ws.get(s, [])) for s in SECRETARIAS}}


@app.get("/suggestions")
async def suggestions(n: int = 3, secretaria: str | None = None):
    """Perguntas mais frequentes da secretaria (por acertos de cache); completa com padrao."""
    ws = _valid_ws(secretaria)
    ranked = sorted((m for m in _meta_by_ws.get(ws, []) if m.get("hits", 0) > 0),
                    key=lambda m: m["hits"], reverse=True)
    qs = [m["question"] for m in ranked[:n]]
    for d in SECRETARIAS[ws]["chips"]:
        if len(qs) >= n:
            break
        if d not in qs:
            qs.append(d)
    return {"suggestions": qs}


@app.get("/stats")
async def stats(secretaria: str | None = None):
    ws = _valid_ws(secretaria)
    st = _stats_by_ws[ws]
    total = st["hits"] + st["misses"]
    return {**st, "secretaria": ws, "cache_size": len(_meta_by_ws.get(ws, [])),
            "hit_rate": round(st["hits"] / total, 3) if total else 0.0}


@app.get("/unanswered", dependencies=[Depends(require_admin)])
async def unanswered(limit: int = 100, secretaria: str | None = None):
    ws = _valid_ws(secretaria)
    con = _db()
    rows = con.execute(
        "SELECT question, COUNT(*) c, MAX(created_at) last FROM unanswered "
        "WHERE workspace = ? GROUP BY lower(trim(question)) ORDER BY c DESC, last DESC LIMIT ?",
        (ws, limit),
    ).fetchall()
    con.close()
    return {"secretaria": ws, "unanswered": [
        {"question": q, "count": c,
         "last_seen": time.strftime("%Y-%m-%d %H:%M", time.localtime(last))}
        for q, c, last in rows
    ]}


class AnswerIn(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    answer: str = Field(..., min_length=2, max_length=5000)
    secretaria: str | None = None


@app.post("/answer", dependencies=[Depends(require_admin)])
async def answer(body: AnswerIn):
    ws = _valid_ws(body.secretaria)
    client: httpx.AsyncClient = app.state.client
    try:
        loc = await _add_knowledge(client, ws, body.question, body.answer)
    except Exception as e:
        raise HTTPException(502, f"Falha ao gravar no AnythingLLM: {e}")
    con = _db()
    con.execute("DELETE FROM unanswered WHERE workspace = ? AND lower(trim(question)) = ?",
                (ws, body.question.strip().lower()))
    con.commit()
    con.close()
    return {"status": "resposta publicada", "location": loc}


class QuestionIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    secretaria: str | None = None


@app.post("/unanswered/dismiss", dependencies=[Depends(require_admin)])
async def dismiss(body: QuestionIn):
    ws = _valid_ws(body.secretaria)
    con = _db()
    con.execute("DELETE FROM unanswered WHERE workspace = ? AND lower(trim(question)) = ?",
                (ws, body.question.strip().lower()))
    con.commit()
    con.close()
    return {"status": "removida"}


@app.post("/cache/clear", dependencies=[Depends(require_admin)])
async def clear_cache(secretaria: str | None = None):
    """Limpa o cache de uma secretaria (?secretaria=) ou de todas."""
    con = _db()
    if secretaria:
        ws = _valid_ws(secretaria)
        con.execute("DELETE FROM cache WHERE workspace = ?", (ws,))
        con.commit()
        con.close()
        async with _lock:
            _vecs_by_ws.pop(ws, None)
            _meta_by_ws.pop(ws, None)
        return {"status": f"cache de {ws} limpo"}
    con.execute("DELETE FROM cache")
    con.commit()
    con.close()
    async with _lock:
        _vecs_by_ws.clear()
        _meta_by_ws.clear()
    return {"status": "cache limpo (todas as secretarias)"}
