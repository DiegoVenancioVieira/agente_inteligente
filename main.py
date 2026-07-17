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
import unicodedata
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

# --- classificacao de intencao (API para outros sistemas; substitui o Dialogflow)
# INTENTS_SEED: copia read-only que vem DENTRO da imagem (Dockerfile COPY sources).
# INTENTS_PATH: fonte-da-verdade editavel, no VOLUME persistente (/data). A tela de
# gestao grava aqui; o reindex le daqui. Semeada do SEED no 1o boot (_ensure_intents_file).
INTENTS_SEED = os.getenv("INTENTS_SEED", "./sources/intents-1doc.json")
INTENTS_PATH = os.getenv("INTENTS_PATH", "/data/intents-1doc.json")
# CALIBRADOS em 2026-07-17 com bge-m3 real (scripts/testa_intent.py):
# 14 frases dentro do escopo (min 0.669, media 0.835) x 8 fora (max 0.681, media 0.536).
# Corte em 0.70: nao aceita nenhuma das 8 de fora e perde 1 das 14 de dentro -- e essa
# 1 vinha com o assunto ERRADO, entao recusar era o certo. O custo e assimetrico:
# formulario errado e pior que nenhum formulario, entao erra-se para o lado de recusar.
INTENT_THRESHOLD = float(os.getenv("INTENT_THRESHOLD", "0.70"))   # abaixo disso: nao identificado
# 0.04 pega as 7 colisoes reais do JSON (todas com margem 0.00-0.03) sem estragar as
# consultas limpas (margem mediana 0.12).
INTENT_MARGIN = float(os.getenv("INTENT_MARGIN", "0.04"))         # 1o e 2o colados: ambiguo
# /intent e' servidor-a-servidor: TODAS as chamadas chegam do mesmo IP, entao o
# limite por IP do /ask (20/min) derrubaria a integracao. Limite proprio e alto.
INTENT_RATE_LIMIT_PER_MIN = int(os.getenv("INTENT_RATE_LIMIT_PER_MIN", "600"))
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
    "sermulher": {
        "label": "SERMULHER",
        "welcome": ("Olá! 👋 Sou o assistente virtual da SERMULHER (Secretaria Municipal do Respeito "
                    "às Políticas Públicas para as Mulheres) de Aracaju. Posso orientar sobre nossos "
                    "serviços e onde buscar ajuda. Se você está em perigo agora, ligue 190. "
                    "Como posso ajudar?"),
        "chips": ["Sofri violência, onde busco ajuda?",
                  "O que é o Disque 180?",
                  "Quais programas a secretaria oferece?"],
    },
    "integraju": {
        "label": "IntegrAju",
        "welcome": ("Olá! 👋 Sou o assistente virtual do IntegrAju, a plataforma digital de serviços "
                    "da Prefeitura de Aracaju. Posso orientar sobre solicitações de serviços, "
                    "denúncias, sugestões e acompanhamento de demandas. Como posso ajudar?"),
        "chips": ["O que é o IntegrAju?",
                  "Como solicito um serviço?",
                  "Como acompanho minha solicitação?"],
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


# Siglas de orgao que aparecem como sufixo no "value" do 1doc.
# Servem para desambiguar sinonimos repetidos ("nota fiscal" existe em 23 orgaos).
ORGAOS: dict[str, str] = {
    "seplog": "SEPLOG — Planejamento e Gestão",
    "emurb": "EMURB — Obras e Urbanização",
    "semfaz": "SEMFAZ — Finanças",
    "sms": "SMS — Saúde",
    "smtt": "SMTT — Transporte e Trânsito",
    "emsurb": "EMSURB — Serviços Urbanos",
    "sema": "SEMA — Meio Ambiente",
    "ajuprev": "AJUPREV — Previdência Municipal",
    "semfas": "SEMFAS — Assistência Social",
    "funcaju": "FUNCAJU — Arte e Cultura",
    "semed": "SEMED — Educação",
    "fundat": "FUNDAT — Qualificação e Trabalho",
    "sejesp": "SEJESP — Esporte",
    "secult": "SECULT — Fomento à Cultura",
    "seminfra": "SEMINFRA — Infraestrutura",
    "secom": "SECOM — Comunicação Social",
    "cgm": "CGM — Controladoria do Município",
    "pgm": "PGM — Advocacia do Município",
    "segov": "SEGOV — Secretaria de Governo",
    "setur": "SETUR — Turismo",
    "semde": "SEMDE — Desenvolvimento Econômico e Inovação",
    "semdef": "SEMDEF — Inclusão Aju",
    "sermulher": "SERMULHER — Políticas Públicas para as Mulheres",
    "ssm": "SSM AJU — Segurança e Cidadania",
    "procon": "PROCON",
    "nucar": "NUCAR",
}
_ORGAO_RE = re.compile(r"\b(" + "|".join(ORGAOS) + r")\b")

# --------------------------------------------------------------- estado em RAM
_lock = asyncio.Lock()
_vecs_by_ws: dict[str, np.ndarray] = {}                     # ws -> (n, dim)
_meta_by_ws: dict[str, list] = defaultdict(list)            # ws -> [{id,question,answer,...}]
_stats_by_ws: dict[str, dict] = defaultdict(lambda: {"hits": 0, "misses": 0, "refusals": 0})

# indice de intencoes: um vetor por SINONIMO, apontando para o "value" do 1doc
_intent_vecs: np.ndarray | None = None                      # (n, dim)
_intent_meta: list[dict] = []                               # [{synonym, value, orgao}]
_intent_orgs: np.ndarray | None = None                      # (n,) sigla ou ""


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def _norm_intent(t: str) -> str:
    """Normalizacao do lado das INTENCOES. O json_normalizado.json vem sem acento
    ('isencao', 'demolicao'), mas o cidadao escreve com ('isenção'). Sem tirar o
    acento dos dois lados, a consulta e o sinonimo nunca casam de verdade."""
    t = unicodedata.normalize("NFKD", _norm_text(t))
    return "".join(c for c in t if not unicodedata.combining(c))


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
    con.execute(
        "CREATE TABLE IF NOT EXISTS intents ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, synonym TEXT, value TEXT, "
        "orgao TEXT, embedding BLOB)"
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


def _orgao_de(value: str) -> str:
    """Extrai a sigla do orgao do 'value' do 1doc. Vazio nos values genericos."""
    m = _ORGAO_RE.search(value)
    return m.group(1) if m else ""


def _load_intents():
    """Carrega o indice de intencoes (1 vetor por sinonimo) do SQLite para a RAM."""
    global _intent_vecs, _intent_orgs
    _intent_meta.clear()
    con = _db()
    rows = con.execute(
        "SELECT synonym, value, orgao, embedding FROM intents ORDER BY id").fetchall()
    con.close()
    if not rows:
        _intent_vecs, _intent_orgs = None, None
        return
    mats = []
    for syn, val, org, emb in rows:
        _intent_meta.append({"synonym": syn, "value": val, "orgao": org or ""})
        mats.append(np.frombuffer(emb, dtype=np.float32))
    _intent_vecs = np.vstack(mats)
    _intent_orgs = np.array([m["orgao"] for m in _intent_meta])


def _read_intents_file() -> list[dict]:
    """Le a fonte-da-verdade (volume). Cai no SEED da imagem se ainda nao semeada."""
    path = INTENTS_PATH if os.path.exists(INTENTS_PATH) else INTENTS_SEED
    return json.load(open(path, encoding="utf-8"))


def _write_intents_file(data: list[dict]):
    """Grava a fonte-da-verdade de forma atomica (tmp + replace)."""
    d = os.path.dirname(INTENTS_PATH)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = INTENTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, INTENTS_PATH)


def _ensure_intents_file():
    """1o boot: copia o SEED da imagem para o volume, para virar editavel."""
    if not os.path.exists(INTENTS_PATH) and os.path.exists(INTENTS_SEED):
        _write_intents_file(json.load(open(INTENTS_SEED, encoding="utf-8")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_cache()
    _ensure_intents_file()
    _load_intents()
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


@app.get("/api-intent", include_in_schema=False)
async def api_intent_page():
    return FileResponse("web/api-intent.html")


@app.get("/relatorio-intent", include_in_schema=False)
async def relatorio_intent_page():
    return FileResponse("web/relatorio-intent.html")


@app.get("/admin-intent", include_in_schema=False)
async def admin_intent_page():
    return FileResponse("web/admin-intent.html")


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


async def _embed_batch(client: httpx.AsyncClient, texts: list[str]) -> list[np.ndarray]:
    """Embeda em lote via /api/embed. Cai no /api/embeddings (1 a 1) se o Ollama for antigo."""
    out: list[np.ndarray] = []
    for i in range(0, len(texts), 64):
        chunk = texts[i:i + 64]
        try:
            r = await client.post(f"{OLLAMA_URL}/api/embed",
                                  json={"model": EMBEDDER, "input": chunk}, timeout=180)
            r.raise_for_status()
            out.extend(_unit(e) for e in r.json()["embeddings"])
        except (httpx.HTTPStatusError, KeyError):
            for t in chunk:
                out.append(await _embed(client, t))
    return out


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


def _search_intents(vec: np.ndarray, orgao: str = "", top_n: int = 3) -> list[dict]:
    """Ranqueia os assuntos do 1doc. Varios sinonimos apontam para o mesmo value,
    entao colapsa por value guardando o melhor sinonimo de cada um."""
    if _intent_vecs is None:
        return []
    sims = _intent_vecs @ vec
    idx = np.where(_intent_orgs == orgao)[0] if orgao else range(len(sims))
    best: dict[str, tuple[float, str]] = {}
    for i in idx:
        m = _intent_meta[i]
        s = float(sims[i])
        if m["value"] not in best or s > best[m["value"]][0]:
            best[m["value"]] = (s, m["synonym"])
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:top_n]
    return [{"intent": v, "score": round(s, 4), "matched_synonym": syn,
             "orgao": _orgao_de(v)} for v, (s, syn) in ranked]


# ------------------------------------------------- rate-limit / admin / log
_hits_by_ip: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _rate_ok(ip: str, limit: int = 0) -> bool:
    now = time.time()
    dq = _hits_by_ip[ip]
    while dq and now - dq[0] > 60:
        dq.popleft()
    if len(dq) >= (limit or RATE_LIMIT_PER_MIN):
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


# ---------------------------------------------- intencao (API p/ outros sistemas)
class IntentIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    orgao: str | None = None      # filtra por secretaria; resolve sinonimo ambiguo
    top_n: int = Field(default=3, ge=1, le=10)


@app.post("/intent")
async def intent(body: IntentIn, request: Request):
    """Classifica texto livre em um assunto do 1doc. NAO usa LLM: so embedding +
    cosseno contra os sinonimos, entao nao ha como inventar um assunto inexistente
    (o retorno sai sempre da lista) e responde em ~150ms sem entrar na fila do 7b."""
    if not _rate_ok(_client_ip(request), INTENT_RATE_LIMIT_PER_MIN):
        raise HTTPException(429, "Muitas requisicoes. Tente em instantes.")
    if _intent_vecs is None:
        raise HTTPException(503, "Indice de intencoes vazio. Rode POST /intent/reindex.")
    orgao = (body.orgao or "").strip().lower()
    if orgao and orgao not in ORGAOS:
        raise HTTPException(400, f"orgao invalido. Validos: {sorted(ORGAOS)}")

    t0 = time.time()
    vec = await _embed(request.app.state.client, _norm_intent(body.text))
    cands = _search_intents(vec, orgao, body.top_n)
    took = int((time.time() - t0) * 1000)

    if not cands or cands[0]["score"] < INTENT_THRESHOLD:
        return {"status": "nao_identificado", "intent": None, "score":
                cands[0]["score"] if cands else 0.0, "candidates": cands,
                "took_ms": took}
    # 1o e 2o colados: o texto nao distingue os dois (ex.: "nota fiscal", que
    # existe em 23 orgaos). Devolve os candidatos p/ o chamador desambiguar.
    if len(cands) > 1 and (cands[0]["score"] - cands[1]["score"]) < INTENT_MARGIN:
        return {"status": "ambiguo", "intent": None, "score": cands[0]["score"],
                "candidates": cands, "took_ms": took}
    return {"status": "ok", "intent": cands[0]["intent"], "score": cands[0]["score"],
            "matched_synonym": cands[0]["matched_synonym"],
            "orgao": cands[0]["orgao"], "candidates": cands[1:], "took_ms": took}


@app.get("/intent/orgaos")
async def intent_orgaos():
    return {"orgaos": [{"sigla": k, "label": v} for k, v in ORGAOS.items()]}


@app.post("/intent/reindex", dependencies=[Depends(require_admin)])
async def intent_reindex(request: Request):
    """Reconstroi TODO o indice a partir do arquivo. Demorado (1 embedding por
    sinonimo): so p/ mudanca em massa ou troca de embedder. Para adicionar 1
    assunto use POST /intent/subjects, que embeda so o que mudou."""
    if not os.path.exists(INTENTS_PATH) and not os.path.exists(INTENTS_SEED):
        raise HTTPException(400, f"arquivo nao encontrado: {INTENTS_PATH}")
    data = _read_intents_file()
    pares = [(_norm_intent(s), e["value"]) for e in data for s in e.get("synonyms", [])
             if s and s.strip()]
    if not pares:
        raise HTTPException(400, "JSON sem sinonimos")

    t0 = time.time()
    vecs = await _embed_batch(request.app.state.client, [s for s, _ in pares])
    con = _db()
    con.execute("DELETE FROM intents")
    con.executemany(
        "INSERT INTO intents (synonym, value, orgao, embedding) VALUES (?,?,?,?)",
        [(s, v, _orgao_de(v), vec.astype(np.float32).tobytes())
         for (s, v), vec in zip(pares, vecs)])
    con.commit()
    con.close()
    async with _lock:
        _load_intents()
    return {"status": "ok", "sinonimos": len(pares),
            "assuntos": len({v for _, v in pares}),
            "segundos": round(time.time() - t0, 1)}


# --------------------- gestao de assuntos (tela /admin-intent, para a equipe)
def _clean_synonyms(syns: list[str]) -> list[str]:
    """Tira espacos, vazios e duplicatas, preservando a ordem e o texto original."""
    out, vistos = [], set()
    for s in syns:
        s = re.sub(r"\s+", " ", (s or "").strip())
        k = _norm_intent(s)
        if s and k not in vistos:
            vistos.add(k)
            out.append(s)
    return out


class SubjectIn(BaseModel):
    value: str = Field(min_length=1, max_length=300)
    synonyms: list[str] = Field(min_length=1)


class SubjectDelIn(BaseModel):
    value: str = Field(min_length=1)


@app.get("/intent/subjects", dependencies=[Depends(require_admin)])
async def intent_subjects(q: str = "", limit: int = 50, offset: int = 0):
    """Lista/busca assuntos do arquivo-fonte (texto original, legivel)."""
    data = _read_intents_file()
    ql = _norm_intent(q) if q else ""
    if ql:
        data = [e for e in data
                if ql in _norm_intent(e["value"] + " " + " ".join(e.get("synonyms", [])))]
    page = data[offset:offset + limit]
    return {"total": len(data), "count": len(page),
            "subjects": [{"value": e["value"], "orgao": _orgao_de(e["value"]),
                          "synonyms": e.get("synonyms", [])} for e in page]}


@app.post("/intent/subjects", dependencies=[Depends(require_admin)])
async def intent_subject_upsert(body: SubjectIn, request: Request):
    """Cria ou atualiza um assunto AO VIVO (sem redeploy, sem reindex completo).
    Embeda so os sinonimos deste assunto. O 'value' e' gravado como veio: PRECISA
    bater exatamente com a string do 1doc."""
    value = re.sub(r"\s+", " ", body.value.strip())
    synonyms = _clean_synonyms(body.synonyms)
    if not synonyms:
        raise HTTPException(400, "informe ao menos um sinonimo nao-vazio")

    pares = [(_norm_intent(s), value) for s in synonyms]
    vecs = await _embed_batch(request.app.state.client, [s for s, _ in pares])

    data = _read_intents_file()
    idx = next((i for i, e in enumerate(data) if e["value"] == value), None)
    novo = idx is None
    if novo:
        data.append({"value": value, "synonyms": synonyms})
    else:
        data[idx] = {**data[idx], "value": value, "synonyms": synonyms}

    con = _db()
    con.execute("DELETE FROM intents WHERE value=?", (value,))
    con.executemany(
        "INSERT INTO intents (synonym, value, orgao, embedding) VALUES (?,?,?,?)",
        [(s, v, _orgao_de(v), vec.astype(np.float32).tobytes())
         for (s, v), vec in zip(pares, vecs)])
    con.commit()
    con.close()
    _write_intents_file(data)
    async with _lock:
        _load_intents()
    return {"status": "ok", "value": value, "novo": novo,
            "sinonimos": len(synonyms), "orgao": _orgao_de(value)}


@app.post("/intent/subjects/delete", dependencies=[Depends(require_admin)])
async def intent_subject_delete(body: SubjectDelIn):
    """Remove um assunto ao vivo (do arquivo-fonte e do indice)."""
    value = re.sub(r"\s+", " ", body.value.strip())
    data = _read_intents_file()
    novo = [e for e in data if e["value"] != value]
    if len(novo) == len(data):
        raise HTTPException(404, f"assunto nao encontrado: {value}")
    con = _db()
    con.execute("DELETE FROM intents WHERE value=?", (value,))
    con.commit()
    con.close()
    _write_intents_file(novo)
    async with _lock:
        _load_intents()
    return {"status": "ok", "removido": value}


@app.get("/health")
async def health():
    return {"status": "ok", "threshold": HIT_THRESHOLD, "ttl_days": TTL_DAYS,
            "secretarias": {s: len(_meta_by_ws.get(s, [])) for s in SECRETARIAS},
            # sinonimos=0 depois de um deploy significa que falta rodar /intent/reindex
            "intent": {"sinonimos": 0 if _intent_vecs is None else int(_intent_vecs.shape[0]),
                       "assuntos": len({m["value"] for m in _intent_meta}),
                       "threshold": INTENT_THRESHOLD, "margin": INTENT_MARGIN}}


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
