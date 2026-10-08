#!/usr/bin/env python3
"""
Compara modelos de geracao do Ollama no hardware real do servidor.

Reproduz o fluxo do agente (modo "query" do AnythingLLM): recupera os trechos
mais parecidos da FAQ com o embedder (bge-m3) e pede ao modelo uma resposta
SO com base nesse contexto. Mede, por modelo:

  - carga: tempo para carregar o modelo na RAM;
  - TTFT: tempo ate o primeiro token (o que o cidadao sente como "travado");
  - total e tokens/s da geracao;
  - memoria: tamanho do modelo carregado, segundo o `ollama ps` (/api/ps);
  - qualidade: acerta os fatos esperados, recusa perguntas fora da FAQ e
    responde em portugues (checagem automatica + respostas salvas para leitura).

So usa a biblioteca padrao: roda no proprio servidor com `python3`.

Uso (no servidor, com o Ollama acessivel):
    python3 scripts/benchmark_modelos.py --ollama http://localhost:11434 --pull
    python3 scripts/benchmark_modelos.py --modelos qwen2.5:3b,llama3.2:3b

Cada modelo e descarregado ao terminar (keep_alive=0), para nao somar memoria
com o proximo: o servidor tem pouca RAM e nenhum swap por padrao.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request

MODELOS_PADRAO = "qwen2.5:1.5b,qwen2.5:3b,llama3.2:3b,qwen2.5:7b"

# Mesmo espirito do prompt do AnythingLLM em modo "query": so o contexto vale.
SISTEMA = (
    "Você é o assistente virtual de uma prefeitura brasileira. Responda em português do "
    "Brasil, de forma curta e clara, usando SOMENTE as informações do contexto abaixo. "
    "Se a resposta não estiver no contexto, diga exatamente: \"Não encontrei essa "
    "informação na base de conhecimento.\" Não invente telefones, prazos ou leis.\n\n"
    "Contexto:\n{contexto}"
)

# Perguntas reais, escritas com palavras diferentes das da FAQ (parafrase).
# "espera": grupos de alternativas; cada grupo precisa aparecer na resposta.
# "fora": pergunta sem resposta na FAQ, o modelo deve recusar.
PERGUNTAS = [
    {"sec": "procon", "q": "Comprei uma geladeira que veio com defeito. Quanto tempo a loja tem pra consertar?",
     "espera": [["30 dias", "trinta dias"]]},
    {"sec": "procon", "q": "Me arrependi de uma compra que fiz pelo site. Ainda consigo cancelar?",
     "espera": [["7 dias", "sete dias"]]},
    {"sec": "procon", "q": "Qual o prazo pra reclamar de defeito num eletrodoméstico?",
     "espera": [["90 dias", "noventa dias"]]},
    {"sec": "sermulher", "q": "Estou sendo agredida agora, o que eu faço?",
     "espera": [["190"], ["180"]]},
    {"sec": "sermulher", "q": "Dá pra fazer denúncia sem me identificar?",
     "espera": [["anonim"]]},
    {"sec": "integraju", "q": "Como eu entro no IntegrAju, preciso criar senha?",
     "espera": [["gov.br"]]},
    {"sec": "integraju", "q": "Onde eu baixo o aplicativo do IntegrAju?",
     "espera": [["google play", "play store"]]},
    {"sec": "integraju", "q": "Posso pedir conserto de poste apagado pelo IntegrAju?",
     "espera": [["iluminacao"]]},
    {"sec": "fora", "q": "Qual a previsão do tempo para amanhã?", "fora": True},
    {"sec": "fora", "q": "Quem ganhou a Copa do Mundo de 2002?", "fora": True},
]

RECUSA = re.compile(r"nao encontrei|nao (tenho|possuo|ha) (essa |esta )?informac|nao consta|"
                    r"nao esta (no|na) (contexto|base)|fora do (escopo|contexto)")
PALAVRAS_PT = {"de", "que", "para", "voce", "pode", "com", "nao", "uma", "por", "dias", "seu", "sua"}
PALAVRAS_EN = {"the", "and", "you", "your", "can", "with", "for", "days", "this", "is"}


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def post(base: str, path: str, body: dict, timeout: float = 600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.load(r)


def carregar_faq(pasta: str) -> list[str]:
    """Um trecho por par Pergunta/Resposta, como o AnythingLLM indexaria."""
    trechos = []
    for arq in sorted(glob.glob(os.path.join(pasta, "faq-*.md"))):
        texto = open(arq, encoding="utf-8").read()
        for bloco in re.split(r"\n(?=\*\*Pergunta:\*\*)", texto):
            if bloco.startswith("**Pergunta:**"):
                trechos.append(bloco.strip())
    return trechos


def embed(base: str, modelo: str, textos: list[str]) -> list[list[float]]:
    with post(base, "/api/embed", {"model": modelo, "input": textos}) as r:
        return json.load(r)["embeddings"]


def cos(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def memoria_modelo(base: str, modelo: str) -> int:
    alvo = modelo if ":" in modelo else f"{modelo}:latest"   # "bge-m3" aparece como "bge-m3:latest"
    for m in get(base, "/api/ps").get("models", []):
        if alvo in (m.get("name"), m.get("model")):
            return int(m.get("size", 0))
    return 0


def gerar(base: str, modelo: str, sistema: str, pergunta: str, opcoes: dict) -> dict:
    """Chat em streaming: mede o tempo ate o 1o token e o total."""
    t0 = time.time()
    ttft, texto, final = None, [], {}
    body = {"model": modelo, "stream": True, "keep_alive": "30m", "options": opcoes,
            "messages": [{"role": "system", "content": sistema},
                         {"role": "user", "content": pergunta}]}
    with post(base, "/api/chat", body) as r:
        for linha in r:
            if not linha.strip():
                continue
            d = json.loads(linha)
            pedaco = d.get("message", {}).get("content", "")
            if pedaco and ttft is None:
                ttft = time.time() - t0
            texto.append(pedaco)
            if d.get("done"):
                final = d
    total = time.time() - t0
    n = final.get("eval_count", 0)
    dur = final.get("eval_duration", 0) / 1e9
    return {"resposta": "".join(texto).strip(), "ttft": ttft or total, "total": total,
            "tokens": n, "tok_s": (n / dur) if dur else 0.0}


def avaliar(p: dict, resposta: str) -> dict:
    r = norm(resposta)
    recusou = bool(RECUSA.search(r))
    pal = re.findall(r"[a-z]+", r)
    pt = sum(w in PALAVRAS_PT for w in pal)
    en = sum(w in PALAVRAS_EN for w in pal)
    em_pt = pt >= en
    if p.get("fora"):
        ok = recusou
    else:
        grupos = p["espera"]
        acertos = sum(any(norm(alt) in r for alt in g) for g in grupos)
        ok = (acertos == len(grupos)) and not recusou
    return {"ok": ok and em_pt, "recusou": recusou, "pt": em_pt}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ollama", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--modelos", default=MODELOS_PADRAO, help="lista separada por virgula")
    ap.add_argument("--embedder", default=os.getenv("OLLAMA_EMBEDDER", "bge-m3"))
    ap.add_argument("--prefeitura", default=os.getenv("PREFEITURA", "aracaju"))
    ap.add_argument("--num-ctx", type=int, default=2048, help="janela de contexto (menor = menos RAM)")
    ap.add_argument("--num-predict", type=int, default=256, help="limite de tokens da resposta")
    ap.add_argument("--num-thread", type=int, default=0, help="0 = padrao do Ollama")
    ap.add_argument("--top-k", type=int, default=3, help="trechos da FAQ no contexto")
    ap.add_argument("--pull", action="store_true", help="baixa os modelos que faltarem")
    ap.add_argument("--saida", default="benchmark-modelos.md", help="relatorio com todas as respostas")
    a = ap.parse_args()
    base = a.ollama.rstrip("/")
    modelos = [m.strip() for m in a.modelos.split(",") if m.strip()]

    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trechos = carregar_faq(os.path.join(raiz, "sources", a.prefeitura))
    if not trechos:
        sys.exit(f"Nenhuma FAQ em sources/{a.prefeitura}/faq-*.md")

    instalados = {m["name"] for m in get(base, "/api/tags").get("models", [])}
    for m in modelos + [a.embedder]:
        if m in instalados or f"{m}:latest" in instalados:
            continue
        if not a.pull:
            sys.exit(f"Modelo '{m}' nao instalado. Rode com --pull ou: ollama pull {m}")
        print(f"Baixando {m}...", flush=True)
        with post(base, "/api/pull", {"model": m, "stream": False}, timeout=3600) as r:
            r.read()

    print(f"Indexando {len(trechos)} trechos da FAQ com {a.embedder}...", flush=True)
    t0 = time.time()
    vec_trechos = embed(base, a.embedder, trechos)
    vec_perguntas = embed(base, a.embedder, [p["q"] for p in PERGUNTAS])
    print(f"  embeddings em {time.time() - t0:.1f}s; memoria do embedder: "
          f"{memoria_modelo(base, a.embedder) / 2**30:.2f} GB", flush=True)
    contextos = []
    for v in vec_perguntas:
        melhores = sorted(range(len(trechos)), key=lambda i: cos(v, vec_trechos[i]), reverse=True)
        contextos.append("\n\n".join(trechos[i] for i in melhores[:a.top_k]))

    opcoes = {"num_ctx": a.num_ctx, "num_predict": a.num_predict, "temperature": 0.2}
    if a.num_thread:
        opcoes["num_thread"] = a.num_thread

    resumo, detalhes = [], []
    for modelo in modelos:
        print(f"\n== {modelo}", flush=True)
        t0 = time.time()
        try:
            with post(base, "/api/generate", {"model": modelo, "prompt": "", "keep_alive": "30m",
                                               "options": opcoes}) as r:
                r.read()
        except urllib.error.HTTPError as e:
            print(f"  falhou ao carregar: {e} (falta de memoria?)")
            resumo.append({"modelo": modelo, "erro": str(e)})
            continue
        carga = time.time() - t0
        mem = memoria_modelo(base, modelo)
        print(f"  carga {carga:.1f}s, memoria {mem / 2**30:.2f} GB", flush=True)

        res = []
        for p, ctx in zip(PERGUNTAS, contextos):
            g = gerar(base, modelo, SISTEMA.format(contexto=ctx), p["q"], opcoes)
            g.update(avaliar(p, g["resposta"]))
            res.append(g)
            marca = "ok " if g["ok"] else "ERR"
            print(f"  [{marca}] ttft {g['ttft']:5.1f}s  total {g['total']:5.1f}s  "
                  f"{g['tok_s']:4.1f} tok/s  {p['q'][:50]}", flush=True)
            detalhes.append((modelo, p, g))

        ttfts = sorted(r["ttft"] for r in res)
        totais = sorted(r["total"] for r in res)
        resumo.append({
            "modelo": modelo, "carga": carga, "mem_gb": mem / 2**30,
            "ttft_med": ttfts[len(ttfts) // 2], "total_med": totais[len(totais) // 2],
            "total_max": totais[-1], "tok_s": sum(r["tok_s"] for r in res) / len(res),
            "acertos": sum(r["ok"] for r in res), "n": len(res),
            "recusas_ok": sum(r["recusou"] for r, p in zip(res, PERGUNTAS) if p.get("fora")),
            "inventou": sum(not r["recusou"] for r, p in zip(res, PERGUNTAS) if p.get("fora")),
            "fora_pt": sum(not r["pt"] for r in res),
        })
        # descarrega antes do proximo, para nao somar memoria
        with post(base, "/api/generate", {"model": modelo, "keep_alive": 0}) as r:
            r.read()

    cab = ("| Modelo | Memória | Carga | TTFT (mediana) | Total (mediana / pior) | tok/s | "
           "Acertos | Recusou fora da FAQ | Fora do português |\n"
           "|---|---|---|---|---|---|---|---|---|\n")
    linhas = []
    n_fora = sum(1 for p in PERGUNTAS if p.get("fora"))
    for s in resumo:
        if "erro" in s:
            linhas.append(f"| `{s['modelo']}` | falhou: {s['erro']} | | | | | | | |")
            continue
        linhas.append(
            f"| `{s['modelo']}` | {s['mem_gb']:.1f} GB | {s['carga']:.0f} s | {s['ttft_med']:.1f} s | "
            f"{s['total_med']:.0f} s / {s['total_max']:.0f} s | {s['tok_s']:.1f} | "
            f"{s['acertos']}/{s['n']} | {s['recusas_ok']}/{n_fora} | {s['fora_pt']} |")
    tabela = cab + "\n".join(linhas)
    print("\n" + tabela)

    with open(a.saida, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark de modelos — {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Ollama: `{base}` · embedder: `{a.embedder}` · num_ctx={a.num_ctx} · "
                f"num_predict={a.num_predict} · top_k={a.top_k} · CPUs: {os.cpu_count()}\n\n")
        f.write(tabela + "\n\n## Respostas (para revisão humana)\n")
        for modelo, p, g in detalhes:
            f.write(f"\n### {modelo} — {p['q']}\n\n"
                    f"{'✅' if g['ok'] else '❌'} {g['total']:.1f} s · TTFT {g['ttft']:.1f} s\n\n"
                    f"> {g['resposta'].replace(chr(10), chr(10) + '> ')}\n")
    print(f"\nRelatório com todas as respostas: {a.saida}")


if __name__ == "__main__":
    main()
