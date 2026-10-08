#!/usr/bin/env python3
"""
Simula o custo de responder as perguntas do agente por uma API de LLM.

Roda as perguntas reais do benchmark (scripts/benchmark_modelos.py) no mesmo
formato do agente (instrucao + trechos da FAQ + pergunta) contra a API do
Claude, mede os tokens REAIS de entrada e saida de cada resposta e projeta o
gasto mensal:

  - no proprio Claude (modelo escolhido em --modelo);
  - numa API do Qwen, aplicando os mesmos volumes de tokens aos precos do Qwen
    (o tokenizador do Qwen conta diferente: ajuste com --fator-tokens-qwen).

A busca dos trechos aqui e lexical (sem Ollama), so para montar um contexto do
tamanho que o agente envia; a qualidade da busca nao entra no custo.

Chave da API (nunca no codigo nem no git), na ordem:
  1. variavel de ambiente SIMULACAO_ANTHROPIC_API_KEY (nome proprio, para nao ser
     confundida com a credencial de outras ferramentas da maquina);
  2. arquivo indicado em --chave-arquivo (padrao: tokens/api_max_claude.txt);
  3. variavel de ambiente ANTHROPIC_API_KEY.

Uso:
  pip install anthropic
  python3 scripts/simula_custos.py
  python3 scripts/simula_custos.py --modelo claude-haiku-5-5 --perguntas-dia 50,200,1000
"""
import argparse
import os
import re
import statistics
import sys
import time

import anthropic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_modelos import PERGUNTAS, SISTEMA, avaliar, carregar_faq, norm  # noqa: E402

# US$ por 1 milhao de tokens (entrada, saida). Claude: tabela oficial da Anthropic.
PRECOS_CLAUDE = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
# Qwen (Alibaba Cloud Model Studio, endpoint Internacional). A CONFERIR na pagina
# oficial antes de contratar: os valores mudam com a versao do modelo e promocoes.
PRECOS_QWEN = {
    "qwen-flash": (0.15, 0.47),
    "qwen-plus": (0.40, 1.60),
}
# Modelos que aceitam a forma "default" do fallback server-side de recusas.
COM_FALLBACK = {"claude-opus-5-5", "claude-sonnet-5-5"}


def ler_chave(caminho: str) -> str | None:
    chave = os.getenv("SIMULACAO_ANTHROPIC_API_KEY", "").strip()
    if not chave and os.path.exists(caminho):
        chave = open(caminho, encoding="utf-8").read().strip()
    return chave or os.getenv("ANTHROPIC_API_KEY", "").strip() or None


def buscar_lexical(pergunta: str, trechos: list[str], k: int) -> str:
    """Top-k trechos por palavras em comum (stand-in do embedding, so para o tamanho)."""
    palavras = {w for w in re.findall(r"[a-z0-9]+", norm(pergunta)) if len(w) > 3}
    nota = [(sum(w in norm(t) for w in palavras), i) for i, t in enumerate(trechos)]
    melhores = [i for _, i in sorted(nota, reverse=True)[:k]]
    return "\n\n".join(trechos[i] for i in melhores)


def custo(tok_in: float, tok_out: float, preco: tuple[float, float]) -> float:
    return tok_in / 1e6 * preco[0] + tok_out / 1e6 * preco[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="claude-opus-5-5", choices=sorted(PRECOS_CLAUDE))
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"],
                    help="profundidade de raciocinio; low basta para FAQ e gasta menos")
    ap.add_argument("--chave-arquivo", default="tokens/api_max_claude.txt")
    ap.add_argument("--base-url", default="https://api.anthropic.com")
    ap.add_argument("--prefeitura", default=os.getenv("PREFEITURA", "aracaju"))
    ap.add_argument("--top-k", type=int, default=3, help="trechos da FAQ no contexto")
    ap.add_argument("--perguntas-dia", default="50,200,1000", help="cenarios de volume")
    ap.add_argument("--taxa-cache", type=float, default=0.3,
                    help="fracao respondida pelo cache semantico do agente (sem LLM)")
    ap.add_argument("--fator-tokens-qwen", type=float, default=1.0,
                    help="tokens no Qwen / tokens no Claude para o mesmo texto")
    ap.add_argument("--cambio", type=float, default=5.50, help="R$ por US$ (ajuste ao dia)")
    ap.add_argument("--saida", default="simulacao-custos.md")
    a = ap.parse_args()

    chave = ler_chave(a.chave_arquivo)
    if not chave:
        sys.exit("Sem chave: defina SIMULACAO_ANTHROPIC_API_KEY ou crie o arquivo "
                 f"{a.chave_arquivo} (fora do git).")
    client = anthropic.Anthropic(api_key=chave, base_url=a.base_url)

    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trechos = carregar_faq(os.path.join(raiz, "sources", a.prefeitura))
    if not trechos:
        sys.exit(f"Nenhuma FAQ em sources/{a.prefeitura}/faq-*.md")

    extra = {}
    if a.modelo in COM_FALLBACK:
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

    linhas, medidas = [], []
    for p in PERGUNTAS:
        sistema = SISTEMA.format(contexto=buscar_lexical(p["q"], trechos, a.top_k))
        t0 = time.time()
        try:
            r = client.beta.messages.create(
                model=a.modelo, max_tokens=2048, system=sistema,
                messages=[{"role": "user", "content": p["q"]}],
                output_config={"effort": a.effort}, **extra,
            )
        except anthropic.AuthenticationError:
            sys.exit("Chave recusada pela API (401). Confira a chave.")
        except anthropic.RateLimitError:
            sys.exit("Limite de requisicoes da conta atingido (429). Tente mais tarde.")
        except anthropic.APIStatusError as e:
            sys.exit(f"Erro da API ({e.status_code}): {e.message}")
        seg = time.time() - t0
        texto = "".join(b.text for b in r.content if b.type == "text").strip()
        if r.stop_reason == "refusal":
            texto = texto or "(recusada pelo modelo)"
        u = r.usage
        tok_in = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
        tok_out = u.output_tokens          # inclui o raciocinio (cobrado como saida)
        av = avaliar(p, texto)
        medidas.append((tok_in, tok_out, seg))
        linhas.append((p, texto, tok_in, tok_out, seg, av["ok"]))
        print(f"[{'ok ' if av['ok'] else 'ERR'}] in {tok_in:5d}  out {tok_out:5d}  "
              f"{seg:4.1f}s  {p['q'][:55]}", flush=True)

    m_in = statistics.mean(m[0] for m in medidas)
    m_out = statistics.mean(m[1] for m in medidas)
    m_seg = statistics.mean(m[2] for m in medidas)
    acertos = sum(l[5] for l in linhas)

    modelos = [(a.modelo, PRECOS_CLAUDE[a.modelo], 1.0)] + \
              [(nome, preco, a.fator_tokens_qwen) for nome, preco in PRECOS_QWEN.items()]
    volumes = [int(v) for v in a.perguntas_dia.split(",") if v.strip()]
    cab = "| Modelo | US$ / 1.000 respostas | " + " | ".join(
        f"{v}/dia → R$/mês" for v in volumes) + " |\n|---|---|" + "---|" * len(volumes) + "\n"
    corpo = []
    for nome, preco, fator in modelos:
        por_resp = custo(m_in * fator, m_out * fator, preco)
        meses = [por_resp * v * 30 * (1 - a.taxa_cache) * a.cambio for v in volumes]
        corpo.append(f"| `{nome}` | {por_resp * 1000:.2f} | " +
                     " | ".join(f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                                for x in meses) + " |")
    tabela = cab + "\n".join(corpo)

    resumo = (f"Medido com `{a.modelo}` (effort `{a.effort}`): média de **{m_in:.0f} tokens de "
              f"entrada** e **{m_out:.0f} de saída** por resposta, {m_seg:.1f} s em média; "
              f"{acertos}/{len(linhas)} respostas corretas na checagem automática.\n\n"
              f"Projeção com {a.taxa_cache:.0%} das perguntas respondidas pelo cache do agente "
              f"(sem LLM), 30 dias e câmbio de R$ {a.cambio:.2f}/US$. Preços do Qwen a conferir; "
              f"tokens do Qwen = tokens do Claude × {a.fator_tokens_qwen}.")
    print("\n" + resumo + "\n\n" + tabela)

    with open(a.saida, "w", encoding="utf-8") as f:
        f.write(f"# Simulação de custos — {time.strftime('%Y-%m-%d %H:%M')}\n\n{resumo}\n\n{tabela}\n\n"
                "## Respostas\n")
        for p, texto, ti, to, seg, ok in linhas:
            f.write(f"\n### {p['q']}\n\n{'✅' if ok else '❌'} {ti} tokens de entrada · {to} de saída · "
                    f"{seg:.1f} s\n\n> {texto.replace(chr(10), chr(10) + '> ')}\n")
    print(f"\nRelatório: {a.saida}")


if __name__ == "__main__":
    main()
