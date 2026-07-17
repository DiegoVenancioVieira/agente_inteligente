# -*- coding: utf-8 -*-
"""Reindexa e calibra o classificador de intencao (/intent) contra o Ollama.

Precisa de VPN ativa (fala com o bge-m3 na VPS1). Uso:
    python scripts/testa_intent.py

Mede, com frases escritas como o cidadao escreve (nao como o sinonimo esta
cadastrado), qual THRESHOLD separa acerto de fora-de-escopo.
"""
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)

import httpx  # noqa: E402
import main  # noqa: E402

# ------------------------------------------------------------------ casos
# (texto do cidadao, value esperado ou None se deve dar "nao identificado")
CASOS = [
    ("quero parcelar meu iptu", "parcelamento de iptu semfaz financas"),
    ("preciso dividir o iptu em varias vezes", "parcelamento de iptu semfaz financas"),
    ("como faco pra pedir isencao do iptu?", "isencao de iptu semfaz financas"),
    ("segunda via da guia do itbi", "liberacao de guia de itbi semfaz financas"),
    ("quero o alvara pra construir minha casa",
     "alvara de construcao uniresidencial emurb obras e urbanizacao"),
    ("preciso demolir um muro, qual autorizacao?",
     "autorizacao de demolicao emurb obras e urbanizacao"),
    ("certidao de habite-se", "certidao de habite se emurb obras e urbanizacao"),
    ("quando sai meu abono de permanencia", "abono de permanencia"),
    ("quero incluir meu filho como dependente", None),   # ambiguo: seplog x ajuprev
    ("meu contracheque nao chegou", "contracheque seplog planejamento e gestao"),
    ("marcar consulta com oftalmologista", "oftalmo"),
    # fora de escopo: tem que dar nao_identificado
    ("qual o horario do onibus pra praia?", None),
    ("quero fazer um bolo de chocolate", None),
    ("qual a temperatura hoje em aracaju", None),
]


async def main_async():
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        # --- 1. reindexa
        data = json.load(open(main.INTENTS_PATH, encoding="utf-8"))
        pares = [(main._norm_intent(s), e["value"])
                 for e in data for s in e.get("synonyms", []) if s and s.strip()]
        print("indexando %d sinonimos -> %d assuntos..."
              % (len(pares), len({v for _, v in pares})))
        t0 = time.time()
        vecs = await main._embed_batch(client, [s for s, _ in pares])
        print("  embeddings prontos em %.1fs (%.0f/s)"
              % (time.time() - t0, len(pares) / max(time.time() - t0, .001)))

        con = main._db()
        con.execute("DELETE FROM intents")
        con.executemany(
            "INSERT INTO intents (synonym, value, orgao, embedding) VALUES (?,?,?,?)",
            [(s, v, main._orgao_de(v), vec.astype(np.float32).tobytes())
             for (s, v), vec in zip(pares, vecs)])
        con.commit()
        con.close()
        main._load_intents()
        print("  indice em RAM:", main._intent_vecs.shape)

        # --- 2. roda os casos
        print("\n%-42s %-7s %s" % ("TEXTO DO CIDADAO", "SCORE", "ASSUNTO RETORNADO"))
        print("-" * 108)
        acertos, positivos, scores_ok, scores_fora = 0, 0, [], []
        for texto, esperado in CASOS:
            t0 = time.time()
            v = await main._embed(client, main._norm_intent(texto))
            r = main._search_intents(v, top_n=3)
            ms = int((time.time() - t0) * 1000)
            top = r[0] if r else {"score": 0, "intent": "-"}
            if esperado:
                positivos += 1
                bateu = top["intent"] == esperado
                acertos += bateu
                scores_ok.append(top["score"])
                marca = "OK " if bateu else "ERRO"
            else:
                scores_fora.append(top["score"])
                marca = "fora"
            print("%-4s %-38s %.3f  %-50s %dms"
                  % (marca, texto[:38], top["score"], top["intent"][:50], ms))
            if esperado and top["intent"] != esperado:
                print("       esperado: %s" % esperado)
                for c in r[1:]:
                    print("       tambem:   %.3f %s" % (c["score"], c["intent"][:60]))

        # --- 3. calibracao
        print("\n" + "=" * 60)
        print("acertos: %d/%d" % (acertos, positivos))
        if scores_ok and scores_fora:
            print("score dos CERTOS : min=%.3f  media=%.3f"
                  % (min(scores_ok), sum(scores_ok) / len(scores_ok)))
            print("score dos FORA   : max=%.3f  media=%.3f"
                  % (max(scores_fora), sum(scores_fora) / len(scores_fora)))
            folga = min(scores_ok) - max(scores_fora)
            print("folga entre as duas nuvens: %.3f" % folga)
            if folga > 0:
                print("=> THRESHOLD sugerido: %.2f" % ((min(scores_ok) + max(scores_fora)) / 2))
            else:
                print("=> AS NUVENS SE SOBREPOEM: nao ha threshold que separe. "
                      "Precisa de mais sinonimos nos assuntos que erraram.")
        print("THRESHOLD atual no codigo: %.2f | MARGIN: %.2f"
              % (main.INTENT_THRESHOLD, main.INTENT_MARGIN))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main_async())
