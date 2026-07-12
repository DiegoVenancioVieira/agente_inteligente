# Agente de FAQ Municipal — API + Cache Semântico

Microserviço que fica **na frente** do AnythingLLM e serve as perguntas dos municípes.
É ao mesmo tempo a **API** de integração (site, WhatsApp, etc.) e a camada de **cache
semântico** que alivia a CPU da VPS1.

## Como funciona

```
Cidadão ─▶ POST /ask ─▶ [1] embedding bge-m3 (VPS1)
                        [2] busca no cache (cosseno, em RAM)
                              │
                  ┌───────────┴───────────┐
          sim ≥ 0,85                sim < 0,85
          (repetida)                 (nova)
              │                          │
      resposta cacheada         AnythingLLM + qwen2.5:7b (VPS1)
        (~0,2 s, sem LLM)         (~5–20 s)  ─▶ grava no cache
```

- **Só respostas fundamentadas são cacheadas.** Se o AnythingLLM recusa (pergunta fora
  de escopo, 0 fontes), a resposta **não** entra no cache — ela já é rápida (~1 s) e a
  FAQ pode mudar.
- **Thread nova por pergunta** ao chamar o AnythingLLM, evitando contaminação de histórico.

## Por que o threshold é 0,85 (calibrado com dados reais)

| Tipo de pergunta | Similaridade bge-m3 | Decisão |
|---|---|---|
| Quase-idêntica (caixa/acento/palavra a mais) | 0,89 – 1,00 | ✅ acerta o cache |
| Paráfrase solta (vocabulário diferente) | 0,60 – 0,71 | ❌ vai ao LLM (seguro) |
| Pergunta diferente | 0,36 – 0,45 | ❌ rejeita |

O corte 0,85 pega os repetidos genuínos e **nunca** devolve a resposta de outra pergunta
— numa prefeitura, errar para o lado de consultar o LLM é o comportamento correto.

## Endpoints

| Método | Rota | Auth | Uso |
|---|---|---|---|
| `POST` | `/ask` | pública* | `{"question": "..."}` → `{answer, cached, similarity, latency_ms}` |
| `GET` | `/` | pública | Interface de chat dos municípes |
| `GET` | `/health` | pública | status, tamanho do cache, config |
| `GET` | `/stats` | pública | hits, misses, taxa de acerto |
| `GET` | `/unanswered` | **admin** | perguntas sem resposta, agrupadas por frequência |
| `POST` | `/cache/clear` | **admin** | limpa o cache (webhook de invalidação) |

\* `/ask` tem **rate-limit por IP** (`RATE_LIMIT_PER_MIN`, padrão 20/min) → responde `429` ao exceder.
Endpoints **admin** exigem o header `Authorization: Bearer <ADMIN_TOKEN>` (defina um token forte no `.env`;
se vazio, os endpoints admin ficam bloqueados por padrão).

Exemplo:
```bash
curl -X POST http://SEU_HOST:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Qual a alíquota do ISSQN pela Lei 183?"}'

# Ver o que os cidadãos perguntam e o agente não responde (para as secretarias):
curl http://SEU_HOST:8000/unanswered -H "Authorization: Bearer $ADMIN_TOKEN"
```

## Deploy na VPS2 (Coolify)

1. Suba esta pasta como um repositório e conecte no Coolify (ou use `docker compose up -d`).
2. Garanta que o `.env` esteja presente (contém `ANYTHINGLLM_API_KEY`, URLs, threshold).
   O `.env` **não** vai para o Git (protegido pelo `.gitignore`).
3. O serviço sobe na porta `8000`. O `cache.db` persiste no volume `faq_cache_data`.
4. Aponte seu front-end / integração para `http://VPS2:8000/ask`.

> **Rede:** o serviço precisa alcançar o Ollama da VPS1 (`OLLAMA_URL`) para os embeddings
> e o AnythingLLM (`ANYTHINGLLM_URL`). Rodando na VPS2, ambos são acessíveis.

## Perguntas sem resposta (insumo para as secretarias)

Toda vez que o agente **recusa** uma pergunta (não há resposta na FAQ), ela é registrada.
Consulte em `GET /unanswered` (admin) — vêm agrupadas por frequência, mostrando o que os
cidadãos mais perguntam e a FAQ ainda não cobre. As secretarias usam isso para decidir o
que acrescentar aos PDFs.

## Invalidação do cache

Quando uma secretaria trocar o PDF da FAQ, as respostas antigas podem ficar desatualizadas.
Chame `POST /cache/clear` (admin) ou espere o TTL expirar. Para automatizar via **n8n**
(detectar a troca de PDF e limpar o cache sozinho), veja
[docs/n8n-invalidacao-cache.md](docs/n8n-invalidacao-cache.md).

## Escala / multi-secretaria

Esta instância atende **um** workspace (`WORKSPACE_SLUG`). Para várias secretarias, suba
uma instância por workspace (portas diferentes) ou estenda o cache para indexar por
workspace. Simples e isolado — cada secretaria com seu cache.
