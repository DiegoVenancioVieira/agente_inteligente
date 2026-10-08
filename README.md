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
      resposta cacheada         AnythingLLM + LLM (Ollama)
        (~0,2 s, sem LLM)         (tempo depende do modelo) ─▶ grava no cache
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
Atrás de proxy reverso, defina `TRUSTED_PROXIES` com as redes (CIDR) dos proxies confiáveis — Traefik
do Coolify e o container do hub, normalmente as redes Docker, ex.: `172.16.0.0/12,10.0.0.0/8`. O
`X-Forwarded-For` só é lido quando a conexão vem de uma dessas redes; quem chama a porta publicada
direto é limitado pelo próprio IP de origem (vazio = comportamento antigo, forjável).
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
3. O container escuta em `8000` internamente, publicado no host em **`8100`** (a `8000`
   do host é do Coolify). O `cache.db` persiste no volume `faq_cache_data`.
4. Aponte seu front-end / integração para `http://VPS2:8100/ask`.

> **Rede:** o serviço precisa alcançar o Ollama da VPS1 (`OLLAMA_URL`) para os embeddings
> e o AnythingLLM (`ANYTHINGLLM_URL`). Rodando na VPS2, ambos são acessíveis.

## Configuração por prefeitura

Um único código atende várias prefeituras — **um deploy por prefeitura**. Os dados do município
ficam em `config/prefeituras/<slug>.json`, escolhido pela env **`PREFEITURA`** (padrão `aracaju`):

| Campo | O que controla |
| --- | --- |
| `prefeitura.nome` / `prefeitura.nomeOficial` | Nome exibido nas páginas (título, rodapés) |
| `marca.logo` / `marca.logoAlt` | Brasão das páginas (`/static/<arquivo>` em `web/` ou URL absoluta) |
| `secretarias` | `{slug: {label, welcome, chips}}` — cada slug é um workspace do AnythingLLM |
| `orgaos` | `{sigla: nome}` usados pela API de intenção para desambiguar assuntos |
| `intentsSeed` | Base de assuntos da API de intenção, ex.: `sources/<slug>/intents-1doc.json` |

A config é validada no boot: `PREFEITURA` desconhecida ou campo obrigatório vazio impedem o
serviço de subir, com a mensagem do problema. `ANYTHINGLLM_URL` e `OLLAMA_URL` não têm mais
endereço padrão no código: defina-as no `.env`.

Para uma nova prefeitura: copie `config/prefeituras/aracaju.json`, ajuste os dados, coloque as
FAQs/intents em `sources/<slug>/`, crie os workspaces no AnythingLLM e suba um deploy com
`PREFEITURA=<slug>`. O conteúdo de `relatorio*.html` e `tutorial.html` descreve a implantação de
Aracaju; só nome e brasão vêm da config.

## Modelo de geração e hardware

O agente **não escolhe o modelo de geração**: quem chama o LLM é o AnythingLLM. O agente só usa o
Ollama diretamente para os embeddings (`OLLAMA_EMBEDDER`, padrão `bge-m3`). Para trocar o modelo:

- no AnythingLLM, em *Settings → LLM* (padrão da instância) ou no *Chat model* de cada workspace;
- ou por env no container do AnythingLLM: `LLM_PROVIDER=ollama`, `OLLAMA_BASE_PATH`,
  `OLLAMA_MODEL_PREF=<modelo>` e `OLLAMA_MODEL_TOKEN_LIMIT=2048` (confira os nomes na versão instalada).

Baixe o modelo antes no Ollama (`ollama pull qwen2.5:3b`). O cache semântico continua valendo:
respostas já dadas não passam pelo LLM.

### Requisitos por modelo (CPU, sem GPU, quantização Q4)

Valores **estimados** para 2 vCPUs, não medidos — use o benchmark abaixo para os números reais.
Em CPU, o tempo até o 1º token é dominado pela leitura do contexto (prompt + trechos da FAQ), por
isso `num_ctx` e o número de trechos pesam tanto quanto o tamanho do modelo.

| Modelo | RAM do modelo | Geração (2 vCPUs) | Qualidade em português | Uso |
| --- | --- | --- | --- | --- |
| `qwen2.5:1.5b` | ~1,5 GB | ~10–15 tok/s | aceitável; mais propenso a errar detalhes | plano B se o 3b ficar lento |
| `qwen2.5:3b` | ~2,5–3 GB | ~5–8 tok/s | boa, segue bem o contexto | **recomendado para este servidor** |
| `llama3.2:3b` | ~2,5–3 GB | ~5–8 tok/s | boa, às vezes mistura inglês | alternativa ao qwen2.5:3b |
| `qwen2.5:7b` | ~5,5–6 GB | ~2–4 tok/s | a melhor | exige ≥ 4 vCPUs e ≥ 12 GB de RAM; não cabe aqui |
| `bge-m3` (embedder) | ~1,2 GB | — | — | sempre carregado junto |

**Servidor atual (2 vCPUs, 7,8 GB, sem GPU, ~2,3 GB já em uso):** `qwen2.5:3b` + `bge-m3` somam
~4 GB, ficando ~1,5 GB de folga. O `qwen2.5:7b` + `bge-m3` (~7 GB) não cabe com segurança: sem
swap, o kernel mata o Ollama quando a memória acaba. Decida entre `qwen2.5:3b`, `llama3.2:3b` e
`qwen2.5:1.5b` pelo benchmark.

### Ajustes para CPU

No container do **Ollama**:

```bash
OLLAMA_KEEP_ALIVE=24h         # não descarrega o modelo entre perguntas (recarregar custa segundos)
OLLAMA_NUM_PARALLEL=1         # uma geração por vez: em 2 vCPUs, paralelo só divide a CPU
OLLAMA_MAX_LOADED_MODELS=2    # LLM + embedder, sem carregar um terceiro
```

No **AnythingLLM**: limite de contexto (`OLLAMA_MODEL_TOKEN_LIMIT`) em **2048** e poucos trechos por
resposta (3–4 em *Max context snippets* do workspace). As FAQs têm trechos curtos; mais contexto só
aumenta o tempo até o 1º token.

**Embedder:** mantenha o `bge-m3`. O limiar do cache (0,85) e o da API de intenção (0,70) foram
calibrados com ele. Trocar por um menor (ex.: `paraphrase-multilingual`, ~0,6 GB) exige recalibrar
os limiares, reindexar a API de intenção (`POST /intent/reindex`), limpar o cache
(`POST /cache/clear`) e re-embedar os documentos dos workspaces no AnythingLLM.

### Swap (obrigatório neste servidor)

Sem swap, um pico de memória derruba o Ollama. Crie 4 GB no host:

```bash
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab   # mantém após reiniciar
free -h                                                        # deve mostrar Swap: 4.0Gi
```

O swap é rede de segurança, não memória de trabalho: se o modelo passar a usar swap, a geração fica
dezenas de vezes mais lenta. Nesse caso, desça para um modelo menor.

### Benchmark no servidor

`scripts/benchmark_modelos.py` reproduz o fluxo do agente com perguntas reais da FAQ (parafraseadas)
e duas perguntas fora da FAQ, e mede carga, tempo até o 1º token, tempo total, tokens/s, memória
(`ollama ps`) e acertos. Só usa a biblioteca padrão do Python. Cada modelo é descarregado ao terminar,
para não somar memória com o próximo. Rode em horário de pouco uso, porque o benchmark disputa CPU
com o atendimento:

```bash
# Ollama com porta no host:
python3 scripts/benchmark_modelos.py --ollama http://localhost:11434 --pull

# Ollama só na rede Docker do Coolify (troque <rede> e <ollama>):
docker run --rm --network <rede> -v "$PWD":/w -w /w python:3.12-slim \
  python scripts/benchmark_modelos.py --ollama http://<ollama>:11434 --pull

# Só alguns modelos, ou testar o 7b à parte (cuidado com a memória):
python3 scripts/benchmark_modelos.py --modelos qwen2.5:3b,llama3.2:3b
```

Saída: uma tabela no terminal e `benchmark-modelos.md` com todas as respostas, para conferir à mão
se o modelo respondeu certo, sem inventar e em português. A checagem automática procura os fatos
esperados (prazos, telefones etc.) e a recusa nas perguntas fora da FAQ.

## Chat no hub qrcode

O hub de QR Codes da prefeitura (repo `qrcode`) tem um widget de chat que chama este serviço
pela rota interna `/api/chat` do Next.js, com seletor de secretaria (`GET /secretarias`).
O hub repassa o IP do cidadão no `X-Forwarded-For`. Para isso funcionar:

- aponte `AGENTE_API_URL` do hub para o endereço **interno** deste container
  (ex.: `http://faq-cache:8000` na rede do Coolify), sem passar pelo Traefik público;
- defina aqui `TRUSTED_PROXIES` cobrindo as redes Docker do Traefik e do hub
  (ex.: `172.16.0.0/12,10.0.0.0/8`): vale tanto para o hub quanto para cidadãos que chegam
  pelo Traefik.

Sem isso, todos os cidadãos que vêm pelo hub dividem o mesmo limite de `RATE_LIMIT_PER_MIN`.

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
