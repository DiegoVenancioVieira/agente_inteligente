# API de Intenção — substituição do Dialogflow

Classifica texto livre do cidadão em um **assunto do 1doc** (a string `value` do
`json_normalizado.json`), para o sistema chamador localizar o formulário e montar o card.

## Diferença em relação ao Dialogflow

Não usa LLM. É embedding (`bge-m3`) + similaridade de cosseno contra os 1.888 sinônimos
cadastrados. Consequências práticas:

- **Não inventa assunto.** O retorno sai sempre da lista do JSON — não existe alucinação.
- **~150 ms**, sem entrar na fila do modelo de linguagem (que serializa na CPU).
- **Entende paráfrase.** O Dialogflow casava o texto contra os sinônimos literais; o
  embedding aproxima "preciso dividir o IPTU em vezes" de "parcelar iptu" sem que
  ninguém cadastre essa frase.
- **Trata ambiguidade explicitamente**, em vez de escolher um assunto no chute.

## `POST /intent`

```jsonc
// requisição
{
  "text": "quero parcelar meu iptu",   // obrigatório, 1..500 chars
  "orgao": "semfaz",                   // opcional — ver "Ambiguidade" abaixo
  "top_n": 3                           // opcional, 1..10 (default 3)
}
```

```jsonc
// resposta — status "ok"
{
  "status": "ok",
  "intent": "parcelamento de iptu semfaz financas",  // <- a string do 1doc
  "score": 0.91,
  "matched_synonym": "parcelar meu iptu",
  "orgao": "semfaz",
  "candidates": [ /* 2º, 3º... */ ],
  "took_ms": 148
}
```

### Os três status — e o que fazer com cada um

| status | significado | o que o chamador faz |
|---|---|---|
| `ok` | um assunto claramente à frente | usa `intent` e abre o formulário |
| `ambiguo` | os melhores empataram; o texto não distingue | mostra `candidates` como card de escolha |
| `nao_identificado` | nada passou do corte | fallback (busca livre, atendente, menu) |

Em `ambiguo` e `nao_identificado`, `intent` vem `null` — nunca chute o primeiro
candidato, é justamente o caso em que ele não é confiável.

## Ambiguidade — leia antes de integrar

O JSON tem **88 sinônimos que apontam para mais de um assunto**. Isso é um teto:
nenhum modelo resolve, porque a informação não está no texto. Os piores:

| sinônimo | nº de assuntos |
|---|---|
| `nota fiscal` | 23 (um por secretaria) |
| `mandado judicial` | 20 |

**O parâmetro `orgao` resolve esses dois casos**, que são os maiores. Se o sistema
chamador já sabe de qual secretaria é o atendimento, mande sempre — ele restringe a
busca e o empate desaparece. Siglas válidas em `GET /intent/orgaos`.

### O que o `orgao` não resolve

Sobram **8 colisões dentro do mesmo órgão**, que são distinções de negócio reais:

- `parcelamento de iptu` × `reparcelamento de iptu` (SEMFAZ)
- `parcelamento de tlf ou iss` × `reparcelamento de iss` (SEMFAZ)
- `parcelamento de tlf` × `reparcelamento de tlf` (SEMFAZ)
- `isencao de iptu` × `isencao de iptu servidor` (SEMFAZ)
- `isencao de itbi` × `deferimento parcial da isencao do itbi` (SEMFAZ)
- `autorizacao de demolicao` × `certidao de demolicao` (EMURB)
- `autorizacao para obras de infraestrutura` — ligações de água/esgoto × obras em geral (EMURB)
- `nota fiscal sms` × `nota fiscal sms saude` (SMS) — **isto é duplicata de cadastro,
  não distinção de negócio; vale corrigir no 1doc**

"Quero parcelar meu IPTU" é genuinamente indistinguível entre parcelamento e
reparcelamento: depende de o cidadão já ter um acordo ativo. Só há duas saídas —
card de desambiguação, ou o chamador decide por regra de negócio (ex.: consulta se
existe parcelamento em vigor). O Dialogflow não resolvia isso; escolhia um dos dois.

## `GET /intent/orgaos`

Lista as siglas aceitas em `orgao`.

## `POST /intent/reindex` (admin)

Reconstrói o índice a partir de `sources/intents-1doc.json`. Exige
`Authorization: Bearer <ADMIN_TOKEN>`. Rode **quando a lista de assuntos do 1doc mudar** —
é o passo lento (um embedding por sinônimo).

## Configuração (`.env`)

| variável | default | o que faz |
|---|---|---|
| `INTENTS_PATH` | `./sources/intents-1doc.json` | fonte dos assuntos |
| `INTENT_THRESHOLD` | `0.70` | abaixo disso → `nao_identificado` |
| `INTENT_MARGIN` | `0.04` | diferença 1º–2º abaixo disso → `ambiguo` |
| `INTENT_RATE_LIMIT_PER_MIN` | `600` | limite do `/intent` |

### Calibração (medida em 17/07/2026, `bge-m3` real)

14 frases dentro do escopo × 8 fora, escritas como o cidadão escreve:

| | mínimo | média | máximo |
|---|---|---|---|
| dentro do escopo | 0,669 | **0,835** | 1,000 |
| fora do escopo | 0,407 | **0,536** | 0,681 |

Corte em **0,70**: recusa todas as 8 de fora e perde 1 das 14 de dentro — e essa uma
("como solicito o habite-se da minha obra") vinha com o assunto **errado**, então
recusar era o comportamento certo. O custo é assimétrico: **abrir o formulário errado
é pior do que não abrir nenhum**, então erra-se para o lado de recusar.

`MARGIN` 0,04 pega as 7 colisões reais (todas com margem 0,00–0,03) sem estragar
consulta limpa (margem mediana 0,12).

Amostra pequena (22 frases). Vale refazer com log de perguntas reais depois que a
integração estiver rodando: `python scripts/testa_intent.py`.

### Duas fragilidades conhecidas do índice

- **`tlf` engana o modelo.** "Preciso comprar um celular novo" pontua 0,681 em
  `reparcelamento de tlf` — o `bge-m3` lê `tlf` como "telefone", mas no cadastro é
  *Taxa de Licença e Funcionamento*. Ficou abaixo do corte, mas por pouco.
- **Verbo genérico domina.** "Como **solicito** o habite-se" foi parar em
  `praca da juventude **solicitar** utilizacao`. `certidao de habite-se` sozinho acerta
  com 0,912. Sinônimos curtos e cheios de verbo genérico atraem consulta errada.

Ambas se corrigem com sinônimos melhores no cadastro — os atuais estão em jargão de
servidor ("uniresidencial"), não em língua de cidadão ("minha casa").

O `/ask` limita 20 req/min **por IP**, o que derrubaria uma integração
servidor-a-servidor (todas as chamadas vêm do mesmo IP). Por isso o `/intent` tem
limite próprio e alto. Se a API for exposta fora da rede interna, troque o limite por
IP por autenticação com chave por cliente.
