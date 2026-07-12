# Invalidação automática do cache via n8n

**Problema:** quando uma secretaria troca um PDF da FAQ no AnythingLLM, as respostas já
guardadas no cache semântico podem ficar **desatualizadas**. Precisamos limpar o cache
sempre que o conteúdo mudar.

O AnythingLLM **não emite um evento** ao atualizar documentos, então usamos o n8n para
**detectar a mudança** e chamar o endpoint de limpeza.

## Importação rápida

Já deixei o fluxo pronto em [`n8n-workflow-invalidacao.json`](n8n-workflow-invalidacao.json).
No n8n: **Workflows → ⋯ → Import from File** e selecione o arquivo. Depois **edite dois valores**:
- No nó *Ler documentos*: troque `TROQUE_PELA_API_KEY_DO_ANYTHINGLLM` pela sua API key.
- No nó *Limpar cache*: troque `TROQUE_PELO_ADMIN_TOKEN` pelo seu `ADMIN_TOKEN`.

Confira também as URLs (`192.168.0.118:3001` e `:8000`) e ative o workflow.
Se algum nó reclamar de versão (o n8n evolui), recrie-o seguindo o passo a passo abaixo.

## Estratégia recomendada: polling com detecção de mudança

O n8n verifica periodicamente a lista de documentos do AnythingLLM. Se ela mudou desde a
última checagem, limpa o cache.

### Fluxo (5 nós)

```
[Schedule Trigger]  a cada 10 min
        │
        ▼
[HTTP Request]  GET  {ANYTHINGLLM}/api/v1/documents
   Header: Authorization: Bearer {API_KEY_ANYTHINGLLM}
        │
        ▼
[Code]  extrai uma "assinatura" da FAQ:
   nomes dos arquivos + datas de publicação (campo "published")
   -> ex.: hash/concatenação de {name, published} de cada doc
        │
        ▼
[IF]  a assinatura mudou em relação à guardada?
   (usar Static Data do workflow: $getWorkflowStaticData('global'))
        │ (sim)
        ▼
[HTTP Request]  POST  {SERVICO}/cache/clear
   Header: Authorization: Bearer {ADMIN_TOKEN}
```

### Detalhes dos nós

**1. Schedule Trigger** — intervalo de 10 minutos (ajuste conforme a frequência de updates).

**2. HTTP Request (ler documentos)**
- Método: `GET`
- URL: `http://VPS2:3001/api/v1/documents`
- Header: `Authorization` = `Bearer <API_KEY_DO_ANYTHINGLLM>`

**3. Code (calcular assinatura)** — exemplo em JavaScript:
```js
// junta nome + data de publicação de cada documento em uma string estável
function walk(node, out) {
  for (const it of (node.items || [])) {
    if (it.type === 'folder') walk(it, out);
    else out.push(`${it.name}|${it.published}`);
  }
}
const out = [];
walk($json.localFiles, out);
const assinatura = out.sort().join('#');

const store = $getWorkflowStaticData('global');
const mudou = store.assinatura !== assinatura;
store.assinatura = assinatura;      // guarda para a próxima execução
return [{ json: { mudou, assinatura } }];
```

**4. IF** — condição: `{{ $json.mudou }}` é `true`.

**5. HTTP Request (limpar cache)** — só no ramo "true":
- Método: `POST`
- URL: `http://VPS2:8000/cache/clear`
- Header: `Authorization` = `Bearer <ADMIN_TOKEN>`

## Alternativa simples: botão manual

Se preferir não fazer polling, crie um workflow com **Webhook Trigger** (ou um formulário
do n8n) que a secretaria aciona **após** subir um PDF. O único nó é o HTTP Request do passo 5.
Menos automático, porém trivial e à prova de erros.

## Teste manual do endpoint

```bash
curl -X POST http://VPS2:8000/cache/clear \
  -H "Authorization: Bearer <ADMIN_TOKEN>"
# -> {"status":"cache limpo"}
```

> Sem o token (ou com token errado) o endpoint responde `401` — por isso o n8n precisa
> enviar o `ADMIN_TOKEN` correto no header.
