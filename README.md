# Fluent Sphere — Free AI Model Aggregator

*By Rudi Tabuti ([@bahjoe5](https://x.com/bahjoe5))*

One server (Python 3.13, standard library only) that speaks the OpenAI Chat
Completions API and the Anthropic Messages API, and routes every request to
the best free model across ten simulated free-tier providers — with failover
when a provider refuses.

Single file: `main.py` (1,039 lines, no dependencies). MIT licensed.

> Status: submitted to the make.fun `free-model-aggregator-01` bounty on
> 15 November 2026 as entry
> ["Fluent Sphere"](https://make.fun/s/01a1dcac-e6f3-76dd-9912-5543146155de).
> Practice trials (trial-0/1/2) all score 99/100. Final result due
> 30 November 2026.

## What it does

- **Two APIs, one server.** `POST /v1/chat/completions` (OpenAI-compatible,
  JSON and SSE streaming) and `POST /v1/messages` (Anthropic-compatible,
  JSON and SSE streaming). `GET /v1/models` lists the free models.
- **Ten providers.** groq, cerebras, gemini, zai, openrouter, sambanova,
  pollinations, cloudflare, ovh, kilol. At startup the server probes every
  provider's catalog route (`GET <base>/v1/models`, Gemini
  `GET <base>/v1beta/models`) and builds the exact catalog it registers,
  falling back to the ratings table for any provider whose probe fails.
- **Best free model first.** Free models are ranked by rating (ties broken
  by provider id, then model id), filtered by the request's requirements:
  tools, vision, structured JSON, minimum context. A JSON-mode OpenAI
  request only goes to an OpenAI or Gemini wire.
- **Failover.** Candidates are tried in ranked order; the first `2xx`
  answer wins. A `429` is retried once on the same provider. A `402`
  (budget exhausted) marks the provider dead for the rest of the trial, as
  do hard rejections (`400`/`401`/`5xx`). A network hiccup (no response)
  does *not* mark the provider dead — it is not a verdict on the provider.
  One request makes exactly one successful provider call.
- **Lossless translation.** System text, the last user message's text, tool
  names in order, `max_tokens`, stop sequences in order, image bytes, and
  the JSON-mode flag are all carried across. The provider's answer text and
  usage counts are relayed verbatim; the stop reason is mapped back to the
  client's stop family.
- **Deterministic.** Routing uses no wall clock and no unseeded randomness:
  the same request against the same catalog always takes the same route.

## How to run it

This server was built to run inside the make.fun grading harness. It needs
the trial brief (provider directory, quality ratings table, workload
manifest) and a control-plane URL:

```sh
MAKEFUN_PROMPT_FILE=<brief file> MAKEFUN_ENV_URL=<control-plane URL> python3 main.py
```

It picks an available port, registers it with the environment
(`POST $MAKEFUN_ENV_URL/register`), serves requests, polls
`GET $MAKEFUN_ENV_URL/work`, and exits 0 when the workload is done.

Print the catalog without serving:

```sh
MAKEFUN_PROMPT_FILE=<brief file> python3 main.py --list
```

`--list` prints the registered catalog as JSON, byte-for-byte.

Requirements: Python 3.13, standard library only — nothing to install.
There is no standalone mode yet: without the harness brief and
control-plane URL the server will not start.

## Endpoints

| Method & path | Auth | Notes |
|---|---|---|
| `GET /v1/models` | `Authorization: Bearer …` or `x-api-key: …` | OpenAI shape under Bearer, Anthropic shape under `x-api-key` |
| `POST /v1/chat/completions` | — | OpenAI-compatible; `"stream": true` returns SSE |
| `POST /v1/messages` | — | Anthropic-compatible; `"stream": true` returns SSE |

Any other route returns `404 {"error": {"type": "not_found"}}`. A request
no provider can serve returns `503 {"error": {"type": "server_error",
"message": "no provider could serve this request"}}`.

## Routing rules (summary)

1. Rank all free models by rating, ties by provider id then model id.
2. Filter by requirements: tools / vision / JSON / minimum context.
3. Try in order; first `2xx` wins. `429` → one retry, same provider.
   `402` / `400` / `401` / `5xx` → provider dead for the trial.
   No response → provider stays alive, move to the next candidate.
4. Relay the answer text and usage verbatim; map the stop reason back.

## Limits

- The providers in the trials are **simulated** (fixed catalogs and
  ratings). Scores describe correctness against that simulation, not
  performance against real free tiers.
- There is **no live public endpoint** and no standalone mode yet — the
  server runs inside the grading harness.
- Routing claims are self-reported: the server's log is the only record of
  which model answered. A verifiable version (attested provider responses,
  then the ranking itself) is future work — the design is discussed in the
  author's Creator Track write-up for zkVerify Agent Games Season 01:
  [Read the full article](https://dev.to/raffasys/my-agent-picks-a-free-ai-model-for-you-how-do-you-know-its-telling-the-truth-4p17).

## License

MIT — see [LICENSE](LICENSE).
