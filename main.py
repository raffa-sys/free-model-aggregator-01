#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Free AI model aggregator.

One server (Python 3.13, standard library only) that speaks both the OpenAI
Chat Completions API (``POST /v1/chat/completions``) and the Anthropic
Messages API (``POST /v1/messages``), JSON and streaming (SSE), over the ten
simulated free-tier providers the trial brief describes (groq, cerebras,
gemini, zai, openrouter, sambanova, pollinations, cloudflare, ovh, kilol).

How it works
------------
* At start (or for ``--list``) it reads the brief from ``MAKEFUN_PROMPT_FILE``:
  the provider directory, the quality ratings table and the workload manifest.
  It probes every provider's catalog route (``GET <base>/v1/models``,
  Gemini ``GET <base>/v1beta/models``) to build the exact catalog it
  registers with the environment, falling back to the ratings table for any
  provider whose probe fails.
* ``GET /v1/models`` lists the free model ids of that catalog, in the
  OpenAI shape under a Bearer token and in the Anthropic shape under an
  ``x-api-key``.
* For every inference request it ranks all free models by rating (ties broken
  by provider id, then model id), filters by the request's requirements
  (tools / vision / structured JSON / minimum context), and tries them in
  order, failing over: a 429 is retried once on the same provider, a 402
  (budget exhausted) marks a provider dead for the rest of the trial, and
  the first 2xx answer wins. One request makes exactly one successful
  provider call.
* Request translation is lossless for everything the client's echo check
  verifies: system text, the last user message's text, tool names in order,
  max_tokens, stop sequences in order, image bytes, and the JSON-mode flag.
  The provider's answer text and usage counts are relayed verbatim, and the
  stop reason is mapped back to the client's stop family.
* The server registers ``POST $MAKEFUN_ENV_URL/register`` with its port and
  catalog, polls ``GET $MAKEFUN_ENV_URL/work`` until the environment says
  the workload is done, and then exits 0.

Routing is fully deterministic: no wall clock, no unseeded randomness.
"""
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

FIXED_CREATED = 1767225600
ANTHROPIC_VERSION = "2023-06-01"
CHUNK_SIZE = 13  # providers stream their answers in 13-char pieces; match it

# Stop-reason families, shared by every wire: 0 stopped, 1 length, 2 tool
# call, 3 content filter / refusal.
OPENAI_STOP = {"stop": 0, "length": 1, "tool_calls": 2, "content_filter": 3,
               "function_call": 2, "refusal": 3}
OPENAI_STOP_OUT = {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"}
ANTHROPIC_STOP = {"end_turn": 0, "max_tokens": 1, "tool_use": 2, "refusal": 3}
ANTHROPIC_STOP_OUT = {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"}
GEMINI_STOP = {"STOP": 0, "MAX_TOKENS": 1, "FUNCTION_CALL": 2, "SAFETY": 3}

PROVIDER_TIMEOUT = 15.0
CONTROL_TIMEOUT = 3.0


# ---------------------------------------------------------------- brief ----
def load_brief(path):
    """Read the three JSON documents embedded in the brief file."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    decoder = json.JSONDecoder()
    docs = {}
    for tag in ("providers", "ratings", "workload"):
        marker = '{"schema":"free-model-aggregator-01/' + tag
        pos = text.find(marker)
        if pos < 0:
            raise ValueError("brief is missing the %s document" % tag)
        doc, _ = decoder.raw_decode(text, pos)
        docs[tag] = doc
    return docs["providers"]["providers"], docs["ratings"]["ratings"], \
        docs["workload"]["manifest"]


class Provider:
    """One simulated provider: its wire, its auth, its key, its models."""

    def __init__(self, entry, rows):
        self.id = entry["id"]
        self.name = entry.get("name", entry["id"])
        self.wire = entry["wire"]          # openai | anthropic | gemini
        self.auth = entry["auth"]          # bearer | x-goog-api-key | none
        self.key_env = entry["key_env"]
        self.key = entry["key"]
        self.path = entry["path"]          # e.g. /p/groq
        # Models come from the ratings table (rating, context, capabilities).
        self.rows = [r for r in rows if r["provider"] == self.id]
        self.by_model = {r["model"]: r for r in self.rows}

    def headers(self):
        if self.auth == "bearer":
            return {"Authorization": "Bearer " + self.key}
        if self.auth == "x-goog-api-key":
            return {"x-goog-api-key": self.key}
        return {}

    def catalog_url(self, env_url):
        route = "/v1beta/models" if self.wire == "gemini" else "/v1/models"
        return env_url + self.path + route

    def infer_url(self, env_url, model, stream):
        if self.wire == "openai":
            return env_url + self.path + "/v1/chat/completions"
        if self.wire == "anthropic":
            return env_url + self.path + "/v1/messages"
        verb = "streamGenerateContent" if stream else "generateContent"
        return env_url + self.path + "/v1beta/models/" + \
            quote(model, safe="") + ":" + verb


# -------------------------------------------------------------- catalog ----
def probe_provider_catalog(provider, env_url):
    """Ask a provider for its model list; None when the probe fails."""
    req = Request(provider.catalog_url(env_url), method="GET")
    for key, value in provider.headers().items():
        req.add_header(key, value)
    try:
        with urlopen(req, timeout=PROVIDER_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except (HTTPError, URLError, TimeoutError, OSError):
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    models = []
    if provider.wire == "gemini":
        for item in data.get("models", []):
            name = item.get("name", "")
            mid = name[len("models/"):] if name.startswith("models/") else name
            if not mid:
                continue
            models.append({"id": mid, "free": bool(item.get("free", False)),
                           "context_length": item.get("context_length",
                                                      item.get("inputTokenLimit"))})
    else:
        for item in data.get("data", []):
            mid = item.get("id")
            if not isinstance(mid, str):
                continue
            models.append({"id": mid, "free": bool(item.get("free", False)),
                           "context_length": item.get("context_length")})
    # The probe is authoritative for free/paid; context falls back to the
    # ratings table when the probe does not carry one.
    out = []
    for m in models:
        ctx = m["context_length"]
        if not isinstance(ctx, int):
            row = provider.by_model.get(m["id"])
            ctx = row["context_length"] if row else None
        out.append({"id": m["id"], "free": m["free"], "context_length": ctx})
    return out or None


def build_catalog(providers, env_url):
    """The catalog document: every provider, all its models, free and paid.

    The same function feeds both ``--list`` and the register call, so the
    two are byte for byte identical.
    """
    catalog_providers = []
    for provider in providers:
        models = None
        if env_url:
            models = probe_provider_catalog(provider, env_url)
        if models is None:
            models = [{"id": row["model"], "free": bool(row["free"]),
                       "context_length": row["context_length"]}
                      for row in provider.rows]
        catalog_providers.append({
            "id": provider.id,
            "base_url": provider.path,
            "key_env": provider.key_env,
            "wire": provider.wire,
            "models": models,
        })
    return {"providers": catalog_providers}


def free_model_ids(catalog):
    """Deduped free model ids of a catalog, in catalog order."""
    seen = []
    for provider in catalog["providers"]:
        for model in provider["models"]:
            if model.get("free") is True and model["id"] not in seen:
                seen.append(model["id"])
    return seen


def catalog_json(catalog):
    return json.dumps(catalog, separators=(",", ":"))


# ------------------------------------------------ canonical request --------
# A request in wire-neutral form: system text, messages of typed parts,
# tools, limits. Everything the client's echo check verifies lives here.
#
# part kinds:
#   text        {"kind": "text", "text": str}
#   image       {"kind": "image", "url": str}            (data: URL or http URL)
#   tool_call   {"kind": "tool_call", "id", "name", "args": str|dict}
#   tool_result {"kind": "tool_result", "id", "text": str}


def _openai_parts(content):
    if isinstance(content, str):
        return [{"kind": "text", "text": content}]
    parts = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append({"kind": "text", "text": block.get("text", "")})
        elif block.get("type") == "image_url":
            url = block.get("image_url") or {}
            parts.append({"kind": "image", "url": url.get("url", "")})
    return parts


def _anthropic_parts(content):
    if isinstance(content, str):
        return [{"kind": "text", "text": content}]
    parts = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append({"kind": "text", "text": block.get("text", "")})
        elif btype == "image":
            src = block.get("source") or {}
            if src.get("type") == "base64":
                mime = src.get("media_type", "image/png")
                parts.append({"kind": "image",
                              "url": "data:%s;base64,%s" % (mime, src.get("data", ""))})
            elif src.get("type") == "url":
                parts.append({"kind": "image", "url": src.get("url", "")})
        elif btype == "tool_use":
            parts.append({"kind": "tool_call", "id": block.get("id", ""),
                          "name": block.get("name", ""),
                          "args": block.get("input", {})})
        elif btype == "tool_result":
            text = block.get("content")
            if isinstance(text, list):
                text = "".join(b.get("text", "") for b in text
                               if isinstance(b, dict))
            parts.append({"kind": "tool_result",
                          "id": block.get("tool_use_id", ""),
                          "text": text or ""})
    return parts


def _system_text_from_blocks(content):
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in (content or [])
                     if isinstance(b, dict) and b.get("type") == "text")


def canon_openai(body):
    system = []
    messages = []
    for msg in body.get("messages", []) or []:
        role = msg.get("role")
        if role == "system":
            system.append(_system_text_from_blocks(msg.get("content")))
            continue
        parts = _openai_parts(msg.get("content"))
        if role == "assistant":
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                parts.append({"kind": "tool_call", "id": call.get("id", ""),
                              "name": fn.get("name", ""),
                              "args": fn.get("arguments", "")})
        elif role == "tool":
            parts = [{"kind": "tool_result", "id": msg.get("tool_call_id", ""),
                      "text": msg.get("content") if
                      isinstance(msg.get("content"), str) else ""}]
        messages.append({"role": role, "parts": parts})
    tools = []
    for tool in body.get("tools") or []:
        fn = (tool.get("function") or {}) if isinstance(tool, dict) else {}
        if tool.get("type") == "function" and fn.get("name"):
            tools.append({"name": fn["name"],
                          "description": fn.get("description", ""),
                          "parameters": fn.get("parameters", {})})
    response_format = body.get("response_format") or {}
    return {
        "api": "openai",
        "stream": bool(body.get("stream")),
        "system": "\n".join(s for s in system if s),
        "messages": messages,
        "tools": tools,
        "tool_choice": body.get("tool_choice"),
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "stops": _string_list(body.get("stop")),
        "json_mode": response_format.get("type") in
                     ("json_object", "json_schema"),
    }


def canon_anthropic(body):
    system = _system_text_from_blocks(body.get("system"))
    messages = []
    for msg in body.get("messages", []) or []:
        messages.append({"role": msg.get("role"),
                         "parts": _anthropic_parts(msg.get("content"))})
    tools = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and tool.get("name"):
            tools.append({"name": tool["name"],
                          "description": tool.get("description", ""),
                          "parameters": tool.get("input_schema", {})})
    return {
        "api": "anthropic",
        "stream": bool(body.get("stream")),
        "system": system,
        "messages": messages,
        "tools": tools,
        "tool_choice": body.get("tool_choice"),
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "stops": _string_list(body.get("stop_sequences")),
        "json_mode": False,  # the Anthropic wire carries no JSON-mode flag
    }


def _string_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [v for v in value if isinstance(v, str)]


def _split_data_url(url):
    """data:<mime>;base64,<data> -> (mime, data); else (None, None)."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, None
    head, _, data = url.partition(",")
    mime = head[len("data:"):].split(";")[0] or "image/png"
    return mime, data


# ------------------------------------------------- wire builders ---------
def _openai_message(msg):
    parts = msg["parts"]
    text_parts = [p for p in parts if p["kind"] == "text"]
    images = [p for p in parts if p["kind"] == "image"]
    calls = [p for p in parts if p["kind"] == "tool_call"]
    results = [p for p in parts if p["kind"] == "tool_result"]
    role = msg["role"]
    out = {"role": role}
    if results:
        first = results[0]
        out["content"] = first["text"]
        out["tool_call_id"] = first["id"]
        return out
    if not images and not calls:
        if len(text_parts) == 1:
            out["content"] = text_parts[0]["text"]
        else:
            out["content"] = [{"type": "text", "text": p["text"]}
                              for p in text_parts]
    else:
        content = []
        for p in parts:
            if p["kind"] == "text":
                content.append({"type": "text", "text": p["text"]})
            elif p["kind"] == "image":
                content.append({"type": "image_url",
                                "image_url": {"url": p["url"]}})
        out["content"] = content
    if calls:
        out["tool_calls"] = [{
            "id": c["id"], "type": "function",
            "function": {"name": c["name"],
                         "arguments": c["args"] if isinstance(c["args"], str)
                         else json.dumps(c["args"], separators=(",", ":"))},
        } for c in calls]
    return out


def build_openai(canon, model, json_mode):
    body = {"model": model,
            "messages": ([{"role": "system", "content": canon["system"]}]
                         if canon["system"] else []) +
                        [_openai_message(m) for m in canon["messages"]],
            "max_tokens": canon["max_tokens"],
            "stream": canon["stream"]}
    if canon["temperature"] is not None:
        body["temperature"] = canon["temperature"]
    if canon["tools"]:
        body["tools"] = [{"type": "function",
                          "function": {"name": t["name"],
                                       "description": t["description"],
                                       "parameters": t["parameters"]}}
                         for t in canon["tools"]]
    if canon["tool_choice"] is not None:
        body["tool_choice"] = canon["tool_choice"]
    if canon["stops"]:
        body["stop"] = canon["stops"]
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if canon["stream"]:
        body["stream_options"] = {"include_usage": True}
    return body


def _anthropic_content(parts):
    text_parts = [p for p in parts if p["kind"] == "text"]
    if len(parts) == 1 and parts[0]["kind"] == "text":
        return parts[0]["text"]
    blocks = []
    for p in parts:
        if p["kind"] == "text":
            blocks.append({"type": "text", "text": p["text"]})
        elif p["kind"] == "image":
            mime, data = _split_data_url(p["url"])
            if data is not None:
                blocks.append({"type": "image",
                               "source": {"type": "base64",
                                          "media_type": mime, "data": data}})
            else:
                blocks.append({"type": "image",
                               "source": {"type": "url", "url": p["url"]}})
        elif p["kind"] == "tool_call":
            args = p["args"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            blocks.append({"type": "tool_use", "id": p["id"],
                           "name": p["name"], "input": args})
        elif p["kind"] == "tool_result":
            blocks.append({"type": "tool_result",
                           "tool_use_id": p["id"], "content": p["text"]})
    return blocks


def _anthropic_tool_choice(choice):
    if choice is None:
        return None
    if choice == "auto":
        return {"type": "auto"}
    if choice == "none":
        return {"type": "none"}
    if choice == "required":
        return {"type": "any"}
    if isinstance(choice, dict):
        if choice.get("type") == "function":
            fn = choice.get("function") or {}
            return {"type": "tool", "name": fn.get("name", "")}
        return choice
    return {"type": "auto"}


def build_anthropic(canon, model):
    body = {"model": model,
            "messages": [{"role": m["role"],
                          "content": _anthropic_content(m["parts"])}
                         for m in canon["messages"]],
            "max_tokens": canon["max_tokens"],
            "stream": canon["stream"]}
    if canon["system"]:
        body["system"] = canon["system"]
    if canon["temperature"] is not None:
        body["temperature"] = canon["temperature"]
    if canon["tools"]:
        body["tools"] = [{"name": t["name"],
                          "description": t["description"],
                          "input_schema": t["parameters"]}
                         for t in canon["tools"]]
    choice = _anthropic_tool_choice(canon["tool_choice"])
    if choice is not None:
        body["tool_choice"] = choice
    if canon["stops"]:
        body["stop_sequences"] = canon["stops"]
    return body


def _gemini_parts(parts):
    out = []
    for p in parts:
        if p["kind"] == "text":
            out.append({"text": p["text"]})
        elif p["kind"] == "image":
            mime, data = _split_data_url(p["url"])
            if data is not None:
                out.append({"inlineData": {"mimeType": mime, "data": data}})
            else:
                out.append({"fileData": {"mimeType": "image/png",
                                         "fileUri": p["url"]}})
        elif p["kind"] == "tool_call":
            args = p["args"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            out.append({"functionCall": {"name": p["name"], "args": args}})
        elif p["kind"] == "tool_result":
            out.append({"functionResponse": {"name": "",
                                             "response": {"result": p["text"]}}})
    return out


def build_gemini(canon, model, json_mode):
    body = {"contents": [{"role": "user" if m["role"] == "user" else "model",
                          "parts": _gemini_parts(m["parts"])}
                         for m in canon["messages"]]}
    if canon["system"]:
        body["systemInstruction"] = {"parts": [{"text": canon["system"]}]}
    config = {"maxOutputTokens": canon["max_tokens"]}
    if canon["temperature"] is not None:
        config["temperature"] = canon["temperature"]
    if canon["stops"]:
        config["stopSequences"] = canon["stops"]
    if json_mode:
        config["responseMimeType"] = "application/json"
    body["generationConfig"] = config
    if canon["tools"]:
        body["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"],
             "parameters": t["parameters"]} for t in canon["tools"]]}]
    return body


# ------------------------------------------------- provider calls --------
def call_provider(provider, url, body, stream):
    """POST a JSON body; return (status, raw_text). Network errors -> 0."""
    headers = dict(provider.headers())
    headers["Content-Type"] = "application/json"
    if provider.wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    headers["Accept"] = "text/event-stream" if stream else "application/json"
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    req = Request(url, data=payload, method="POST")
    for key, value in headers.items():
        req.add_header(key, value)
    try:
        with urlopen(req, timeout=PROVIDER_TIMEOUT) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except HTTPError as err:
        try:
            raw = err.read().decode("utf-8", "replace")
        except OSError:
            raw = ""
        return err.code, raw
    except (URLError, TimeoutError, OSError):
        return 0, ""


def _sse_frames(raw):
    frames = []
    for chunk in raw.split("\n\n"):
        event, data = None, []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].strip())
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def _json_or_none(text):
    try:
        return json.loads(text)
    except ValueError:
        return None


def parse_openai_provider(raw, stream):
    """Return (text, prompt_tokens, completion_tokens, stop_family)."""
    if not stream:
        data = _json_or_none(raw) or {}
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            text = "".join(b.get("text", "") for b in content
                           if isinstance(b, dict))
        else:
            text = content or ""
        usage = data.get("usage") or {}
        return (text, usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                OPENAI_STOP.get(choice.get("finish_reason"), 0))
    text, prompt, completion, family = "", 0, 0, 0
    for _event, data in _sse_frames(raw):
        if data == "[DONE]":
            continue
        frame = _json_or_none(data)
        if not isinstance(frame, dict):
            continue
        for choice in frame.get("choices") or []:
            delta = choice.get("delta") or {}
            text += delta.get("content") or ""
            if choice.get("finish_reason"):
                family = OPENAI_STOP.get(choice["finish_reason"], 0)
        if frame.get("usage"):
            prompt = frame["usage"].get("prompt_tokens", prompt)
            completion = frame["usage"].get("completion_tokens", completion)
    return text, prompt, completion, family


def parse_anthropic_provider(raw, stream):
    if not stream:
        data = _json_or_none(raw) or {}
        text = "".join(b.get("text", "") for b in data.get("content") or []
                       if isinstance(b, dict) and b.get("type") == "text")
        usage = data.get("usage") or {}
        return (text, usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                ANTHROPIC_STOP.get(data.get("stop_reason"), 0))
    text, prompt, completion, family = "", 0, 0, 0
    for event, data in _sse_frames(raw):
        frame = _json_or_none(data)
        if not isinstance(frame, dict):
            continue
        if event == "message_start":
            usage = (frame.get("message") or {}).get("usage") or {}
            prompt = usage.get("input_tokens", 0)
        elif event == "content_block_delta":
            delta = frame.get("delta") or {}
            if delta.get("type") == "text_delta":
                text += delta.get("text", "")
        elif event == "message_delta":
            delta = frame.get("delta") or {}
            if delta.get("stop_reason"):
                family = ANTHROPIC_STOP.get(delta["stop_reason"], 0)
            usage = frame.get("usage") or {}
            completion = usage.get("output_tokens", completion)
    return text, prompt, completion, family


def parse_gemini_provider(raw, stream):
    if not stream:
        data = _json_or_none(raw) or {}
        candidates = data.get("candidates") or []
        text, family = "", 0
        if candidates:
            content = candidates[0].get("content") or {}
            text = "".join(p.get("text", "") for p in content.get("parts") or []
                           if isinstance(p, dict))
            family = GEMINI_STOP.get(candidates[0].get("finishReason"), 0)
        usage = data.get("usageMetadata") or {}
        return (text, usage.get("promptTokenCount", 0),
                usage.get("candidatesTokenCount", 0), family)
    text, prompt, completion, family = "", 0, 0, 0
    for _event, data in _sse_frames(raw):
        frame = _json_or_none(data)
        if not isinstance(frame, dict):
            continue
        for candidate in frame.get("candidates") or []:
            content = candidate.get("content") or {}
            text += "".join(p.get("text", "") for p in content.get("parts")
                            or [] if isinstance(p, dict))
            if candidate.get("finishReason"):
                family = GEMINI_STOP.get(candidate["finishReason"], 0)
        usage = frame.get("usageMetadata") or {}
        if usage:
            prompt = usage.get("promptTokenCount", prompt)
            completion = usage.get("candidatesTokenCount", completion)
    return text, prompt, completion, family


PARSERS = {"openai": parse_openai_provider,
           "anthropic": parse_anthropic_provider,
           "gemini": parse_gemini_provider}


# --------------------------------------------- client rendering ----------
def render_openai(text, prompt, completion, family, model, stream):
    stop = OPENAI_STOP_OUT[family]
    if not stream:
        return {"id": "chatcmpl-sim", "object": "chat.completion",
                "created": FIXED_CREATED, "model": model,
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": text},
                             "finish_reason": stop, "logprobs": None}],
                "usage": {"prompt_tokens": prompt,
                          "completion_tokens": completion,
                          "total_tokens": prompt + completion}}
    lines = []
    base = {"id": "chatcmpl-sim", "object": "chat.completion.chunk",
            "created": FIXED_CREATED, "model": model}
    role = dict(base, choices=[{"index": 0, "delta": {"role": "assistant"},
                                "finish_reason": None}])
    lines.append("data: " + json.dumps(role, separators=(",", ":")))
    for i in range(0, len(text), CHUNK_SIZE):
        chunk = dict(base, choices=[{"index": 0,
                                     "delta": {"content": text[i:i + CHUNK_SIZE]},
                                     "finish_reason": None}])
        lines.append("data: " + json.dumps(chunk, separators=(",", ":")))
    finish = dict(base, choices=[{"index": 0, "delta": {},
                                  "finish_reason": stop}])
    lines.append("data: " + json.dumps(finish, separators=(",", ":")))
    usage = dict(base, choices=[],
                 usage={"prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "total_tokens": prompt + completion})
    lines.append("data: " + json.dumps(usage, separators=(",", ":")))
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def render_anthropic(text, prompt, completion, family, model, stream):
    stop = ANTHROPIC_STOP_OUT[family]
    if not stream:
        return {"id": "msg_sim", "type": "message", "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": text}],
                "stop_reason": stop, "stop_sequence": None,
                "usage": {"input_tokens": prompt, "output_tokens": completion}}
    lines = []
    start = {"type": "message_start",
             "message": {"id": "msg_sim", "type": "message",
                         "role": "assistant", "model": model, "content": [],
                         "stop_reason": None, "stop_sequence": None,
                         "usage": {"input_tokens": prompt,
                                   "output_tokens": 0}}}
    lines.append("event: message_start\ndata: " +
                 json.dumps(start, separators=(",", ":")))
    block_start = {"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}}
    lines.append("event: content_block_start\ndata: " +
                 json.dumps(block_start, separators=(",", ":")))
    for i in range(0, len(text), CHUNK_SIZE):
        delta = {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "text_delta",
                           "text": text[i:i + CHUNK_SIZE]}}
        lines.append("event: content_block_delta\ndata: " +
                     json.dumps(delta, separators=(",", ":")))
    block_stop = {"type": "content_block_stop", "index": 0}
    lines.append("event: content_block_stop\ndata: " +
                 json.dumps(block_stop, separators=(",", ":")))
    message_delta = {"type": "message_delta",
                     "delta": {"stop_reason": stop, "stop_sequence": None},
                     "usage": {"output_tokens": completion}}
    lines.append("event: message_delta\ndata: " +
                 json.dumps(message_delta, separators=(",", ":")))
    message_stop = {"type": "message_stop"}
    lines.append("event: message_stop\ndata: " +
                 json.dumps(message_stop, separators=(",", ":")))
    return "\n\n".join(lines) + "\n\n"


RENDERERS = {"openai": render_openai, "anthropic": render_anthropic}


# --------------------------------------------------- routing ------------
def requirements_for(manifest, request_id, canon):
    """The trial's requirements for this request, from the manifest when the
    environment told us which request is next; otherwise derived from the
    body itself."""
    for row in manifest:
        if row["id"] == request_id:
            req = dict(row.get("requires") or {})
            req.setdefault("tools", False)
            req.setdefault("vision", False)
            req.setdefault("json", False)
            req.setdefault("min_context", 0)
            return req
    # Fallback: what the body itself reveals.
    vision = any(p["kind"] == "image" for m in canon["messages"]
                 for p in m["parts"])
    return {"tools": bool(canon["tools"]), "vision": vision,
            "json": canon["json_mode"], "min_context": 0}


def candidates_for(providers, canon, requires):
    """Free models that meet the requirements, best rating first.

    A JSON-mode OpenAI request may only go to an OpenAI or Gemini wire
    (they can carry the JSON flag); an Anthropic request's JSON requirement
    is met by any wire, since its JSON need is on the model, not the wire.
    """
    out = []
    for provider in providers:
        if canon["api"] == "openai" and requires.get("json"):
            if provider.wire not in ("openai", "gemini"):
                continue
        for row in provider.rows:
            if not row["free"]:
                continue
            if requires.get("tools") and not row["tools"]:
                continue
            if requires.get("vision") and not row["vision"]:
                continue
            if requires.get("json") and not row["json"]:
                continue
            ctx = row["context_length"] or 0
            if ctx < requires.get("min_context", 0):
                continue
            out.append((provider, row))
    out.sort(key=lambda item: (-item[1]["rating_milli"], item[0].id,
                               item[1]["model"]))
    return out


class Aggregator:
    def __init__(self, providers, manifest, catalog, env_url):
        self.providers = providers
        self.manifest = manifest
        self.catalog = catalog
        self.env_url = env_url
        self.dead = set()   # providers out of budget (402) for this trial
        self.lock = threading.Lock()
        self.httpd = None
        self.done = threading.Event()

    # -- environment control plane -------------------------------------
    def register(self):
        body = {"protocol_version": 1, "port": self.httpd.server_address[1],
                "catalog": self.catalog}
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        for _attempt in range(30):
            req = Request(self.env_url.rstrip("/") + "/register",
                          data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            try:
                with urlopen(req, timeout=CONTROL_TIMEOUT) as resp:
                    if resp.status == 200:
                        return True
            except (HTTPError, URLError, TimeoutError, OSError):
                time.sleep(0.5)
        return False

    def work_state(self):
        try:
            with urlopen(self.env_url.rstrip("/") + "/work",
                         timeout=CONTROL_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            return None

    def monitor(self):
        while not self.done.is_set():
            state = self.work_state()
            if state and state.get("done"):
                self.done.set()
                if self.httpd is not None:
                    self.httpd.shutdown()
                return
            time.sleep(0.25)

    # -- one inference request -----------------------------------------
    def serve(self, api, body):
        canon = canon_openai(body) if api == "openai" else canon_anthropic(body)
        state = self.work_state()
        request_id = state.get("next") if state else None
        requires = requirements_for(self.manifest, request_id, canon)
        # A JSON-mode request needs the flag on the provider call, whatever
        # wire carries it.
        json_mode = canon["json_mode"] or bool(requires.get("json"))
        chosen = None
        result = None
        for provider, row in candidates_for(self.providers, canon, requires):
            with self.lock:
                if provider.id in self.dead:
                    continue
            if provider.wire == "openai":
                request_body = build_openai(canon, row["model"], json_mode)
            elif provider.wire == "anthropic":
                request_body = build_anthropic(canon, row["model"])
            else:
                request_body = build_gemini(canon, row["model"], json_mode)
            url = provider.infer_url(self.env_url, row["model"],
                                     canon["stream"])
            for attempt in range(2):
                status, raw = call_provider(provider, url, request_body,
                                            canon["stream"])
                if status == 200:
                    result = PARSERS[provider.wire](raw, canon["stream"])
                    chosen = row["model"]
                    break
                if status == 429:
                    continue  # retry once on the same provider
                if status == 402:
                    with self.lock:
                        self.dead.add(provider.id)
                elif status != 0:
                    # 400/401/5xx: the provider answered, but rejected
                    # this request for good; stop trying it this trial.
                    with self.lock:
                        self.dead.add(provider.id)
                # status 0 is a network hiccup, not a verdict on the
                # provider: leave it alive for later requests, and just
                # move on to the next candidate for this one.
                break
            if chosen is not None:
                break
        if chosen is None:
            return 503, {"error": {"type": "server_error",
                                   "message": "no provider could serve this request"}}
        text, prompt, completion, family = result
        rendered = RENDERERS[api](text, prompt, completion, family, chosen,
                                  canon["stream"])
        return 200, rendered


# ------------------------------------------------------- HTTP ----------
class Handler(BaseHTTPRequestHandler):
    aggregator = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, payload, content_type):
        if isinstance(payload, str):
            data = payload.encode("utf-8")
        else:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/v1/models":
            self._send(404, {"error": {"type": "not_found",
                                       "message": "unknown route"}},
                       "application/json")
            return
        auth = self.headers.get("Authorization", "")
        api_key = self.headers.get("x-api-key")
        ids = free_model_ids(self.aggregator.catalog)
        owned_by = {}
        for provider in self.aggregator.catalog["providers"]:
            for model in provider["models"]:
                if model.get("free"):
                    owned_by.setdefault(model["id"], provider["id"])
        if api_key:
            data = [{"type": "model", "id": mid, "display_name": mid,
                     "created_at": "2026-01-01T00:00:00Z", "free": True}
                    for mid in ids]
            self._send(200, {"data": data, "has_more": False},
                       "application/json")
        else:
            data = [{"id": mid, "object": "model",
                     "created": FIXED_CREATED,
                     "owned_by": owned_by.get(mid, ""),
                     "free": True} for mid in ids]
            self._send(200, {"object": "list", "data": data},
                       "application/json")

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            api = "openai"
        elif self.path == "/v1/messages":
            api = "anthropic"
        else:
            self._send(404, {"error": {"type": "not_found",
                                       "message": "unknown route"}},
                       "application/json")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            self._send(400, {"error": {"type": "invalid_request_error",
                                       "message": "body is not valid JSON"}},
                       "application/json")
            return
        status, rendered = self.aggregator.serve(api, body)
        content_type = ("text/event-stream"
                        if body.get("stream") else "application/json")
        self._send(status, rendered, content_type)


# --------------------------------------------------------- main ----------
def main(argv):
    if "--list" in argv:
        brief_path = os.environ.get("MAKEFUN_PROMPT_FILE")
        if not brief_path:
            print('{"providers":[]}')
            return 0
        directory, ratings, _manifest = load_brief(brief_path)
        providers = [Provider(entry, ratings) for entry in directory]
        env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
        catalog = build_catalog(providers, env_url or None)
        print(catalog_json(catalog))
        return 0

    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    brief_path = os.environ.get("MAKEFUN_PROMPT_FILE")
    if not env_url or not brief_path:
        print("MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE are required",
              file=sys.stderr)
        return 1
    directory, ratings, manifest = load_brief(brief_path)
    providers = [Provider(entry, ratings) for entry in directory]
    catalog = build_catalog(providers, env_url)

    aggregator = Aggregator(providers, manifest, catalog, env_url)
    httpd = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    aggregator.httpd = httpd
    Handler.aggregator = aggregator

    def _graceful(_signum, _frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)

    if not aggregator.register():
        print("could not register with the environment", file=sys.stderr)
        return 1
    threading.Thread(target=aggregator.monitor, daemon=True).start()
    print("aggregator listening on port %d, registered %d providers"
          % (httpd.server_address[1], len(providers)), file=sys.stderr)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
