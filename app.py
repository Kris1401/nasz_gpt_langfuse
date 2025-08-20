# app.py
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import math
import random
from uuid import uuid4
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from contextlib import contextmanager, nullcontext

import streamlit as st
from dotenv import load_dotenv
import httpx
import openai as openai_pkg
from openai import OpenAI
from langfuse import get_client as get_langfuse_client

# ============================ KONFIG / WERSJE ============================

APP_VERSION = "1.3.7"  # + GPT-5 Responses API routing (stream + non-stream)
PROMPT_VERSION = "2025-08-18"
SDK_VERSIONS = {
    "openai": getattr(openai_pkg, "__version__", "unknown"),
    "httpx": getattr(httpx, "__version__", "unknown"),
    "streamlit": getattr(st, "__version__", "unknown"),
    "app": APP_VERSION,
}

# Oficjalne stawki USD / 1M tokenów (input/output). Cached input nie jest tu liczony.
DEFAULT_PRICING: Dict[str, Dict[str, Optional[float]]] = {
    "gpt-4o":         {"input_tokens": 2.50 / 1_000_000,  "output_tokens": 10.00 / 1_000_000},
    "gpt-4o-mini":    {"input_tokens": 0.15 / 1_000_000,  "output_tokens": 0.60 / 1_000_000},
    # GPT-5 rodzina — uzupełnij tylko jeśli masz dostęp (wartości przykładowe)
    "gpt-5":          {"input_tokens": 1.25 / 1_000_000,  "output_tokens": 10.00 / 1_000_000},
    "gpt-5-thinking": {"input_tokens": 1.25 / 1_000_000,  "output_tokens": 10.00 / 1_000_000},
}
USD_TO_PLN = 3.97

MODEL_FAMILIES = {
    "GPT-4o family": ["gpt-4o", "gpt-4o-mini"],
    "GPT-5 family (jeśli masz dostęp)": ["gpt-5", "gpt-5-thinking"],
}

DEFAULT_SAMPLE_RATE = 1.0
ALWAYS_LOG_FIRST_MESSAGE = True

# ============================ ENV + KLIENCI ==============================

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Brak OPENAI_API_KEY.")

# trust_env=False -> httpx NIE dziedziczy proxy ze środowiska (fix dla 'proxies')
_http_client = httpx.Client(timeout=60, trust_env=False)
openai_client = OpenAI(api_key=API_KEY, http_client=_http_client)

# Langfuse: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST (opcjonalnie)
langfuse = get_langfuse_client()

# ============================ KOMPAT WRAPPERY (Langfuse) =================

class _NoopGen:
    def update(self, *args, **kwargs):
        pass

@contextmanager
def lf_generation_ctx(**kwargs):
    """Zwraca obiekt 'generation' z metodą .update(...) (no-op jeśli brak API)."""
    if hasattr(langfuse, "start_as_current_generation"):
        cm = langfuse.start_as_current_generation(**kwargs)
        try:
            gen = cm.__enter__()
        except Exception:
            gen = _NoopGen()
        try:
            yield gen
        finally:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
    else:
        yield _NoopGen()

@contextmanager
def lf_trace_ctx(**kwargs):
    if hasattr(langfuse, "start_as_current_trace"):
        cm = langfuse.start_as_current_trace(**kwargs)
        try:
            cm.__enter__()
            yield
        finally:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
    else:
        yield

@contextmanager
def lf_span_ctx(**kwargs):
    if hasattr(langfuse, "start_as_current_span"):
        cm = langfuse.start_as_current_span(**kwargs)
        try:
            cm.__enter__()
            yield
        finally:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
    else:
        yield

# ============================ UTILS ======================================

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-\s.]*)?(?:\(?\d{2,4}\)?[-\s.]*)?\d{3}[-\s.]?\d{2,4}")

def redact_pii(text: str) -> str:
    return PHONE_RE.sub("[phone]", EMAIL_RE.sub("[email]", text))

def redact_messages_for_logging(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [{"role": m.get("role", "user"), "content": redact_pii(m.get("content", ""))} for m in messages]

def approx_tokens_from_text(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))  # ~4 znaki ≈ 1 token

def approx_prompt_tokens(messages: List[Dict[str, str]]) -> int:
    return approx_tokens_from_text("".join(m.get("content", "") for m in messages))

def safe_index(options: List[str], value: str, default: int = 0) -> int:
    try:
        return options.index(value)
    except Exception:
        return default

# ============================ I/O POMOCNICZE =============================

DB_PATH = Path("db")
DB_CONVERSATIONS_PATH = DB_PATH / "conversations"
PRICING_PATH = DB_PATH / "pricing.json"
SESSION_PATH = DB_PATH / "session.json"

def _safe_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _safe_read_json(path: Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {} if default is None else default

def load_pricing() -> Dict[str, Dict[str, Optional[float]]]:
    data = _safe_read_json(PRICING_PATH, default={})
    if not data:
        _safe_write_json(PRICING_PATH, DEFAULT_PRICING)
        return DEFAULT_PRICING.copy()
    merged = DEFAULT_PRICING.copy()
    merged.update(data)
    return merged

def save_pricing(pricing: Dict[str, Dict[str, Optional[float]]]) -> None:
    _safe_write_json(PRICING_PATH, pricing)

# ============================ BAZA ROZMÓW ================================

DEFAULT_PERSONALITY = (
    "Jesteś pomocnym asystentem. Odpowiadasz jasno, zwięźle i rzeczowo. "
    "Gdy to możliwe, podajesz konkretne przykłady."
)

def _bootstrap_db_if_needed() -> None:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    DB_CONVERSATIONS_PATH.mkdir(parents=True, exist_ok=True)
    current_path = DB_PATH / "current.json"
    if not current_path.exists():
        conversation_id = 1
        conversation = {
            "id": conversation_id,
            "name": "Konwersacja 1",
            "chatbot_personality": DEFAULT_PERSONALITY,
            "messages": [],
        }
        _safe_write_json(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", conversation)
        _safe_write_json(current_path, {"current_conversation_id": conversation_id})

def load_conversation_to_state(conversation: Dict[str, Any]) -> None:
    st.session_state["id"] = conversation["id"]
    st.session_state["name"] = conversation["name"]
    st.session_state["messages"] = conversation["messages"]
    st.session_state["chatbot_personality"] = conversation["chatbot_personality"]

def load_current_conversation() -> None:
    _bootstrap_db_if_needed()
    current = _safe_read_json(DB_PATH / "current.json", default={"current_conversation_id": 1})
    conversation_id = current["current_conversation_id"]
    conversation = _safe_read_json(
        DB_CONVERSATIONS_PATH / f"{conversation_id}.json",
        default={
            "id": conversation_id,
            "name": f"Konwersacja {conversation_id}",
            "chatbot_personality": DEFAULT_PERSONALITY,
            "messages": [],
        },
    )
    load_conversation_to_state(conversation)

def save_current_conversation_messages() -> None:
    conversation_id = st.session_state["id"]
    path = DB_CONVERSATIONS_PATH / f"{conversation_id}.json"
    conv = _safe_read_json(path, default={})
    conv.update({"messages": st.session_state.get("messages", [])})
    _safe_write_json(path, conv)

def save_current_conversation_name() -> None:
    conversation_id = st.session_state["id"]
    path = DB_CONVERSATIONS_PATH / f"{conversation_id}.json"
    conv = _safe_read_json(path, default={})
    conv.update({"name": st.session_state["new_conversation_name"]})
    _safe_write_json(path, conv)

def save_current_conversation_personality() -> None:
    conversation_id = st.session_state["id"]
    path = DB_CONVERSATIONS_PATH / f"{conversation_id}.json"
    conv = _safe_read_json(path, default={})
    conv.update({"chatbot_personality": st.session_state["new_chatbot_personality"]})
    _safe_write_json(path, conv)

def create_new_conversation() -> None:
    ids: List[int] = []
    for p in DB_CONVERSATIONS_PATH.glob("*.json"):
        try:
            ids.append(int(p.stem))
        except ValueError:
            pass
    conversation_id = (max(ids) + 1) if ids else 1
    personality = st.session_state.get("chatbot_personality") or DEFAULT_PERSONALITY
    conversation = {
        "id": conversation_id,
        "name": f"Konwersacja {conversation_id}",
        "chatbot_personality": personality,
        "messages": [],
    }
    _safe_write_json(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", conversation)
    _safe_write_json(DB_PATH / "current.json", {"current_conversation_id": conversation_id})
    load_conversation_to_state(conversation)
    # tu rerun jest OK (nie w callbacku)
    st.rerun()

def switch_conversation(conversation_id: int) -> None:
    """Bez st.rerun() — wywołamy rerun w miejscu kliknięcia przycisku (poza callbackiem)."""
    conv = _safe_read_json(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", default=None)
    if not conv:
        st.warning("Nie znaleziono konwersacji.")
        return
    _safe_write_json(DB_PATH / "current.json", {"current_conversation_id": conversation_id})
    load_conversation_to_state(conv)

def list_conversations() -> List[Dict[str, Any]]:
    conversations: List[Dict[str, Any]] = []
    for p in DB_CONVERSATIONS_PATH.glob("*.json"):
        conv = _safe_read_json(p, default=None)
        if conv:
            conversations.append({"id": conv["id"], "name": conv["name"]})
    return conversations

# ============================ CENNIK / KOSZTY ============================

def load_pricing_into_state():
    if "pricing_map" not in st.session_state:
        st.session_state["pricing_map"] = load_pricing()

def get_pricing_for_model(model: str) -> Dict[str, Optional[float]]:
    pm = st.session_state["pricing_map"]
    if model in pm:
        return pm[model]
    pm[model] = {"input_tokens": None, "output_tokens": None}
    save_pricing(pm)
    return pm[model]

def compute_cost_usd(usage: Dict[str, int], model: str) -> float:
    pr = get_pricing_for_model(model)
    it = pr.get("input_tokens")
    ot = pr.get("output_tokens")
    if it is None or ot is None:
        return 0.0
    return usage.get("prompt_tokens", 0) * it + usage.get("completion_tokens", 0) * ot

# ============================ RETRY =====================================

def call_with_retry(func, *, retries: int = 1, base_delay: float = 0.6):
    try:
        return func()
    except Exception:
        if retries <= 0:
            raise
        time.sleep(base_delay)
        return call_with_retry(func, retries=retries - 1, base_delay=base_delay * 2)

# ============================ MODEL: STREAM / NON-STREAM =================

# ------ Chat Completions (gpt-4o, 4o-mini, 3.5, itp.) ------

def _chat_stream_completion(
    messages: List[Dict[str, str]],
    model_name: str,
    placeholder: st.delta_generator.DeltaGenerator,
    *,
    max_tokens: int = 1024,
) -> Tuple[str, float, float]:
    full_text = ""
    t_start = time.perf_counter()
    t_first = None

    def _do():
        return openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
        )

    try:
        stream = call_with_retry(_do, retries=1)
        for chunk in stream:
            if chunk and chunk.choices:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    if t_first is None:
                        t_first = time.perf_counter()
                    full_text += delta
                    placeholder.markdown(full_text)
    except Exception as e:
        placeholder.error(f"❌ Błąd streamingu (Chat Completions): {e}")
        return "", 0.0, 0.0

    t_total = time.perf_counter()
    ttfb_ms = (t_first - t_start) * 1000 if t_first else (t_total - t_start) * 1000
    total_ms = (t_total - t_start) * 1000
    return full_text, ttfb_ms, total_ms


def _chat_non_stream_completion(
    messages: List[Dict[str, str]],
    model_name: str,
    *,
    max_tokens: int = 1024,
) -> Tuple[str, Dict[str, int], float]:
    t0 = time.perf_counter()

    def _do():
        return openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
        )

    resp = call_with_retry(_do, retries=1)
    t1 = time.perf_counter()

    content = resp.choices[0].message.content if (resp and resp.choices) else ""
    usage: Dict[str, int] = {}
    if getattr(resp, "usage", None):
        usage = {
            "completion_tokens": resp.usage.completion_tokens,
            "prompt_tokens": resp.usage.prompt_tokens,
            "total_tokens": resp.usage.total_tokens,
        }

    latency_ms = (t1 - t0) * 1000.0
    return content, usage, latency_ms

# ------ GPT‑5: Responses API (NOWE ENDPOINTY) ------

def _messages_to_input_text(messages: List[Dict[str, str]]) -> str:
    """Łączy messages w prompt tekstowy dla Responses API (konwersacyjny styl)."""
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _responses_stream_completion(
    messages: List[Dict[str, str]],
    model_name: str,
    placeholder: st.delta_generator.DeltaGenerator,
    *,
    max_tokens_wanted: int = 1024,
) -> Tuple[str, float, float]:
    """Streaming dla GPT‑5 z kompatybilnością różnych SDK (param nazwy lub brak)."""
    full_text = ""
    t_start = time.perf_counter()
    t_first = None

    base_kwargs = dict(model=model_name, input=_messages_to_input_text(messages))

    def _stream_with(kwargs):
        return openai_client.responses.stream(**kwargs)

    try:
        try:
            # 1) Nowy wariant
            with _stream_with({**base_kwargs, "max_completion_tokens": max_tokens_wanted}) as stream:
                for event in stream:
                    if getattr(event, "type", "") == "response.output_text.delta":
                        if t_first is None:
                            t_first = time.perf_counter()
                        full_text += event.delta
                        placeholder.markdown(full_text)
        except TypeError:
            # 2) Alternatywna nazwa
            with _stream_with({**base_kwargs, "max_output_tokens": max_tokens_wanted}) as stream:
                for event in stream:
                    if getattr(event, "type", "") == "response.output_text.delta":
                        if t_first is None:
                            t_first = time.perf_counter()
                        full_text += event.delta
                        placeholder.markdown(full_text)
    except TypeError:
        # 3) Stary build – bez limitu
        with _stream_with(base_kwargs) as stream:
            for event in stream:
                if getattr(event, "type", "") == "response.output_text.delta":
                    if t_first is None:
                        t_first = time.perf_counter()
                    full_text += event.delta
                    placeholder.markdown(full_text)
    except Exception as e:
        placeholder.error(f"❌ Błąd streamingu (Responses API): {e}")
        return "", 0.0, 0.0

    t_end = time.perf_counter()
    ttfb_ms = (t_first - t_start) * 1000 if t_first else (t_end - t_start) * 1000
    total_ms = (t_end - t_start) * 1000
    return full_text, ttfb_ms, total_ms


def _responses_non_stream_completion(
    messages: List[Dict[str, str]],
    model_name: str,
    *,
    max_tokens_wanted: int = 1024,
) -> Tuple[str, Dict[str, int], float]:
    """Non‑stream dla GPT‑5 z kompatybilnością nazw parametrów i pól usage."""
    base_kwargs = dict(model=model_name, input=_messages_to_input_text(messages))

    t0 = time.perf_counter()
    try:
        try:
            resp = openai_client.responses.create(**{**base_kwargs, "max_completion_tokens": max_tokens_wanted})
        except TypeError:
            resp = openai_client.responses.create(**{**base_kwargs, "max_output_tokens": max_tokens_wanted})
    except TypeError:
        resp = openai_client.responses.create(**base_kwargs)
    t1 = time.perf_counter()

    try:
        content = resp.output_text
    except Exception:
        try:
            content = "".join(
                blk.text
                for out in getattr(resp, "output", [])
                for blk in getattr(out, "content", [])
                if hasattr(blk, "text")
            )
        except Exception:
            content = ""

    usage: Dict[str, int] = {}
    if getattr(resp, "usage", None):
        try:
            usage = {
                "prompt_tokens": int(getattr(resp.usage, "input_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp.usage, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
            }
        except Exception:
            usage = {}

    latency_ms = (t1 - t0) * 1000.0
    return content, usage, latency_ms

# ============================ HIGH-LEVEL AKCJA ===========================

def _build_messages(memory: List[Dict[str, Any]], user_prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": st.session_state["chatbot_personality"]}]
    messages.extend({"role": m["role"], "content": m["content"]} for m in memory)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _is_gpt5(model_name: str) -> bool:
    return model_name.lower().startswith("gpt-5")


def chatbot_reply(
    user_prompt: str,
    memory: List[Dict[str, Any]],
    model_name: str,
    stream_output: bool,
    variant: str,
    log_to_langfuse: bool,
) -> Dict[str, Any]:
    messages = _build_messages(memory, user_prompt)
    redacted = redact_messages_for_logging(messages)

    trace_cm = lf_trace_ctx(
        name="chat-trace",
        user_id=st.session_state["user_id"],
        session_id=st.session_state["session_uuid"],
        metadata={
            "conversation_id": st.session_state.get("id"),
            "variant": variant,
            "app_version": APP_VERSION,
            "prompt_version": PROMPT_VERSION,
            "sdk_versions": SDK_VERSIONS,
        },
    ) if log_to_langfuse else nullcontext()

    with trace_cm:
        with lf_span_ctx(name="retrieve", metadata={"note": "placeholder RAG"}):
            pass

        with lf_generation_ctx(
            name=f"chat-completion-{variant}",
            model=model_name,
            input=redacted,
            metadata={"streaming": stream_output},
        ) as generation:
            try:
                is_gpt5 = _is_gpt5(model_name)

                # -------- STREAM --------
                if stream_output:
                    placeholder = st.empty()
                    if is_gpt5:
                        text, ttfb_ms, total_ms = _responses_stream_completion(
                            messages, model_name, placeholder, max_tokens_wanted=1024
                        )
                    else:
                        text, ttfb_ms, total_ms = _chat_stream_completion(
                            messages, model_name, placeholder, max_tokens=1024
                        )

                    # Responses stream zwykle nie ma usage → estymujemy
                    est_prompt = approx_prompt_tokens(messages)
                    est_completion = approx_tokens_from_text(text)
                    usage = {
                        "completion_tokens": est_completion,
                        "prompt_tokens": est_prompt,
                        "total_tokens": est_prompt + est_completion,
                    }

                    generation.update(
                        output=redact_pii(text),
                        usage_details={"input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"]},
                        metadata={"ttfb_ms": round(ttfb_ms, 2), "latency_ms": round(total_ms, 2)},
                    )

                    return {
                        "role": "assistant",
                        "content": text,
                        "usage": usage,
                        "model": model_name,
                        "variant": variant,
                    }

                # -------- NON-STREAM --------
                if is_gpt5:
                    content, usage, latency_ms = _responses_non_stream_completion(
                        messages, model_name, max_tokens_wanted=1024
                    )
                else:
                    content, usage, latency_ms = _chat_non_stream_completion(
                        messages, model_name, max_tokens=1024
                    )

                tokens_per_s = (
                    (usage.get("completion_tokens", 0) / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0
                )

                generation.update(
                    output=redact_pii(content),
                    usage_details={
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                    },
                    metadata={
                        "ttfb_ms": round(latency_ms, 2),
                        "latency_ms": round(latency_ms, 2),
                        "tokens_per_s": round(tokens_per_s, 2),
                    },
                )

                return {
                    "role": "assistant",
                    "content": content,
                    "usage": usage,
                    "model": model_name,
                    "variant": variant,
                }

            except Exception as e:
                st.error(f"❌ Błąd wywołania modelu: {e}")
                generation.update(output=str(e), metadata={"error": True})
                return {
                    "role": "assistant",
                    "content": "Przepraszam, spróbuj ponownie.",
                    "usage": {},
                    "model": model_name,
                    "variant": variant,
                }

# ============================ UI: STAN POCZĄTKOWY =======================

def init_state():
    if "pricing_map" not in st.session_state:
        st.session_state["pricing_map"] = load_pricing()
    st.session_state.setdefault("model_family", list(MODEL_FAMILIES.keys())[0])
    st.session_state.setdefault("model", MODEL_FAMILIES[st.session_state["model_family"]][0])
    st.session_state.setdefault("enable_streaming", True)
    st.session_state.setdefault("ab_mode", False)
    st.session_state.setdefault("ab_ratio_b", 0.5)
    st.session_state.setdefault("model_a", "gpt-4o")
    st.session_state.setdefault("model_b", "gpt-4o-mini")
    st.session_state.setdefault("fallback_model", "gpt-4o")
    st.session_state.setdefault("log_sampling", DEFAULT_SAMPLE_RATE)

def load_or_init_session():
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = "local-user"
    if "session_uuid" not in st.session_state:
        st.session_state["session_uuid"] = str(uuid4())

load_or_init_session()
load_current_conversation()
init_state()

# ============================ UI: NAGŁÓWEK ===============================

st.title(":classical_building: NaszGPT (Langfuse • Trace/Span • A/B • Streaming • TTFB)")

# Historia + feedback
for i, msg in enumerate(st.session_state.get("messages", [])):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            c1, c2 = st.columns(2)
            with c1:
                if st.button("👍", key=f"fb_up_{i}"):
                    with lf_generation_ctx(
                        name="feedback",
                        input={"message_index": i, "vote": "up"},
                        metadata={"feedback": "up"},
                    ) as g:
                        g.update(output="ok")
                    st.toast("Dzięki za feedback 👍")
            with c2:
                if st.button("👎", key=f"fb_down_{i}"):
                    with lf_generation_ctx(
                        name="feedback",
                        input={"message_index": i, "vote": "down"},
                        metadata={"feedback": "down"},
                    ) as g:
                        g.update(output="ok")
                    st.toast("Zapisano feedback 👎")

# ============================ INPUT + ROUTING ============================

prompt = st.chat_input("Napisz wiadomość.")
if prompt:
    # wybór modelu (single / A/B)
    chosen_model = st.session_state["model"]
    variant = "single"
    if st.session_state["ab_mode"]:
        use_b = random.random() < float(st.session_state["ab_ratio_b"])
        chosen_model = st.session_state["model_b"] if use_b else st.session_state["model_a"]
        variant = "B" if use_b else "A"

    if _is_gpt5(chosen_model):
        st.warning("Wybrano model z rodziny GPT-5. Jeżeli nie masz dostępu, nastąpi fallback.")

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # sampling – pierwsza odpowiedź zawsze logowana
    is_first_turn = sum(1 for m in st.session_state["messages"] if m["role"] == "assistant") == 0
    log_to_lf = True if (ALWAYS_LOG_FIRST_MESSAGE and is_first_turn) else (
        random.random() < float(st.session_state["log_sampling"]) 
    )

    with st.chat_message("assistant"):
        resp = chatbot_reply(
            prompt,
            memory=st.session_state["messages"][-10:],
            model_name=chosen_model,
            stream_output=st.session_state["enable_streaming"],
            variant=variant,
            log_to_langfuse=log_to_lf,
        )
        # Bez podwójnego renderowania: przy streamingu nic tu nie wyświetlamy
        if (not st.session_state["enable_streaming"]) and resp["content"]:
            st.markdown(resp["content"])

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": resp["content"],
            "usage": resp.get("usage", {}),
            "model": resp.get("model", chosen_model),
            "variant": resp.get("variant", variant),
        }
    )
    # zapis konwersacji
    path = DB_CONVERSATIONS_PATH / f"{st.session_state['id']}.json"
    conv = _safe_read_json(path, default={})
    conv.update({"messages": st.session_state.get("messages", [])})
    _safe_write_json(path, conv)

# ============================ SIDEBAR ===================================

with st.sidebar:
    st.subheader("Ustawienia modelu")

    def _fetch_models(prefix: str) -> List[str]:
        try:
            models_list = openai_client.models.list()
            return sorted([m.id for m in models_list.data if m.id.startswith(prefix)])
        except Exception:
            return []

    family_options = list(MODEL_FAMILIES.keys())
    st.session_state["model_family"] = st.selectbox(
        "Rodzina modelu",
        family_options,
        index=safe_index(family_options, st.session_state.get("model_family", family_options[0])),
    )

    if st.session_state["model_family"].startswith("GPT-5"):
        family_models = _fetch_models("gpt-5") or ["gpt-5"]
    else:
        family_models = _fetch_models("gpt-4o") or ["gpt-4o", "gpt-4o-mini"]

    if st.session_state.get("model") not in family_models:
        st.session_state["model"] = family_models[0]

    st.session_state["model"] = st.selectbox(
        "Model (single mode)",
        family_models,
        index=safe_index(family_models, st.session_state["model"]),
        key="single_model_select",
    )

    st.session_state["enable_streaming"] = st.checkbox(
        "🔁 Streaming odpowiedzi",
        value=st.session_state.get("enable_streaming", True),
    )

    st.markdown("---")
    st.subheader("A/B testing")
    st.session_state["ab_mode"] = st.checkbox("Włącz A/B switch", value=st.session_state.get("ab_mode", False))

    all_models = (_fetch_models("gpt-4o") or ["gpt-4o", "gpt-4o-mini"]) + (_fetch_models("gpt-5") or ["gpt-5"])

    if st.session_state["ab_mode"]:
        if st.session_state.get("model_a") not in all_models:
            st.session_state["model_a"] = all_models[0]
        if st.session_state.get("model_b") not in all_models:
            st.session_state["model_b"] = all_models[1] if len(all_models) > 1 else all_models[0]

        st.session_state["model_a"] = st.selectbox(
            "Model A",
            all_models,
            index=safe_index(all_models, st.session_state["model_a"]),
        )
        st.session_state["model_b"] = st.selectbox(
            "Model B",
            all_models,
            index=safe_index(all_models, st.session_state["model_b"]),
        )
        st.session_state["ab_ratio_b"] = st.slider(
            "Udział wariantu B",
            0.0, 1.0,
            float(st.session_state.get("ab_ratio_b", 0.5)),
            0.05,
        )

    if st.session_state.get("fallback_model") not in all_models:
        st.session_state["fallback_model"] = all_models[0]
    st.session_state["fallback_model"] = st.selectbox(
        "Fallback model",
        all_models,
        index=safe_index(all_models, st.session_state["fallback_model"]),
    )

    st.markdown("---")
    st.subheader("Użytkownik / Sesja")
    st.session_state["user_id"] = st.text_input("User ID", value=st.session_state.get("user_id", "local-user"))
    st.caption(f"Session ID: `{st.session_state['session_uuid']}`")

    st.markdown("---")
    st.subheader("Logowanie Langfuse")
    st.session_state["log_sampling"] = st.slider(
        "Sampling do Langfuse",
        0.0, 1.0,
        float(st.session_state.get("log_sampling", 1.0)),
        0.05,
    )
    st.caption("Pierwsza interakcja jest zawsze logowana.")

    st.markdown("---")
    st.subheader("Cennik i koszty")
    total_cost_usd = 0.0
    for m in st.session_state.get("messages", []):
        usage = m.get("usage") or {}
        model_used = m.get("model") or st.session_state["model"]
        total_cost_usd += compute_cost_usd(usage, model_used)

    c0, c1 = st.columns(2)
    with c0:
        st.metric("Koszt (USD)", f"${total_cost_usd:.4f}")
    with c1:
        st.metric("Koszt (PLN)", f"{total_cost_usd * USD_TO_PLN:.4f}")

    st.write("**Stawki modeli (USD / token)**")

    for fam_name, prefixes in MODEL_FAMILIES.items():
        with st.expander(fam_name, expanded=False):
            # Zbierz modele w tej rodzinie i zdeduplikuj
            fam_models = sorted({m for m in all_models if any(m.startswith(pfx) for pfx in prefixes)})
            if not fam_models:
                st.caption("Brak modeli w tej rodzinie.")
                continue

            for m in fam_models:
                pr = get_pricing_for_model(m)
                col1, col2 = st.columns(2)

                # KLUCZE UNIKALNE: rodzina + nazwa modelu
                key_in = f"price_in::{fam_name}::{m}"
                key_out = f"price_out::{fam_name}::{m}"

                with col1:
                    pr["input_tokens"] = st.number_input(
                        f"{m} · input",
                        min_value=0.0,
                        value=float(pr.get("input_tokens") or 0.0),
                        step=0.000001,
                        format="%.6f",
                        key=key_in,
                    )
                with col2:
                    pr["output_tokens"] = st.number_input(
                        f"{m} · output",
                        min_value=0.0,
                        value=float(pr.get("output_tokens") or 0.0),
                        step=0.000001,
                        format="%.6f",
                        key=key_out,
                    )

                st.session_state["pricing_map"][m] = pr

    if st.button("💾 Zapisz stawki"):
        save_pricing(st.session_state["pricing_map"])
        st.success("Zapisano pricing do db/pricing.json")

    st.markdown("---")
    st.subheader("Konwersacje")
    st.session_state["name"] = st.text_input(
        "Nazwa konwersacji",
        value=st.session_state["name"],
        key="new_conversation_name",
        on_change=save_current_conversation_name,
    )
    st.session_state["chatbot_personality"] = st.text_area(
        "Osobowość chatbota",
        max_chars=1000,
        height=160,
        value=st.session_state["chatbot_personality"],
        key="new_chatbot_personality",
        on_change=save_current_conversation_personality,
    )
    if st.button("Nowa konwersacja"):
        create_new_conversation()

    for conv in sorted(list_conversations(), key=lambda x: x["id"], reverse=True)[:5]:
        c0, c1 = st.columns([10, 3])
        with c0:
            st.write(conv["name"])
        with c1:
            if st.button(
                "załaduj",
                key=f"load_{conv['id']}",
                disabled=(conv['id'] == st.session_state['id']),
            ):
                switch_conversation(conv["id"])  # bez st.rerun() w funkcji
                st.rerun()  # tu jest OK (poza callbackiem)
