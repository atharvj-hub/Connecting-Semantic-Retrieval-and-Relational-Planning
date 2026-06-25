"""
opik_ollama.py  —  teach Opik to read token usage from ChatOllama runs.

WHY THIS EXISTS
Opik ships token-usage extractors for OpenAI, Anthropic, Google, Groq, Bedrock,
etc. — but none for Ollama. So out of the box, ChatOllama spans in the Opik UI
show no token counts (the latency + step tree work, just not tokens).

The data is actually right there: ChatOllama reports usage in the standard
LangChain shape ({input_tokens, output_tokens, total_tokens}) and tags every run
with `ls_provider="ollama"` in its metadata. Opik's extraction is gated purely on
a provider being *recognised* — if no extractor says "this run is mine", the
search for usage never runs. So all we need is a tiny extractor that:
  1. recognises an Ollama run via that ls_provider tag, and
  2. reuses Opik's OWN search helper to pull the standard usage out.

We register it by appending to Opik's extractor registry at import time. This
touches an Opik internal, so it's best-effort and fully guarded: if Opik's layout
changes in a future version, register() just returns False and tracing carries on
exactly as before (tokens blank again, nothing breaks).

Note: cost stays $0 / unset — Ollama is local and free, and isn't in Opik's cost
model. This restores the token COUNTS, which is what we actually wanted.
"""

import logging

logger = logging.getLogger(__name__)

_REGISTERED = False


def register():
    """Append an Ollama usage extractor to Opik's registry (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        from opik import llm_usage
        from opik.integrations.langchain.provider_usage_extractors import (
            usage_extractor,
            provider_usage_extractor_protocol,
        )
        from opik.integrations.langchain.provider_usage_extractors.langchain_run_helpers import (  # noqa: E501
            helpers as lc_helpers,
        )
    except Exception as exc:                       # Opik internals moved/missing
        logger.debug("Skipping Ollama usage extractor: %s", exc)
        return False

    class OllamaUsageExtractor(
        provider_usage_extractor_protocol.ProviderUsageExtractorProtocol
    ):
        PROVIDER = "ollama"

        def is_provider_run(self, run_dict):
            try:
                md = lc_helpers.try_get_ls_metadata(run_dict)
                return md is not None and md.provider == "ollama"
            except Exception:
                return False

        def get_llm_usage_info(self, run_dict):
            usage = None
            model = None
            try:
                # candidate_keys=None -> search for the standard LangChain usage
                # keys; returns a LangChainUsage when found.
                found = lc_helpers.try_to_get_usage_by_search(run_dict, None)
                if found is not None:
                    openai_like = found.map_to_openai_completions_usage()
                    usage = llm_usage.OpikUsage.from_openai_completions_dict(openai_like)
                md = lc_helpers.try_get_ls_metadata(run_dict)
                if md is not None:
                    model = md.model
            except Exception:
                logger.debug("Ollama usage extraction failed", exc_info=True)
            return llm_usage.LLMUsageInfo(provider="ollama", model=model, usage=usage)

    reg = usage_extractor._REGISTERED_PROVIDER_USAGE_EXTRACTORS
    if not any(type(e).__name__ == "OllamaUsageExtractor" for e in reg):
        reg.append(OllamaUsageExtractor())
    _REGISTERED = True
    return True


# --- timing breakdown --------------------------------------------------------
# Ollama returns, per call, a nanosecond timing split that Opik's tracer drops:
#   load_duration       — time to load the model into memory
#   prompt_eval_duration— time spent reading/processing the INPUT prompt
#   eval_duration       — time spent GENERATING the output
# That's the closest thing to a "where did the work go" breakdown we can get for a
# local model. We surface it (plus generation tokens/sec) into each LLM span's
# metadata by subclassing the tracer's end-of-span hook. Fully guarded: any
# failure falls back to the stock tracer with no timing, nothing breaks.

def _find_dict_with(data, key):
    if isinstance(data, dict):
        if key in data:
            return data
        for v in list(data.values())[::-1]:
            r = _find_dict_with(v, key)
            if r is not None:
                return r
    elif isinstance(data, (list, tuple)):
        for it in reversed(data):
            r = _find_dict_with(it, key)
            if r is not None:
                return r
    return None


def _ollama_timing(run_dict):
    d = _find_dict_with(run_dict.get("outputs"), "eval_duration")
    if not d:
        return None

    def ms(ns):
        return round(ns / 1e6, 1) if isinstance(ns, (int, float)) else None

    out = {}
    if d.get("load_duration") is not None:
        out["model_load_ms"] = ms(d["load_duration"])
    if d.get("prompt_eval_duration") is not None:
        out["input_processing_ms"] = ms(d["prompt_eval_duration"])
    if d.get("eval_duration") is not None:
        out["generation_ms"] = ms(d["eval_duration"])
    if d.get("total_duration") is not None:
        out["total_ms"] = ms(d["total_duration"])
    ec, ed = d.get("eval_count"), d.get("eval_duration")
    if ec and ed:
        out["generation_tokens_per_sec"] = round(ec / (ed / 1e9), 1)
    return {"ollama_timing": out} if out else None


def make_tracer(project_name):
    """Return an OpikTracer that also logs Ollama's per-call timing breakdown.
    Falls back to a stock OpikTracer (or None) if anything is unavailable."""
    register()
    try:
        from opik.integrations.langchain import OpikTracer
    except Exception:
        return None

    try:
        class _TimingTracer(OpikTracer):
            def _process_end_span(self, run):
                try:
                    sd = self._span_data_map.get(run.id)
                    if sd is not None:
                        extra = _ollama_timing(run.dict())
                        if extra:
                            sd.metadata = {**(sd.metadata or {}), **extra}
                except Exception:
                    logger.debug("Failed to add Ollama timing", exc_info=True)
                return super()._process_end_span(run)

        return _TimingTracer(project_name=project_name)
    except Exception:
        logger.debug("Timing tracer unavailable, using stock OpikTracer", exc_info=True)
        return OpikTracer(project_name=project_name)
