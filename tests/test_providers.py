"""Tests for indelible.providers.*"""
import pytest
from unittest.mock import patch, MagicMock

from indelible.providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAICompatProvider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderServerError,
    get_provider,
    raise_for_status,
)


def _mock_resp(status: int, data: dict) -> MagicMock:
    m = MagicMock()
    m.status_code = status
    m.json.return_value = data
    m.text = str(data)
    return m


# --- OpenAICompatProvider ---

def test_openai_basic():
    resp = _mock_resp(200, {"choices": [{"message": {"content": "Hello"}}]})
    with patch("indelible.providers.openai_compat.httpx.post", return_value=resp):
        p = OpenAICompatProvider("https://api.example.com", "gpt-4o", "key-x")
        content, tools, latency = p.complete("sys", "hi")
    assert content == "Hello"
    assert tools == []
    assert latency >= 0.0


def test_openai_content_zero_string():
    """Content='0' must not be coerced to empty string."""
    resp = _mock_resp(200, {"choices": [{"message": {"content": "0"}}]})
    with patch("indelible.providers.openai_compat.httpx.post", return_value=resp):
        content, _, _ = OpenAICompatProvider("https://api.example.com", "gpt-4o").complete("s", "u")
    assert content == "0"


def test_openai_content_null_becomes_empty():
    """content=null in response must become empty string, not crash."""
    resp = _mock_resp(200, {"choices": [{"message": {"content": None}}]})
    with patch("indelible.providers.openai_compat.httpx.post", return_value=resp):
        content, _, _ = OpenAICompatProvider("https://api.example.com", "gpt-4o").complete("s", "u")
    assert content == ""


def test_openai_tool_calls():
    resp = _mock_resp(200, {
        "choices": [{"message": {
            "content": "ok",
            "tool_calls": [
                {"function": {"name": "search"}},
                {"function": {"name": "read_file"}},
            ],
        }}]
    })
    with patch("indelible.providers.openai_compat.httpx.post", return_value=resp):
        p = OpenAICompatProvider("https://api.example.com", "gpt-4o")
        _, tools, _ = p.complete("sys", "hi")
    assert tools == ["search", "read_file"]


def test_openai_no_api_key_no_auth_header():
    resp = _mock_resp(200, {"choices": [{"message": {"content": "ok"}}]})
    with patch("indelible.providers.openai_compat.httpx.post", return_value=resp) as mock_post:
        OpenAICompatProvider("https://api.example.com", "model").complete("s", "u")
    _, kwargs = mock_post.call_args
    headers = kwargs.get("headers", mock_post.call_args[1].get("headers", {}))
    assert "Authorization" not in headers


def test_openai_http_error():
    resp = _mock_resp(500, {})
    with patch("indelible.providers.openai_compat.httpx.post", return_value=resp):
        # Existing RuntimeError contract preserved (typed exceptions subclass it)
        with pytest.raises(RuntimeError, match="500"):
            OpenAICompatProvider("https://api.example.com", "m").complete("s", "u")


def test_openai_500_raises_typed_server_error():
    """P1-8: 5xx must raise ProviderServerError (still a RuntimeError subclass)."""
    resp = _mock_resp(503, {})
    with patch("indelible.providers.openai_compat.httpx.post", return_value=resp):
        with pytest.raises(ProviderServerError) as ei:
            OpenAICompatProvider("https://api.example.com", "m").complete("s", "u")
    assert ei.value.status_code == 503


def test_openai_401_raises_typed_auth_error():
    resp = _mock_resp(401, {})
    with patch("indelible.providers.openai_compat.httpx.post", return_value=resp):
        with pytest.raises(ProviderAuthError) as ei:
            OpenAICompatProvider("https://api.example.com", "m").complete("s", "u")
    assert ei.value.status_code == 401


def test_openai_429_raises_typed_rate_limit():
    resp = _mock_resp(429, {})
    with patch("indelible.providers.openai_compat.httpx.post", return_value=resp):
        with pytest.raises(ProviderRateLimitError) as ei:
            OpenAICompatProvider("https://api.example.com", "m").complete("s", "u")
    assert ei.value.status_code == 429


def test_typed_exceptions_are_runtime_error_subclasses():
    """Backward compat: existing `except RuntimeError` must still catch."""
    assert issubclass(ProviderError, RuntimeError)
    assert issubclass(ProviderAuthError, ProviderError)
    assert issubclass(ProviderRateLimitError, ProviderError)
    assert issubclass(ProviderServerError, ProviderError)


def test_raise_for_status_200_is_noop():
    raise_for_status(200, "ignored")  # must not raise


def test_raise_for_status_unknown_status_raises_base():
    with pytest.raises(ProviderError) as ei:
        raise_for_status(418, "I'm a teapot")
    assert ei.value.status_code == 418
    # Must NOT be one of the more specific subclasses
    assert not isinstance(ei.value, (ProviderAuthError, ProviderRateLimitError, ProviderServerError))


# --- AnthropicProvider ---

def test_anthropic_basic():
    resp = _mock_resp(200, {"content": [{"type": "text", "text": "Hello Claude"}]})
    with patch("indelible.providers.anthropic.httpx.post", return_value=resp):
        p = AnthropicProvider("claude-opus-4-7", "sk-ant-xxx")
        content, tools, latency = p.complete("sys", "hi")
    assert content == "Hello Claude"
    assert tools == []
    assert latency >= 0.0


def test_anthropic_tool_use_blocks():
    resp = _mock_resp(200, {"content": [
        {"type": "text", "text": "I'll search for that."},
        {"type": "tool_use", "name": "search"},
        {"type": "tool_use", "name": "read_file"},
    ]})
    with patch("indelible.providers.anthropic.httpx.post", return_value=resp):
        content, tools, _ = AnthropicProvider("claude-opus-4-7", "key").complete("s", "u")
    assert content == "I'll search for that."
    assert tools == ["search", "read_file"]


def test_anthropic_tool_first_text_second():
    """Text block not at index 0 — must still extract it."""
    resp = _mock_resp(200, {"content": [
        {"type": "tool_use", "name": "think"},
        {"type": "text", "text": "Done."},
    ]})
    with patch("indelible.providers.anthropic.httpx.post", return_value=resp):
        content, tools, _ = AnthropicProvider("claude-opus-4-7", "key").complete("s", "u")
    assert content == "Done."
    assert tools == ["think"]


def test_anthropic_multiple_text_blocks_concatenated():
    """Anthropic can return text → tool_use → text. All text must be captured."""
    resp = _mock_resp(200, {"content": [
        {"type": "text",     "text": "Let me search. "},
        {"type": "tool_use", "name": "search"},
        {"type": "text",     "text": "Found it."},
    ]})
    with patch("indelible.providers.anthropic.httpx.post", return_value=resp):
        content, tools, _ = AnthropicProvider("claude-opus-4-7", "key").complete("s", "u")
    assert content == "Let me search. Found it."
    assert tools == ["search"]


def test_anthropic_http_error():
    resp = _mock_resp(401, {})
    with patch("indelible.providers.anthropic.httpx.post", return_value=resp):
        with pytest.raises(RuntimeError, match="401"):
            AnthropicProvider("claude-opus-4-7", "bad-key").complete("s", "u")


# --- OllamaProvider ---

def test_ollama_basic():
    resp = _mock_resp(200, {"message": {"content": "Bonjour"}})
    with patch("indelible.providers.ollama.httpx.post", return_value=resp):
        p = OllamaProvider("ollama/llama3.3")
        content, tools, latency = p.complete("sys", "hi")
    assert content == "Bonjour"
    assert tools == []
    assert p.model == "llama3.3"   # prefix stripped


def test_ollama_prefix_stripped():
    p = OllamaProvider("ollama/qwen2.5:14b")
    assert p.model == "qwen2.5:14b"


def test_ollama_http_error():
    resp = _mock_resp(500, {})
    with patch("indelible.providers.ollama.httpx.post", return_value=resp):
        with pytest.raises(RuntimeError, match="500"):
            OllamaProvider("ollama/m").complete("s", "u")


# --- get_provider routing ---

@pytest.mark.parametrize("model,expected_cls", [
    ("claude-opus-4-7", AnthropicProvider),
    ("claude-haiku-4-5", AnthropicProvider),
    ("anthropic/claude-3", AnthropicProvider),
    ("ollama/llama3.3", OllamaProvider),
    ("gpt-4o", OpenAICompatProvider),
    ("llama-3.3-70b-versatile", OpenAICompatProvider),
])
def test_get_provider_routing(model, expected_cls):
    p = get_provider(model, "https://api.example.com", "key")
    assert isinstance(p, expected_cls)


@pytest.mark.live
def test_ollama_live():
    import os
    import httpx as _httpx
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        _httpx.get(f"{host}/api/tags", timeout=2)
    except Exception:
        pytest.skip("Ollama not reachable")
    p = OllamaProvider("ollama/qwen2.5:7b", host=host)
    content, tools, latency = p.complete("You are a helpful assistant.", "Say: hello")
    assert content
    assert latency > 0
