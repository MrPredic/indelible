"""Tests for bedrock_attest.providers.*"""
import pytest
from unittest.mock import patch, MagicMock

from bedrock_attest.providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAICompatProvider,
    get_provider,
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
    with patch("bedrock_attest.providers.openai_compat.httpx.post", return_value=resp):
        p = OpenAICompatProvider("https://api.example.com", "gpt-4o", "key-x")
        content, tools, latency = p.complete("sys", "hi")
    assert content == "Hello"
    assert tools == []
    assert latency >= 0.0


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
    with patch("bedrock_attest.providers.openai_compat.httpx.post", return_value=resp):
        p = OpenAICompatProvider("https://api.example.com", "gpt-4o")
        _, tools, _ = p.complete("sys", "hi")
    assert tools == ["search", "read_file"]


def test_openai_no_api_key_no_auth_header():
    resp = _mock_resp(200, {"choices": [{"message": {"content": "ok"}}]})
    with patch("bedrock_attest.providers.openai_compat.httpx.post", return_value=resp) as mock_post:
        OpenAICompatProvider("https://api.example.com", "model").complete("s", "u")
    _, kwargs = mock_post.call_args
    headers = kwargs.get("headers", mock_post.call_args[1].get("headers", {}))
    assert "Authorization" not in headers


def test_openai_http_error():
    resp = _mock_resp(500, {})
    with patch("bedrock_attest.providers.openai_compat.httpx.post", return_value=resp):
        with pytest.raises(RuntimeError, match="500"):
            OpenAICompatProvider("https://api.example.com", "m").complete("s", "u")


# --- AnthropicProvider ---

def test_anthropic_basic():
    resp = _mock_resp(200, {"content": [{"type": "text", "text": "Hello Claude"}]})
    with patch("bedrock_attest.providers.anthropic.httpx.post", return_value=resp):
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
    with patch("bedrock_attest.providers.anthropic.httpx.post", return_value=resp):
        content, tools, _ = AnthropicProvider("claude-opus-4-7", "key").complete("s", "u")
    assert content == "I'll search for that."
    assert tools == ["search", "read_file"]


def test_anthropic_tool_first_text_second():
    """Text block not at index 0 — must still extract it."""
    resp = _mock_resp(200, {"content": [
        {"type": "tool_use", "name": "think"},
        {"type": "text", "text": "Done."},
    ]})
    with patch("bedrock_attest.providers.anthropic.httpx.post", return_value=resp):
        content, tools, _ = AnthropicProvider("claude-opus-4-7", "key").complete("s", "u")
    assert content == "Done."
    assert tools == ["think"]


def test_anthropic_http_error():
    resp = _mock_resp(401, {})
    with patch("bedrock_attest.providers.anthropic.httpx.post", return_value=resp):
        with pytest.raises(RuntimeError, match="401"):
            AnthropicProvider("claude-opus-4-7", "bad-key").complete("s", "u")


# --- OllamaProvider ---

def test_ollama_basic():
    resp = _mock_resp(200, {"message": {"content": "Bonjour"}})
    with patch("bedrock_attest.providers.ollama.httpx.post", return_value=resp):
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
    with patch("bedrock_attest.providers.ollama.httpx.post", return_value=resp):
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
