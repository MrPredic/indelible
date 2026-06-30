"""Tests for indelible.collectors.*"""
import sys
from unittest.mock import MagicMock

import pytest

from indelible.collectors import Collector
from indelible.collectors.anchor_drift import AnchorDriftCollector
from indelible.collectors.embedding_profile import EmbeddingProfileCollector
from indelible.collectors.latency import LatencyCollector
from indelible.collectors.refusal import RefusalCollector
from indelible.collectors.tool_distribution import ToolDistributionCollector
from indelible.collectors.tool_schema_hash import ToolSchemaHashCollector
from indelible.collectors.vocab_entropy import VocabEntropyCollector
from indelible.types import Signal

# helpers
_CALL = dict(inputs=["What is 2+2?"], anchor_text="You are a helpful assistant.", tools_called=[])


def _collect(collector, outputs, tools_called=None):
    tc = tools_called if tools_called is not None else [[] for _ in outputs]
    return collector.collect(
        outputs=outputs,
        inputs=_CALL["inputs"],
        anchor_text=_CALL["anchor_text"],
        tools_called=tc,
    )


# --- Protocol ---

def test_collector_protocol_satisfied():
    assert isinstance(RefusalCollector(), Collector)
    assert isinstance(LatencyCollector(), Collector)
    assert isinstance(VocabEntropyCollector(), Collector)
    assert isinstance(ToolDistributionCollector(), Collector)
    assert isinstance(ToolSchemaHashCollector([]), Collector)


# --- RefusalCollector ---

def test_refusal_no_refusal():
    s = _collect(RefusalCollector(), ["The answer is 42.", "Here you go."])
    assert isinstance(s, Signal)
    assert s.name == "refusal_rate"
    assert s.value == 0.0


def test_refusal_full_refusal():
    s = _collect(RefusalCollector(), ["I cannot help with that.", "I'm not able to do this."])
    assert s.value == 1.0


def test_refusal_partial():
    s = _collect(RefusalCollector(), ["Sure!", "I cannot do that."])
    assert s.value == pytest.approx(0.5)


def test_refusal_empty_outputs():
    s = _collect(RefusalCollector(), [])
    assert s.value == 0.0


def test_refusal_default_tolerance_is_010():
    """P1-1: refusal_rate default tolerance must be 0.10, not the 0.05 floor."""
    s = _collect(RefusalCollector(), ["I cannot help."])
    assert s.tolerance == 0.10


def test_refusal_custom_patterns():
    """P1-9: RefusalCollector accepts custom patterns for non-English agents."""
    fr_patterns = [r"je ne peux pas", r"désolé.*impossible"]
    c = RefusalCollector(patterns=fr_patterns)
    # Default English phrases should NOT match anymore
    s = _collect(c, ["I cannot help.", "Je ne peux pas faire ça."])
    assert s.value == pytest.approx(0.5)


def test_refusal_empty_patterns_raises():
    with pytest.raises(ValueError, match="non-empty"):
        RefusalCollector(patterns=[])


# --- LatencyCollector ---

def test_latency_values():
    c = LatencyCollector()
    c.set_latencies([0.1, 0.2, 0.3, 0.4, 0.5])
    s = _collect(c, ["ok"])
    assert s.name == "latency"
    assert s.value == pytest.approx(0.3)
    assert s.p50 is not None
    assert s.p95 is not None


def test_latency_empty():
    s = _collect(LatencyCollector(), ["ok"])
    assert s.value == 0.0
    assert s.p50 is None


def test_latency_single():
    c = LatencyCollector()
    c.set_latencies([1.5])
    s = _collect(c, ["ok"])
    assert s.value == pytest.approx(1.5)


def test_latency_default_tolerance_is_030():
    """P1-1: latency default tolerance must be 0.30 (300ms), the only value
    that doesn't breach-storm in CI for slow providers."""
    c = LatencyCollector()
    c.set_latencies([0.1])
    s = _collect(c, ["ok"])
    assert s.tolerance == 0.30


def test_vocab_entropy_default_tolerance_is_050():
    s = _collect(VocabEntropyCollector(), ["word word word"])
    assert s.tolerance == 0.50


def test_tool_distribution_default_tolerance_is_010():
    s = _collect(ToolDistributionCollector(), ["ok"], tools_called=[[]])
    assert s.tolerance == 0.10


# --- VocabEntropyCollector ---

def test_entropy_single_token():
    s = _collect(VocabEntropyCollector(), ["word word word"])
    assert s.name == "vocab_entropy"
    assert s.value == pytest.approx(0.0)


def test_entropy_diverse():
    s = _collect(VocabEntropyCollector(), ["apple banana cherry date elderberry"])
    assert s.value > 0.0


def test_entropy_empty():
    s = _collect(VocabEntropyCollector(), [])
    assert s.value == 0.0


# --- ToolDistributionCollector ---

def test_tool_distribution_no_calls():
    s = _collect(ToolDistributionCollector(), ["ok"], tools_called=[[]])
    assert s.name == "tool_distribution"
    assert s.value == 0.0


def test_tool_distribution_single_tool():
    s = _collect(ToolDistributionCollector(), ["ok", "ok"], tools_called=[["search"], ["search"]])
    assert s.value == 1.0
    assert s.distribution == {"search": 1.0}


def test_tool_distribution_multiple():
    s = _collect(ToolDistributionCollector(), ["ok", "ok"],
                 tools_called=[["search", "read"], ["search"]])
    assert s.value == 2.0
    assert abs(s.distribution["search"] - 2 / 3) < 0.01


# --- ToolSchemaHashCollector ---

def test_schema_hash_deterministic():
    tools = [{"name": "read_file", "description": "reads"}]
    c1 = ToolSchemaHashCollector(tools)
    c2 = ToolSchemaHashCollector(tools)
    assert c1.schema_hash_str == c2.schema_hash_str


def test_schema_hash_changes_with_schema():
    c1 = ToolSchemaHashCollector([{"name": "a"}])
    c2 = ToolSchemaHashCollector([{"name": "b"}])
    assert c1.schema_hash_str != c2.schema_hash_str


def test_schema_hash_collect_returns_signal():
    c = ToolSchemaHashCollector([{"name": "read_file"}])
    s = _collect(c, ["ok"])
    assert s.name == "tool_schema_hash"
    assert s.value == 0.0  # value is now a placeholder; verify uses `digest`
    assert s.distribution is not None


def test_schema_hash_publishes_digest():
    """tool_schema_hash must publish the full hex SHA-256 in `digest` so
    verify can do exact-match comparison instead of numeric tolerance.
    Was a real-bug fix: review found numeric value could false-pass on
    tool schema changes that landed within tolerance × 1e9."""
    c = ToolSchemaHashCollector([{"name": "read_file"}])
    s = _collect(c, ["ok"])
    assert s.digest is not None
    assert len(s.digest) == 64  # SHA-256 hex
    assert s.digest == c.schema_hash_str


def test_schema_hash_digest_differs_per_schema():
    s1 = _collect(ToolSchemaHashCollector([{"name": "a"}]), ["ok"])
    s2 = _collect(ToolSchemaHashCollector([{"name": "b"}]), ["ok"])
    assert s1.digest != s2.digest


def test_schema_hash_empty_tools():
    c = ToolSchemaHashCollector([])
    s = _collect(c, ["ok"])
    assert isinstance(s, Signal)


# --- EmbeddingProfileCollector + AnchorDriftCollector (graceful skip) ---

def test_embedding_profile_skip_without_extras(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    s = _collect(EmbeddingProfileCollector(), ["hello world"])
    assert s.name == "embedding_profile"
    assert s.value == 0.0


def test_anchor_drift_skip_without_extras(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    s = _collect(AnchorDriftCollector(), ["hello"])
    assert s.name == "anchor_drift"
    assert s.value == 0.0


def test_embedding_profile_needs_extras():
    assert "deep" in EmbeddingProfileCollector.needs_extras


def test_anchor_drift_needs_extras():
    assert "deep" in AnchorDriftCollector.needs_extras


# --- embedding happy-path via mocked sentence_transformers ---

try:
    import numpy as np  # type: ignore[import-not-found]
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


def _make_st_mock():
    mock_st = MagicMock()
    mock_model = MagicMock()
    n = 3
    embs = np.random.rand(n, 32).astype("float32")
    mock_model.encode.return_value = embs
    mock_model.cos_sim.side_effect = lambda a, b: MagicMock(__float__=lambda self: 0.9)
    mock_st.SentenceTransformer.return_value = mock_model
    # util.cos_sim returns float-like
    mock_st.util.cos_sim.return_value = 0.9
    return mock_st


@pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed (deep extra)")
def test_embedding_profile_happy_path(monkeypatch):
    mock_st = _make_st_mock()
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st)
    monkeypatch.setitem(sys.modules, "numpy", np)
    outputs = ["The cat sat.", "A dog ran.", "Birds flew."]
    s = _collect(EmbeddingProfileCollector(), outputs)
    assert s.name == "embedding_profile"
    assert s.value >= 0.0


@pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed (deep extra)")
def test_anchor_drift_happy_path(monkeypatch):
    mock_st = _make_st_mock()
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st)
    outputs = ["The cat sat.", "A dog ran."]
    s = _collect(AnchorDriftCollector(), outputs)
    assert s.name == "anchor_drift"
    assert s.value >= 0.0


def test_anchor_drift_empty_outputs(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    s = _collect(AnchorDriftCollector(), [])
    assert s.value == 0.0
