"""IndelibleConfig — loads/saves agent attestation configuration."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import tomllib  # stdlib Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w


class IndelibleConfig:
    """Agent configuration for behavioral attestation.

    TOML format::

        [agent]
        name = "my-coding-agent"
        system_prompt = "You are a helpful assistant."
        model = "claude-opus-4-7"
        provider_url = "https://api.anthropic.com"
        tolerance_default = 0.05

        [[tools]]
        name = "read_file"
        description = "Reads a file"
    """

    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        tools: List[Dict[str, Any]],
        model: str,
        provider_url: str,
        tolerance_default: float = 0.05,
        maintainer: str = "",
        refusal_patterns: Optional[List[str]] = None,
        temperature: float = 0.0,
    ) -> None:
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.tools: List[Dict[str, Any]] = copy.deepcopy(tools)
        self.model = model
        self.provider_url = provider_url
        self.tolerance_default = tolerance_default
        self.maintainer = maintainer
        # 0.0 = deterministic sampling. The whole tool compares run A to run B,
        # so a non-zero default would inject sampling noise straight into every
        # signal and manufacture false breaches.
        self.temperature = float(temperature)
        # None = use collector defaults; explicit list = override
        self.refusal_patterns: Optional[List[str]] = (
            list(refusal_patterns) if refusal_patterns is not None else None
        )

    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> IndelibleConfig:
        """Load config from a TOML file.

        Raises ValueError if required fields are missing.
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)

        if "agent" not in data:
            raise ValueError("Missing required section [agent] in config TOML")

        agent = data["agent"]
        for field in ("name", "system_prompt", "model", "provider_url"):
            if field not in agent:
                raise ValueError(f"Missing required field 'agent.{field}' in config TOML")

        refusal_section = data.get("refusal", {})
        refusal_patterns = refusal_section.get("patterns") if isinstance(refusal_section, dict) else None

        return cls(
            agent_name=agent["name"],
            system_prompt=agent["system_prompt"],
            tools=data.get("tools", []),
            model=agent["model"],
            provider_url=agent["provider_url"],
            tolerance_default=agent.get("tolerance_default", 0.05),
            maintainer=agent.get("maintainer", ""),
            refusal_patterns=refusal_patterns,
            temperature=agent.get("temperature", 0.0),
        )

    def to_toml(self, path: Union[str, Path]) -> None:
        """Write config to a TOML file."""
        agent: Dict[str, Any] = {
            "name": self.agent_name,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "provider_url": self.provider_url,
            "tolerance_default": self.tolerance_default,
            "temperature": self.temperature,
        }
        if self.maintainer:
            agent["maintainer"] = self.maintainer
        data: Dict[str, Any] = {"agent": agent}
        if self.tools:
            data["tools"] = self.tools
        if self.refusal_patterns is not None:
            data["refusal"] = {"patterns": self.refusal_patterns}
        with open(path, "wb") as f:
            tomli_w.dump(data, f)

    def to_dict(self) -> dict:
        # NOTE: `maintainer` is intentionally NOT part of the canonical config
        # dict — it identifies *who* attested, not *what* was attested.
        # Including it would cause a maintainer change (new team member,
        # email change) to invalidate every existing fingerprint.
        # `refusal_patterns` IS part of canonical: different patterns = different
        # refusal_rate baseline = different "what was attested".
        d: dict = {
            "agent_name": self.agent_name,
            "system_prompt": self.system_prompt,
            "tools": copy.deepcopy(self.tools),
            "model": self.model,
            "provider_url": self.provider_url,
            "tolerance_default": self.tolerance_default,
            # temperature IS canonical: it changes the sampling distribution,
            # hence the attested vocab_entropy/refusal baseline.
            "temperature": self.temperature,
        }
        if self.refusal_patterns is not None:
            d["refusal_patterns"] = list(self.refusal_patterns)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> IndelibleConfig:
        return cls(
            agent_name=d["agent_name"],
            system_prompt=d["system_prompt"],
            tools=d.get("tools", []),
            model=d["model"],
            provider_url=d["provider_url"],
            tolerance_default=d.get("tolerance_default", 0.05),
            maintainer=d.get("maintainer", ""),
            refusal_patterns=d.get("refusal_patterns"),
            temperature=d.get("temperature", 0.0),
        )

    def canonical_hash(self) -> str:
        """SHA-256 of the canonical JSON representation (sorted keys)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndelibleConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"IndelibleConfig(agent_name={self.agent_name!r}, model={self.model!r})"
