"""BedrockConfig — loads/saves agent attestation configuration."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Union

try:
    import tomllib  # stdlib Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w


class BedrockConfig:
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
    ) -> None:
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.tools: List[Dict[str, Any]] = copy.deepcopy(tools)
        self.model = model
        self.provider_url = provider_url
        self.tolerance_default = tolerance_default

    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> BedrockConfig:
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

        return cls(
            agent_name=agent["name"],
            system_prompt=agent["system_prompt"],
            tools=data.get("tools", []),
            model=agent["model"],
            provider_url=agent["provider_url"],
            tolerance_default=agent.get("tolerance_default", 0.05),
        )

    def to_toml(self, path: Union[str, Path]) -> None:
        """Write config to a TOML file."""
        data: Dict[str, Any] = {
            "agent": {
                "name": self.agent_name,
                "system_prompt": self.system_prompt,
                "model": self.model,
                "provider_url": self.provider_url,
                "tolerance_default": self.tolerance_default,
            },
        }
        if self.tools:
            data["tools"] = self.tools
        with open(path, "wb") as f:
            tomli_w.dump(data, f)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "system_prompt": self.system_prompt,
            "tools": copy.deepcopy(self.tools),
            "model": self.model,
            "provider_url": self.provider_url,
            "tolerance_default": self.tolerance_default,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BedrockConfig:
        return cls(
            agent_name=d["agent_name"],
            system_prompt=d["system_prompt"],
            tools=d.get("tools", []),
            model=d["model"],
            provider_url=d["provider_url"],
            tolerance_default=d.get("tolerance_default", 0.05),
        )

    def canonical_hash(self) -> str:
        """SHA-256 of the canonical JSON representation (sorted keys)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BedrockConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"BedrockConfig(agent_name={self.agent_name!r}, model={self.model!r})"
