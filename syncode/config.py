"""Persistent configuration for Syncode (~/.syncode/config.json)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path(os.environ.get("SYNCODE_HOME", Path.home() / ".syncode"))
CONFIG_FILE = CONFIG_DIR / "config.json"

# Thu muc cu truoc khi rename (de migration 1 lan, giu key + history).
_LEGACY_DIR = Path(os.environ.get("KIDAGENT_HOME", Path.home() / ".kidagent"))


def _migrate_legacy_dir() -> None:
    """Copy config.json + history.json tu ~/.kidagent sang 1 lan duy nhat."""
    try:
        if CONFIG_DIR.exists() or not _LEGACY_DIR.exists():
            return
        if _LEGACY_DIR.resolve() == CONFIG_DIR.resolve():
            return
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "history.json"):
            src = _LEGACY_DIR / name
            dst = CONFIG_DIR / name
            if src.is_file() and not dst.exists():
                dst.write_bytes(src.read_bytes())
    except OSError:
        pass


_migrate_legacy_dir()

DEFAULTS: Dict[str, Any] = {
    "api_key": "",
    "base_url": "https://integrate.api.nvidia.com/v1",
    "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "planner_model": "",     # de trong = dung model chinh
    "worker_model": "",      # dung cho tat ca worker
    "worker1_model": "",     # model rieng w1 (uu hon worker_model)
    "worker2_model": "",     # model rieng w2 (uu hon worker_model)
    "critic_model": "",
    "refiner_model": "",
    "judge_model": "",
    "synthesizer_model": "",  # key cu, van doc lam fallback cho judge
    "temperature": 0.7,
    "max_tokens": 2048,
    "timeout": 120.0,    # giay cho moi luot goi LLM
    "max_retries": 3,    # so lan thu lai khi loi mang/429/5xx
    "tool_output_max_chars": 8000,  # cat ket qua tool dua vao context
    "max_tool_rounds": 8,  # so vong tool-call toi da cua worker
    "confirm_dangerous": True,  # hoi truoc khi chay lenh shell/xoa file
    "max_rounds": 1,      # so vong revise sau khi Critic phan hoi
    "num_workers": 2,     # so Worker trong swarm
    "stream": True,
    "thinking": "auto",  # auto | off | low (tat/bot reasoning rieng cua model)
    # MCP servers: {"<ten>": {"command": "...", "args": [...], "env": {...}}}
    "mcp_servers": {},
}

# Cac key duoc che gia khi hien thi
SECRET_KEYS = {"api_key"}


class Config:
    """Load/save settings. Uu tien: env NVIDIA_API_KEY > config file."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = dict(DEFAULTS)
        self.load()

    # ------------------------------------------------------------------ IO
    def load(self) -> None:
        self._data = dict(DEFAULTS)
        if CONFIG_FILE.exists():
            try:
                stored = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                for key in DEFAULTS:
                    if key in stored:
                        self._data[key] = stored[key]
            except (json.JSONDecodeError, OSError):
                pass  # file loi -> dung default
        # Env var co do uu tien cao hon de tich hop CI/thu cung
        env_key = os.environ.get("NVIDIA_API_KEY")
        if env_key:
            self._data["api_key"] = env_key

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        try:  # khong quan trong neu that bai
            CONFIG_FILE.chmod(0o600)
        except OSError:
            pass

    # ------------------------------------------------------------- Access
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> bool:
        if key not in DEFAULTS:
            return False
        current = DEFAULTS[key]
        if isinstance(current, bool):
            value = str(value).strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, (dict, list)):
            # gia tri cau truc (vd mcp_servers): nhan chuoi JSON
            if isinstance(value, str):
                value = json.loads(value)  # loi JSON -> de caller bat
            if not isinstance(value, type(current)):
                raise ValueError(f"{key} phai la {type(current).__name__}")
        else:
            value = str(value)
        self._data[key] = value
        self.save()
        return True

    def reset(self) -> None:
        self._data = dict(DEFAULTS)
        self.save()

    def as_display(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self._data.items():
            if key in SECRET_KEYS and value:
                out[key] = value[:6] + "..." + value[-4:]
            else:
                out[key] = value
        return out

    @property
    def has_api_key(self) -> bool:
        return bool(self._data.get("api_key"))
