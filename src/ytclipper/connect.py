"""Register the MCP server with the Claude desktop app."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

NAME = "ytclipper"


def config_path() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if system == "Windows":
        return Path(os.environ.get("APPDATA", Path.home())) / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def entry() -> dict:
    """How the Claude app should start the server: this very Python, so it finds the package."""
    path = os.pathsep.join(["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"])
    return {"command": sys.executable, "args": ["-m", "ytclipper.cli", "mcp"], "env": {"PATH": path}}


def connect(path: Path | None = None) -> tuple[Path, bool]:
    """Add (or update) the ytclipper server in the config. Returns (path, was_already_there)."""
    path = path or config_path()
    config: dict = {}
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            try:
                config = json.loads(text)
            except ValueError as e:
                raise RuntimeError(f"{path} is not valid JSON, so it was left alone ({e}).") from e
        shutil.copy2(path, path.with_name(f"{path.name}.backup-{time.strftime('%Y%m%d-%H%M%S')}"))
    servers = config.setdefault("mcpServers", {})
    existed = NAME in servers
    servers[NAME] = entry()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path, existed


def disconnect(path: Path | None = None) -> bool:
    path = path or config_path()
    if not path.exists():
        return False
    config = json.loads(path.read_text(encoding="utf-8") or "{}")
    if NAME not in config.get("mcpServers", {}):
        return False
    del config["mcpServers"][NAME]
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return True
