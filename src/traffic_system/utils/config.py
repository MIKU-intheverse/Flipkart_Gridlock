"""
Configuration loader for the traffic violation detection system.

Every module in this codebase receives its settings through an instance of
`AppConfig` produced by `load_config()`. Nothing downstream should read a
YAML/JSON file directly or hardcode a threshold/path — if a new setting is
needed, it is added to config/config.yaml and exposed here.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


class ConfigError(Exception):
    """Raised when the configuration file is missing required keys or is malformed."""


def _require(d: dict, key: str, path: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required config key '{key}' under '{path}'")
    return d[key]


@dataclass
class SourceConfig:
    camera_id: str
    uri: str
    gps_lat: Optional[float]
    gps_lon: Optional[float]
    zone_config_path: str


@dataclass
class AppConfig:
    """
    Typed, validated view over config.yaml. Holds the raw dict too (`raw`)
    for any section a module needs that doesn't yet have a typed accessor —
    new modules should add a typed accessor here rather than reading `raw`
    directly, to keep config access centralized and discoverable.
    """
    raw: dict
    config_path: Path

    # ---- typed section accessors -------------------------------------------------
    @property
    def system(self) -> dict:
        return self.raw["system"]

    @property
    def preprocessing(self) -> dict:
        return self.raw["preprocessing"]

    @property
    def detection(self) -> dict:
        return self.raw["detection"]

    @property
    def tracking(self) -> dict:
        return self.raw["tracking"]

    @property
    def violations(self) -> dict:
        return self.raw["violations"]

    @property
    def lpr(self) -> dict:
        return self.raw["lpr"]

    @property
    def evidence(self) -> dict:
        return self.raw["evidence"]

    @property
    def storage(self) -> dict:
        return self.raw["storage"]

    @property
    def dashboard(self) -> dict:
        return self.raw["dashboard"]

    @property
    def training(self) -> dict:
        return self.raw["training"]

    @property
    def sources(self) -> list[SourceConfig]:
        out = []
        for s in self.raw["sources"]:
            out.append(SourceConfig(
                camera_id=_require(s, "camera_id", "sources"),
                uri=_require(s, "uri", "sources"),
                gps_lat=s.get("gps_lat"),
                gps_lon=s.get("gps_lon"),
                zone_config_path=_require(s, "zone_config", "sources"),
            ))
        return out

    # ---- path resolution -----------------------------------------------------
    def resolve_path(self, relative_path: str) -> Path:
        """
        Resolves any path found in the config relative to the project root
        (the parent of the directory containing config.yaml), so the system
        runs correctly regardless of the current working directory it's
        launched from.
        """
        p = Path(relative_path)
        if p.is_absolute():
            return p
        project_root = self.config_path.parent.parent
        return (project_root / p).resolve()

    def class_name_to_id(self) -> dict[str, int]:
        return {v: k for k, v in self.detection["classes"].items()}


def _apply_env_overrides(cfg: dict) -> dict:
    """
    A small set of values are allowed to be overridden via environment
    variables, specifically secrets that should never live in a committed
    YAML file (e.g. DB password). This intentionally only touches secrets,
    not general settings, so config.yaml remains the single source of truth
    for everything else.
    """
    pg_password = os.environ.get("TVS_DB_PASSWORD")
    if pg_password is not None:
        cfg.setdefault("storage", {}).setdefault("postgresql", {})["password"] = pg_password
    return cfg


def load_config(config_path: str | os.PathLike = "config/config.yaml") -> AppConfig:
    """
    Loads and validates config.yaml, returning a typed AppConfig.
    Raises ConfigError with a clear message if anything required is missing,
    rather than failing later with a confusing KeyError deep in a module.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise ConfigError(f"Config file not found at {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    raw = _apply_env_overrides(raw)

    required_top_level = [
        "system", "sources", "preprocessing", "detection", "tracking",
        "violations", "lpr", "evidence", "storage", "dashboard", "training",
    ]
    for key in required_top_level:
        _require(raw, key, "<root>")

    return AppConfig(raw=raw, config_path=path)


def load_zone_config(app_config: AppConfig, zone_config_path: str) -> dict:
    """
    Loads a per-camera zone JSON file (stop lines, no-parking polygons, lane
    direction). Kept separate from the main YAML config because these are
    drawn/measured per camera during site setup, not authored by hand like
    thresholds — see scripts/calibrate_camera.py for how they're produced.
    """
    resolved = app_config.resolve_path(zone_config_path)
    if not resolved.exists():
        raise ConfigError(f"Zone config not found at {resolved}")
    with open(resolved, "r") as f:
        return json.load(f)
