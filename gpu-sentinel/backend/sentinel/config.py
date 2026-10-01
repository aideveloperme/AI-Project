"""Runtime configuration (12-factor, environment variables prefixed ``SENTINEL_``)."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTINEL_", env_file=".env", extra="ignore")

    # --- core
    env: str = "development"
    demo_mode: bool = True
    database_url: str = "sqlite:///./sentinel.db"
    tenant: str = "default"

    # --- telemetry
    telemetry_source: str = "simulator"  # simulator | prometheus
    prometheus_url: str = "http://localhost:9090"
    gpu_vendor: str = "nvidia"
    simulator_url: str = "http://localhost:9400"  # fault-injection API of the external simulator
    sim_nodes: int = 12
    sim_gpus_per_node: int = 8
    sim_seed: int | None = None

    # --- analysis
    analysis_interval_s: float = 10.0
    history_samples: int = 720
    peer_group_keys: str = "gpu_model,server_type,workload_kind"
    peer_min_group: int = 3
    incident_min_cycles: int = 2        # signal must persist N cycles before an incident opens
    incident_resolve_cycles: int = 6    # quiet cycles before auto-resolve
    persist_every_cycles: int = 3       # write samples to DB every N cycles (downsampling)

    # --- retention (days)
    retention_samples_days: int = 7
    retention_incidents_days: int = 365
    retention_audit_days: int = 365

    # --- security
    jwt_secret: str = "change-me-in-production-please-32b+"
    jwt_ttl_minutes: int = 480
    secret_key: str | None = None      # Fernet key for encrypting stored credentials
    bootstrap_admin_user: str = "admin"
    bootstrap_admin_password: str = "sentinel-admin"
    cors_origins: str = "http://localhost:3000"

    # --- AI (local by default; cloud only if explicitly allowed)
    llm_provider: str = "auto"         # auto | template | ollama | openai_compatible
    llm_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b-instruct-q4_K_M"
    llm_api_key: str | None = None
    llm_timeout_s: float = 30.0
    allow_cloud_llm: bool = False

    # --- notifications
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str = "gpu-sentinel@localhost"
    notify_min_severity: str = "warning"
    allow_external_notifications: bool = False  # Slack/Teams/PagerDuty are third-party SaaS

    # --- licensing
    license_file: str | None = None
    license_public_key: str | None = None

    @property
    def peer_keys(self) -> list[str]:
        return [k.strip() for k in self.peer_group_keys.split(",") if k.strip()]

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
