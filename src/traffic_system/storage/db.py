"""
Database engine + session factory. The backend (SQLite vs PostgreSQL) is
selected entirely through config.yaml's storage.backend key — no module
outside this file constructs a connection string or an Engine directly.
"""

from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from traffic_system.utils.config import AppConfig
from traffic_system.utils.logging_utils import get_logger
from traffic_system.storage.models import Base

logger = get_logger(__name__)


def build_engine_url(app_config: AppConfig) -> str:
    cfg = app_config.storage
    backend = cfg["backend"]

    if backend == "sqlite":
        db_path = app_config.resolve_path(cfg["sqlite_path"])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"

    if backend == "postgresql":
        pg = cfg["postgresql"]
        return (
            f"postgresql+psycopg2://{pg['user']}:{pg['password']}"
            f"@{pg['host']}:{pg['port']}/{pg['database']}"
        )

    raise ValueError(f"Unsupported storage.backend '{backend}' in config.yaml")


class Database:
    def __init__(self, app_config: AppConfig):
        url = build_engine_url(app_config)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self._engine = create_engine(url, connect_args=connect_args, future=True)
        self._SessionFactory = sessionmaker(bind=self._engine, expire_on_commit=False)
        logger.info("Database engine created (backend=%s)", app_config.storage["backend"])

    def create_all(self) -> None:
        Base.metadata.create_all(self._engine)
        logger.info("Database schema ensured (tables created if missing)")

    @contextmanager
    def session(self) -> Session:
        session = self._SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
