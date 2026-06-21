"""
Repository layer over the violations table — the only place that issues
SQLAlchemy queries. Both the live pipeline (inserts) and the dashboard
(reads/aggregations) go through this class rather than touching the ORM
session directly, keeping query logic in one place.

Timezone handling: SQLite's DateTime column does not round-trip timezone
info (it silently returns naive datetimes on read, even if a tz-aware
datetime was stored). To keep behavior identical across the sqlite and
postgresql backends, every datetime is normalized to naive UTC at the
write boundary (`_to_naive_utc`) before it touches the ORM, and all
comparisons in this file use naive UTC values for the same reason.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import select, func

from traffic_system.utils.logging_utils import get_logger
from traffic_system.storage.db import Database
from traffic_system.storage.models import ViolationORM

logger = get_logger(__name__)


def _to_naive_utc(dt: datetime) -> datetime:
    """Converts any datetime (naive or tz-aware) to a naive UTC datetime,
    which is the only representation guaranteed to round-trip identically
    through both SQLite and PostgreSQL via this codebase's ORM models."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=None)


def _parse_iso_to_naive_utc(iso_string: str) -> datetime:
    dt = datetime.fromisoformat(iso_string)
    return _to_naive_utc(dt)


class ViolationRepository:
    def __init__(self, db: Database):
        self._db = db

    # ------------------------------------------------------------------ writes

    def insert(self, metadata_record: dict) -> None:
        with self._db.session() as session:
            row = ViolationORM(
                violation_id=metadata_record["violation_id"],
                timestamp=_parse_iso_to_naive_utc(metadata_record["timestamp"]),
                camera_id=metadata_record["camera_id"],
                camera_gps_lat=metadata_record.get("camera_gps_lat"),
                camera_gps_lon=metadata_record.get("camera_gps_lon"),
                violation_type=metadata_record["violation_type"],
                confidence=metadata_record["confidence"],
                detector_confidence=metadata_record["detector_confidence"],
                classifier_confidence=metadata_record["classifier_confidence"],
                rule_certainty=metadata_record["rule_certainty"],
                track_id=metadata_record["track_id"],
                vehicle_class=metadata_record["vehicle_class"],
                bbox=str(metadata_record["bbox"]),
                plate_text=metadata_record.get("plate_text"),
                plate_ocr_confidence=metadata_record.get("plate_ocr_confidence"),
                plate_format_valid=bool(metadata_record.get("plate_format_valid", False)),
                plate_used_super_resolution=bool(metadata_record.get("plate_used_super_resolution", False)),
                plate_needs_review=bool(metadata_record.get("plate_needs_review", True)),
                evidence_image_path=metadata_record["evidence_image_path"],
                extra=metadata_record.get("extra"),
                reviewed=bool(metadata_record.get("reviewed", False)),
                review_decision=metadata_record.get("review_decision"),
                created_at=_parse_iso_to_naive_utc(metadata_record["created_at"]),
            )
            session.add(row)

    def mark_reviewed(self, violation_id: str, decision: str) -> None:
        if decision not in ("confirmed", "rejected"):
            raise ValueError("decision must be 'confirmed' or 'rejected'")
        with self._db.session() as session:
            row = session.get(ViolationORM, violation_id)
            if row is None:
                raise KeyError(f"No violation with id {violation_id}")
            row.reviewed = True
            row.review_decision = decision
            row.review_timestamp = _to_naive_utc(datetime.now(timezone.utc))

    # ------------------------------------------------------------------ reads

    def search(
        self,
        plate: Optional[str] = None,
        violation_type: Optional[str] = None,
        camera_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        reviewed_only: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ViolationORM]:
        with self._db.session() as session:
            stmt = select(ViolationORM)
            if plate:
                stmt = stmt.where(ViolationORM.plate_text.ilike(f"%{plate}%"))
            if violation_type:
                stmt = stmt.where(ViolationORM.violation_type == violation_type)
            if camera_id:
                stmt = stmt.where(ViolationORM.camera_id == camera_id)
            if date_from:
                stmt = stmt.where(ViolationORM.timestamp >= _to_naive_utc(date_from))
            if date_to:
                stmt = stmt.where(ViolationORM.timestamp <= _to_naive_utc(date_to))
            if reviewed_only is not None:
                stmt = stmt.where(ViolationORM.reviewed == reviewed_only)

            stmt = stmt.order_by(ViolationORM.timestamp.desc()).limit(limit).offset(offset)
            return list(session.execute(stmt).scalars().all())

    def count_by_type(self, date_from: Optional[datetime] = None) -> dict[str, int]:
        with self._db.session() as session:
            stmt = select(ViolationORM.violation_type, func.count(ViolationORM.violation_id))
            if date_from:
                stmt = stmt.where(ViolationORM.timestamp >= _to_naive_utc(date_from))
            stmt = stmt.group_by(ViolationORM.violation_type)
            return {vtype: count for vtype, count in session.execute(stmt).all()}

    def count_by_date(self, days_back: int = 30) -> list[tuple[str, str, int]]:
        """Returns (date, violation_type, count) tuples for trend charts."""
        with self._db.session() as session:
            cutoff = _to_naive_utc(datetime.now(timezone.utc) - timedelta(days=days_back))
            stmt = select(ViolationORM).where(ViolationORM.timestamp >= cutoff)
            rows = session.execute(stmt).scalars().all()

        from collections import defaultdict
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for row in rows:
            day = row.timestamp.strftime("%Y-%m-%d")
            counts[(day, row.violation_type)] += 1

        return [(day, vtype, count) for (day, vtype), count in sorted(counts.items())]

    def pending_review_count(self) -> int:
        with self._db.session() as session:
            stmt = select(func.count(ViolationORM.violation_id)).where(
                ViolationORM.reviewed == False  # noqa: E712 (SQLAlchemy requires == here)
            )
            return session.execute(stmt).scalar_one()

    def get_by_id(self, violation_id: str) -> ViolationORM | None:
        with self._db.session() as session:
            return session.get(ViolationORM, violation_id)
