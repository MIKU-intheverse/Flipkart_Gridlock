"""
SQLAlchemy ORM models for the violations table. Using the ORM (rather than
raw SQL strings) means the same model definitions work unchanged against
both SQLite (prototype/dev) and PostgreSQL (production), since the engine
URL is the only thing that changes — see db.py.
"""

from __future__ import annotations

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text, Index,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ViolationORM(Base):
    __tablename__ = "violations"

    violation_id = Column(String(36), primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    camera_id = Column(String(64), nullable=False)
    camera_gps_lat = Column(Float, nullable=True)
    camera_gps_lon = Column(Float, nullable=True)

    violation_type = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    detector_confidence = Column(Float, nullable=False)
    classifier_confidence = Column(Float, nullable=False)
    rule_certainty = Column(Float, nullable=False)

    track_id = Column(Integer, nullable=False)
    vehicle_class = Column(String(32), nullable=False)
    bbox = Column(Text, nullable=False)               # JSON-encoded [x1,y1,x2,y2]

    plate_text = Column(String(16), nullable=True)
    plate_ocr_confidence = Column(Float, nullable=True)
    plate_format_valid = Column(Boolean, default=False)
    plate_used_super_resolution = Column(Boolean, default=False)
    plate_needs_review = Column(Boolean, default=True)

    evidence_image_path = Column(Text, nullable=False)
    extra = Column(Text, nullable=True)                # JSON-encoded dict

    reviewed = Column(Boolean, default=False)
    review_decision = Column(String(16), nullable=True)   # "confirmed" | "rejected" | None
    review_timestamp = Column(DateTime, nullable=True)

    created_at = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("idx_violations_timestamp", "timestamp"),
        Index("idx_violations_camera", "camera_id"),
        Index("idx_violations_plate", "plate_text"),
        Index("idx_violations_type", "violation_type"),
    )
