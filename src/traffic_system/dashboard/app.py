#!/usr/bin/env python3
"""
Streamlit dashboard: violation analytics, search, and human review queue.

Usage:
    streamlit run src/traffic_system/dashboard/app.py -- --config config/config.yaml

All data comes from ViolationRepository — this file contains zero direct
SQL/ORM calls, keeping the dashboard a pure presentation layer over the
same repository the live pipeline writes through.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from traffic_system.utils.config import load_config
from traffic_system.storage.db import Database
from traffic_system.storage.repository import ViolationRepository


@st.cache_resource
def get_repository(config_path: str) -> ViolationRepository:
    app_config = load_config(config_path)
    db = Database(app_config)
    db.create_all()
    return ViolationRepository(db)


def _get_config_path() -> str:
    # Streamlit passes script args after "--"; fall back to the default
    # location if none was supplied so `streamlit run app.py` still works.
    args = sys.argv[1:]
    if "--config" in args:
        return args[args.index("--config") + 1]
    return "config/config.yaml"


def render_overview(repo: ViolationRepository) -> None:
    st.header("Overview")

    col1, col2, col3 = st.columns(3)
    counts = repo.count_by_type()
    total = sum(counts.values())
    pending = repo.pending_review_count()

    col1.metric("Total Violations Logged", total)
    col2.metric("Pending Human Review", pending)
    col3.metric("Violation Types Active", len(counts))

    if counts:
        df = pd.DataFrame(
            [{"violation_type": k, "count": v} for k, v in counts.items()]
        ).sort_values("count", ascending=False)
        st.bar_chart(df.set_index("violation_type"))
    else:
        st.info("No violations recorded yet.")


def render_trends(repo: ViolationRepository) -> None:
    st.header("Trends Over Time")
    days_back = st.slider("Days to look back", min_value=1, max_value=90, value=30)

    rows = repo.count_by_date(days_back=days_back)
    if not rows:
        st.info("No data in the selected window.")
        return

    df = pd.DataFrame(rows, columns=["date", "violation_type", "count"])
    pivot = df.pivot_table(index="date", columns="violation_type", values="count", fill_value=0)
    st.line_chart(pivot)


def render_search(repo: ViolationRepository) -> None:
    st.header("Search Violation Records")

    col1, col2, col3 = st.columns(3)
    plate = col1.text_input("Plate number contains")
    violation_type = col2.text_input("Violation type")
    camera_id = col3.text_input("Camera ID")

    col4, col5 = st.columns(2)
    date_from = col4.date_input("From date", value=None)
    date_to = col5.date_input("To date", value=None)

    results = repo.search(
        plate=plate or None,
        violation_type=violation_type or None,
        camera_id=camera_id or None,
        date_from=datetime.combine(date_from, datetime.min.time()) if date_from else None,
        date_to=datetime.combine(date_to, datetime.max.time()) if date_to else None,
        limit=200,
    )

    if not results:
        st.info("No matching records.")
        return

    df = pd.DataFrame([{
        "violation_id": r.violation_id,
        "timestamp": r.timestamp,
        "camera_id": r.camera_id,
        "violation_type": r.violation_type,
        "confidence": round(r.confidence, 3),
        "plate_text": r.plate_text,
        "reviewed": r.reviewed,
        "review_decision": r.review_decision,
    } for r in results])

    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Export results as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"violations_export_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
    )

    selected_id = st.selectbox(
        "View evidence image for violation_id:", ["(none)"] + df["violation_id"].tolist()
    )
    if selected_id != "(none)":
        record = repo.get_by_id(selected_id)
        if record and Path(record.evidence_image_path).exists():
            st.image(record.evidence_image_path, caption=f"{record.violation_type} — {record.plate_text}")
        else:
            st.warning("Evidence image file not found on disk for this record.")


def render_review_queue(repo: ViolationRepository) -> None:
    st.header("Human Review Queue")

    pending = repo.search(reviewed_only=False, limit=100)
    if not pending:
        st.success("No violations awaiting review.")
        return

    for record in pending:
        with st.expander(
            f"{record.violation_type} | {record.camera_id} | "
            f"{record.timestamp:%Y-%m-%d %H:%M:%S} | confidence={record.confidence:.0%}"
        ):
            if Path(record.evidence_image_path).exists():
                st.image(record.evidence_image_path)
            st.write(f"Plate: {record.plate_text or 'Not read'}")
            st.write(f"Vehicle class: {record.vehicle_class}")

            col1, col2 = st.columns(2)
            if col1.button("Confirm violation", key=f"confirm_{record.violation_id}"):
                repo.mark_reviewed(record.violation_id, "confirmed")
                st.rerun()
            if col2.button("Reject (false positive)", key=f"reject_{record.violation_id}"):
                repo.mark_reviewed(record.violation_id, "rejected")
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="Traffic Violation Dashboard", layout="wide")
    st.title("Traffic Violation Detection — Operations Dashboard")

    config_path = _get_config_path()
    repo = get_repository(config_path)

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends", "Search & Export", "Review Queue"])
    with tab1:
        render_overview(repo)
    with tab2:
        render_trends(repo)
    with tab3:
        render_search(repo)
    with tab4:
        render_review_queue(repo)


if __name__ == "__main__":
    main()
