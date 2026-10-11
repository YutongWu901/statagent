"""Shared session state for the Streamlit UI."""
from __future__ import annotations

import streamlit as st

from statagent.agent.pipeline import AnalysisPipeline
from statagent.config import AppConfig, get_config
from statagent.data.store import DatasetStore


def init_state() -> None:
    # Recompute every rerun so .env / config edits take effect immediately
    # (get_config is cached per process, so this is cheap).
    st.session_state.cfg = get_config()
    if "store" not in st.session_state:
        st.session_state.store = DatasetStore()
    if "records" not in st.session_state:
        st.session_state.records = {}
    if "pending" not in st.session_state:
        st.session_state.pending = None
    if "blocked" not in st.session_state:
        st.session_state.blocked = None
    if "error" not in st.session_state:
        st.session_state.error = None
    if "llm_warning" not in st.session_state:
        st.session_state.llm_warning = None


def get_cfg() -> AppConfig:
    init_state()
    return st.session_state.cfg


def get_store() -> DatasetStore:
    init_state()
    return st.session_state.store


def get_pipeline() -> AnalysisPipeline:
    init_state()
    if (
        "pipeline" not in st.session_state
        or st.session_state.get("pipeline_cfg") is not st.session_state.cfg
    ):
        st.session_state.pipeline = AnalysisPipeline(st.session_state.cfg)
        st.session_state.pipeline_cfg = st.session_state.cfg
    return st.session_state.pipeline


def add_record(record) -> None:
    init_state()
    st.session_state.records[record.record_id] = record


def get_records() -> dict:
    init_state()
    return st.session_state.records
