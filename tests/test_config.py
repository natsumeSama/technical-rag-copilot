
import pytest

from technical_rag import config_v2
from technical_rag.config import (
    HYBRID_PROFILE,
    V1_PROFILE,
    V2_PROFILE,
    get_pipeline_profile,
)


def test_default_profile_is_v1(monkeypatch):
    monkeypatch.delenv("RAG_COLLECTION", raising=False)

    profile = get_pipeline_profile()

    assert profile == V1_PROFILE
    assert profile.embedding_revision is None


def test_v2_uses_pinned_embedding_model(monkeypatch):
    monkeypatch.setenv(
        "RAG_COLLECTION",
        "technical_docs_v2",
    )

    profile = get_pipeline_profile()

    assert profile == V2_PROFILE
    assert profile.embedding_revision is not None


def test_hybrid_uses_same_dense_model_as_v2():
    assert (
        HYBRID_PROFILE.embedding_model_id
        == V2_PROFILE.embedding_model_id
    )

    assert (
        HYBRID_PROFILE.embedding_revision
        == V2_PROFILE.embedding_revision
    )


def test_hybrid_profile_is_selectable(monkeypatch):
    monkeypatch.setenv(
        "RAG_COLLECTION",
        HYBRID_PROFILE.collection_name,
    )

    assert get_pipeline_profile() == HYBRID_PROFILE

def test_legacy_v2_imports_are_preserved():
    assert config_v2.MODEL_ID == V2_PROFILE.embedding_model_id
    assert config_v2.MODEL_REVISION == V2_PROFILE.embedding_revision
    assert config_v2.COLLECTION_NAME == V2_PROFILE.collection_name
