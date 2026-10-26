
from technical_rag.config_v2 import REGISTRY_PATH

from script import scan_corpus, sync_corpus


def test_both_scripts_use_same_registry():
    assert scan_corpus.REGISTRY_PATH == REGISTRY_PATH
    assert sync_corpus.REGISTRY_PATH == REGISTRY_PATH


def test_registry_uses_v2_database():
    assert REGISTRY_PATH.name == "registry_v2.db"
