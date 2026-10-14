"""Shared isolation for tests that exercise immutable publication receipts."""

import pytest


@pytest.fixture(autouse=True)
def isolated_publication_store(tmp_path_factory, monkeypatch):
    # Replace inherited settings too: a developer's store is not test state.
    # Module fixtures and test bodies can still select their own temporary store.
    # Keep the store disjoint even when the whole tmp_path is an attempt root.
    store = tmp_path_factory.mktemp("publications")
    monkeypatch.setenv("OXYFORMER_PUBLICATION_STORE", str(store))
