"""Shared test setup.

THE COLLISION WORLD (robot/world.py) is remembered across runs in the repo's data/world/ and, the
first time, built from every view set saved under data/ and sim/data/. A test's Robot must neither
read the bench's world nor write it: each test gets a world directory of its own, empty."""

import pytest


@pytest.fixture(autouse=True)
def _a_world_of_its_own(tmp_path_factory, monkeypatch):
    monkeypatch.setenv('CABLE_ASSEMBLY_WORLD_DIR', str(tmp_path_factory.mktemp('world')))
