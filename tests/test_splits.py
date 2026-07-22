from pathlib import Path

import pytest

from ct_denoising.data import split_sources


def test_source_split_is_deterministic_and_disjoint():
    paths = [Path(f"image_{index}.png") for index in range(9)]
    first = split_sources(paths, seed=42)
    second = split_sources(paths, seed=42)
    assert first == second
    train, validation, test = map(set, first)
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert train | validation | test == set(paths)


def test_split_requires_three_images():
    with pytest.raises(ValueError, match="At least three"):
        split_sources([Path("a.png"), Path("b.png")])
