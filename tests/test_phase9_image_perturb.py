"""Phase 9B unit 1b — pure-numpy seeded image perturbation operators.

Machinery tier: no extras, no flags, no network, no keys.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fi.alk import image_perturb as ip


def _raster(seed: int = 7, h: int = 24, w: int = 24) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_image_perturb_determinism_byte_identical() -> None:
    raster = _raster()
    a = ip.apply_image_perturbations(
        raster, operators=["blur", "jpeg_compress"], seed=1142
    )
    b = ip.apply_image_perturbations(
        raster, operators=["blur", "jpeg_compress"], seed=1142
    )
    assert np.array_equal(a["raster"], b["raster"])
    assert a["stanza"] == b["stanza"]
    # a different seed differs for the seed-keyed operators (occlusion).
    c = ip.apply_image_perturbations(raster, operators=["occlusion"], seed=1)
    d = ip.apply_image_perturbations(raster, operators=["occlusion"], seed=99)
    assert not np.array_equal(c["raster"], d["raster"])


def test_image_perturb_unknown_operator_raises() -> None:
    raster = _raster()
    with pytest.raises(ip.ImagePerturbationError) as excinfo:
        ip.apply_image_perturbations(raster, operators=["sharpen"], seed=0)
    assert "sharpen" in str(excinfo.value)


def test_image_perturb_pure_numpy_no_pillow() -> None:
    """The v1 dep contract: numpy only. Mirror the audioop-absence test 9A uses
    for _codec.py — scan import statements (not prose) and assert no heavy CV
    import. The module's docstring NAMES the banned libs to explain the mandate,
    so we match only on actual import lines."""
    source = Path(ip.__file__).read_text(encoding="utf-8")
    import_lines = [
        ln.strip()
        for ln in source.splitlines()
        if ln.strip().startswith(("import ", "from "))
    ]
    for line in import_lines:
        for banned in ("PIL", "cv2", "scipy", "imageio", "skimage"):
            assert banned not in line, f"banned import {banned!r}: {line!r}"
    # the only third-party import is numpy.
    assert any("numpy" in ln for ln in import_lines)


def test_image_perturb_paired_clean_link() -> None:
    raster = _raster()
    result = ip.apply_image_perturbations(
        raster, operators=["resolution_drop"], seed=5, paired_clean_run="clean-1"
    )
    assert result["paired_clean_run"] == "clean-1"
    assert result["stanza"]["paired_clean_run"] == "clean-1"
    ops = [r["operator"] for r in result["stanza"]["operators"]]
    assert ops == ["resolution_drop"]


def test_image_perturb_each_operator_runs() -> None:
    raster = _raster(h=32, w=40)
    for operator in ip.V1_IMAGE_PERTURBATION_OPERATORS:
        out = ip.apply_image_perturbations(raster, operators=[operator], seed=3)["raster"]
        assert out.shape == (32, 40, 3)
        assert out.dtype == np.uint8


def test_image_perturb_rejects_bad_raster() -> None:
    with pytest.raises(ip.ImagePerturbationError):
        ip.apply_image_perturbations(
            np.zeros((8, 8), dtype=np.uint8), operators=["blur"], seed=0
        )
    with pytest.raises(ip.ImagePerturbationError):
        ip.apply_image_perturbations(
            np.zeros((8, 8, 3), dtype=np.float32), operators=["blur"], seed=0
        )


def test_image_perturb_operators_closed_set() -> None:
    assert ip.V1_IMAGE_PERTURBATION_OPERATORS == (
        "blur", "jpeg_compress", "resolution_drop", "occlusion"
    )
