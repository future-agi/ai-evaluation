"""Phase 9B unit 1b — pure-numpy seeded image perturbation operators (the image
analogue of 9A's ``live/_perturb.py`` acoustic operators).

ARCH-9B §2.1 / decision 9B-A1b (companion module home — substrate not loop),
9B-A6 (PURE-NUMPY v1, ZERO new dep — settles Open Q4).

MANDATORY (9B-A6): imports are **numpy + stdlib ONLY**. There is NO Pillow, no
scipy, no cv2, no imageio, no scikit-image — verified ``pyproject.toml`` carries
only ``numpy>=1.26.4``. Adding Pillow for the perturbation set would be a NEW
dependency + a license-audit obligation on the public repo's Apache-2.0 posture.
The kit's live substrate already imports numpy directly. A true-libjpeg or
PNG-render path is a NAMED post-v1 Pillow extra, auto-skip when absent, never a
v1 gate dependency.

Operators are deterministic under a recorded seed so stressed runs replay
byte-identically (the determinism the gate re-asserts). Each operates on a numpy
``uint8`` raster (H x W x C) and is computed as a paired clean-vs-stressed delta
(the ``_perturb.apply_text_perturbations`` discipline).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# closed set; analogue of _perturb.py PERTURBATION_OPERATORS. All pure-numpy, all
# deterministic-under-seed. apply_image_perturbations RAISES for any operator not
# in this set (the _perturb.py raise-wall pattern generalized).
V1_IMAGE_PERTURBATION_OPERATORS = ("blur", "jpeg_compress", "resolution_drop", "occlusion")


class ImagePerturbationError(ValueError):
    """Raised for an unknown operator or a mis-shaped raster (a contract error —
    the _perturb.py raise-wall analogue). A ``ValueError`` subclass."""


def _require_raster(raster: Any, *, where: str) -> np.ndarray:
    """Type-guard the input as a numpy ``uint8`` H x W x C raster. A non-uint8 /
    non-3D input raises ``ImagePerturbationError`` — we never silently
    mis-shape."""

    if not isinstance(raster, np.ndarray):
        raise ImagePerturbationError(
            f"{where} needs a numpy uint8 H x W x C raster; got {type(raster).__name__}"
        )
    if raster.ndim != 3:
        raise ImagePerturbationError(
            f"{where} needs a 3-D (H x W x C) raster; got ndim={raster.ndim}"
        )
    if raster.dtype != np.uint8:
        raise ImagePerturbationError(
            f"{where} needs a uint8 raster; got dtype={raster.dtype}"
        )
    return raster


def blur(raster: np.ndarray, *, kernel_radius: int = 1, seed: int = 0) -> np.ndarray:
    """Separable box-kernel blur as a numpy stride convolution (no scipy). A box
    average over a ``(2*kernel_radius+1)`` window, applied separably across rows
    then columns with edge replication. Deterministic (no rng draw); the ``seed``
    is accepted for a uniform operator signature."""

    arr = _require_raster(raster, where="blur").astype(np.float64)
    radius = max(int(kernel_radius), 0)
    if radius == 0:
        return arr.astype(np.uint8)
    width = 2 * radius + 1

    def _box_axis(data: np.ndarray, axis: int) -> np.ndarray:
        padded = np.pad(
            data,
            [(radius, radius) if a == axis else (0, 0) for a in range(data.ndim)],
            mode="edge",
        )
        acc = np.zeros_like(data)
        for offset in range(width):
            sl = [slice(None)] * data.ndim
            sl[axis] = slice(offset, offset + data.shape[axis])
            acc = acc + padded[tuple(sl)]
        return acc / float(width)

    out = _box_axis(_box_axis(arr, 0), 1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def jpeg_compress(raster: np.ndarray, *, quality: int = 50, seed: int = 0) -> np.ndarray:
    """Block-DCT quantization approximation in pure numpy (8x8 DCT-II matrices +
    a quality-keyed quant table). A true libjpeg path is the post-v1 Pillow extra
    (auto-skip). Deterministic (no rng draw); ``seed`` accepted for a uniform
    signature."""

    arr = _require_raster(raster, where="jpeg_compress").astype(np.float64)
    q = int(np.clip(quality, 1, 100))
    # the standard JPEG quality -> scale heuristic.
    if q < 50:
        scale = 5000.0 / q
    else:
        scale = 200.0 - 2.0 * q
    quant = max(1.0, scale / 16.0)  # a single flat quantization step (luma-ish)

    n = 8
    k = np.arange(n)
    # DCT-II orthonormal basis (8x8), built deterministically.
    basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    basis *= np.sqrt(2.0 / n)
    basis[0, :] = np.sqrt(1.0 / n)
    # basis[i, x] applies the i-th cosine over sample x; forward = basis @ block.

    h, w, c = arr.shape
    pad_h = (-h) % n
    pad_w = (-w) % n
    padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    out = np.empty_like(padded)
    for ch in range(c):
        plane = padded[:, :, ch] - 128.0
        for r0 in range(0, padded.shape[0], n):
            for c0 in range(0, padded.shape[1], n):
                block = plane[r0:r0 + n, c0:c0 + n]
                coeffs = basis @ block @ basis.T
                quantized = np.round(coeffs / quant) * quant
                restored = basis.T @ quantized @ basis
                out[r0:r0 + n, c0:c0 + n, ch] = restored + 128.0

    out = out[:h, :w, :]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def resolution_drop(raster: np.ndarray, *, scale: float = 0.5, seed: int = 0) -> np.ndarray:
    """numpy decimate -> upsample (nearest), the band-limit analogue (the
    ``resample_8k`` analogue from voice). Downscale by ``scale`` then nearest-
    neighbour back to the original shape, destroying high-frequency detail.
    Deterministic (no rng draw); ``seed`` accepted for a uniform signature."""

    arr = _require_raster(raster, where="resolution_drop")
    s = float(scale)
    if not 0.0 < s < 1.0:
        # scale outside (0,1) is a no-op (full resolution).
        return arr.copy()
    h, w, _ = arr.shape
    small_h = max(1, int(round(h * s)))
    small_w = max(1, int(round(w * s)))
    # deterministic nearest-neighbour decimation.
    row_idx = (np.arange(small_h) * (h / small_h)).astype(np.int64)
    col_idx = (np.arange(small_w) * (w / small_w)).astype(np.int64)
    small = arr[np.ix_(row_idx, col_idx, np.arange(arr.shape[2]))]
    # nearest-neighbour upsample back to (h, w).
    up_rows = (np.arange(h) * (small_h / h)).astype(np.int64)
    up_cols = (np.arange(w) * (small_w / w)).astype(np.int64)
    up = small[np.ix_(up_rows, up_cols, np.arange(arr.shape[2]))]
    return up.astype(np.uint8)


def occlusion(raster: np.ndarray, *, coverage: float = 0.2, seed: int = 0) -> np.ndarray:
    """Seeded rectangular mask zeroing a region (``np.random.default_rng(seed)``).
    The mask covers approximately ``coverage`` of the area; its position is keyed
    on the seed so a re-run is byte-identical."""

    arr = _require_raster(raster, where="occlusion").copy()
    cov = float(np.clip(coverage, 0.0, 1.0))
    if cov <= 0.0:
        return arr
    h, w, _ = arr.shape
    rng = np.random.default_rng(seed)
    side = float(np.sqrt(cov))
    box_h = max(1, int(round(h * side)))
    box_w = max(1, int(round(w * side)))
    top = int(rng.integers(0, max(1, h - box_h + 1)))
    left = int(rng.integers(0, max(1, w - box_w + 1)))
    arr[top:top + box_h, left:left + box_w, :] = 0
    return arr


_OPERATOR_FNS = {
    "blur": blur,
    "jpeg_compress": jpeg_compress,
    "resolution_drop": resolution_drop,
    "occlusion": occlusion,
}


def perturbations_stanza(
    applied: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    paired_clean_run: str | None = None,
) -> dict[str, Any]:
    """The applied-operator stanza (the ``_perturb.perturbations_stanza``
    analogue): operator list, recorded seed, and the clean-twin link (deltas
    render upstream)."""

    return {
        "operators": [dict(record) for record in applied],
        "seed": seed,
        "paired_clean_run": paired_clean_run,
    }


def apply_image_perturbations(
    raster: np.ndarray,
    *,
    operators: Sequence[str],
    seed: int = 0,
    params: Mapping[str, Any] | None = None,
    paired_clean_run: str | None = None,
) -> dict[str, Any]:
    """Walk the operator list applying each with ``seed + index`` (the
    ``_perturb.apply_text_perturbations`` pattern). Returns
    ``{"raster": np.ndarray, "stanza": {...}, "paired_clean_run": <ref>}``.

    The stanza mirrors ``perturbations_stanza`` — the applied-operator list + the
    ``paired_clean_run`` link. The ``WorldSpec.perturbation_profile`` field
    (contract.py:214) carries the profile LABEL on the stressed run.

    RAISES ``ImagePerturbationError`` for any operator not in
    ``V1_IMAGE_PERTURBATION_OPERATORS`` (a contract error — the raise-wall).

    DETERMINISM (the gate asserts this, unit 5): same raster + same operators +
    same seed => byte-identical output raster. No wall-clock, no randomness
    outside the keyed rng."""

    out = _require_raster(raster, where="apply_image_perturbations").copy()
    params = dict(params or {})
    applied: list[dict[str, Any]] = []
    for index, operator in enumerate(operators):
        if operator not in V1_IMAGE_PERTURBATION_OPERATORS:
            raise ImagePerturbationError(
                f"unknown perturbation operator {operator!r}; "
                f"expected one of {V1_IMAGE_PERTURBATION_OPERATORS}"
            )
        op_seed = seed + index
        op_params = dict(params.get(operator) or {})
        out = _OPERATOR_FNS[operator](out, seed=op_seed, **op_params)
        applied.append({"operator": operator, "seed": op_seed, **op_params})

    return {
        "raster": out,
        "stanza": perturbations_stanza(applied, seed=seed, paired_clean_run=paired_clean_run),
        "paired_clean_run": paired_clean_run,
    }
