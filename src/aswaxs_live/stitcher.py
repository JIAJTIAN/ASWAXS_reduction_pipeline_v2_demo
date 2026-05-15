"""Live HDF5 stitching helpers for paired detector reductions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


DEFAULT_OVERLAP_Q_MAX = 0.15


@dataclass
class ReductionRows:
    path: Path
    q: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    mtime_ns: int
    size: int


def find_analysis_h5(output_dir: Path) -> Path | None:
    candidates = sorted(
        [*output_dir.glob("*_analysis.h5"), *output_dir.glob("*_analysis.hdf5")],
        key=lambda path: path.stat().st_mtime_ns if path.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def update_live_stitched_averages(
    pil300k_output_dir: Path,
    eig1m_output_dir: Path,
    combined_h5_path: Path | None = None,
    overlap_q_max: float = DEFAULT_OVERLAP_Q_MAX,
) -> Path | None:
    """Copy detector analysis records and stitch matching rows.

    The two detector reducers keep writing their own private HDF5 files during a
    live run. This coordinator function is the single writer for the combined
    batch file, so the public analysis HDF5 stays organized as one record:
    ``/entry/Pil300K``, ``/entry/Eig1M``, and ``/entry/stitched_averages``.
    """
    pil300k_h5 = find_analysis_h5(pil300k_output_dir)
    eig1m_h5 = find_analysis_h5(eig1m_output_dir)
    if pil300k_h5 is None or eig1m_h5 is None:
        return None
    pil300k = read_reduction_rows(pil300k_h5)
    eig1m = read_reduction_rows(eig1m_h5)
    if pil300k is None or eig1m is None:
        return None

    n_rows = min(pil300k.intensity.shape[0], eig1m.intensity.shape[0])
    if n_rows <= 0:
        return None

    target_h5 = combined_h5_path or pil300k_h5
    target_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target_h5, "a") as handle:
        entry = handle.require_group("entry")
        if combined_h5_path is not None:
            _copy_detector_entry(pil300k_h5, entry, detector="Pil300K")
            _copy_detector_entry(eig1m_h5, entry, detector="Eig1M")
        group = entry.require_group("stitched_averages")
        group.attrs["NX_class"] = "NXprocess"
        group.attrs["process_stage"] = "detector_stitching"
        group.attrs["low_q_detector"] = "Pil300K"
        group.attrs["high_q_detector"] = "Eig1M"
        group.attrs["low_q_analysis_h5"] = str(pil300k.path)
        group.attrs["high_q_analysis_h5"] = str(eig1m.path)
        group.attrs["overlap_q_max"] = float(overlap_q_max)
        curves = group.require_group("curves")
        curves.attrs["NX_class"] = "NXdata"
        curves.attrs["signal"] = "I"
        curves.attrs["axes"] = "q"
        for row in range(n_rows):
            try:
                stitched, scale, q_min, q_max, n_overlap = stitch_one_row(pil300k, eig1m, row, overlap_q_max)
            except ValueError:
                continue
            name = f"curve_{row + 1:03d}"
            if name in curves:
                del curves[name]
            curve = curves.create_group(name)
            curve.create_dataset("q", data=stitched[:, 0])
            curve.create_dataset("I", data=stitched[:, 1])
            curve.create_dataset("sigma_I", data=stitched[:, 2])
            curve.attrs["NX_class"] = "NXdata"
            curve.attrs["signal"] = "I"
            curve.attrs["axes"] = "q"
            curve.attrs["row_index"] = row + 1
            curve.attrs["high_q_scale_factor"] = float(scale)
            curve.attrs["overlap_q_min"] = float(q_min)
            curve.attrs["overlap_q_max"] = float(q_max)
            curve.attrs["n_overlap_points"] = int(n_overlap)
        handle.flush()
    return target_h5


def _copy_detector_entry(source_h5: Path, combined_entry: h5py.Group, detector: str) -> None:
    """Replace one detector branch in the combined analysis file.

    Copying the detector branch keeps the combined file self-contained while
    avoiding concurrent writes from the two live reducer processes.
    """
    if detector in combined_entry:
        del combined_entry[detector]
    detector_group = combined_entry.create_group(detector)
    detector_group.attrs["NX_class"] = "NXcollection"
    detector_group.attrs["detector"] = detector
    detector_group.attrs["source_analysis_h5"] = str(source_h5)
    with h5py.File(source_h5, "r") as source:
        if "entry" not in source:
            return
        for name in source["entry"]:
            if name == "stitched_averages":
                continue
            source.copy(source["entry"][name], detector_group, name=name)


def read_reduction_rows(path: Path) -> ReductionRows | None:
    data_path = "/entry/process_01_reduction/data"
    if not path.exists():
        return None
    stat = path.stat()
    with h5py.File(path, "r") as handle:
        if f"{data_path}/q" not in handle or f"{data_path}/I" not in handle:
            return None
        q = np.asarray(handle[f"{data_path}/q"][()], dtype=float)
        intensity = np.asarray(handle[f"{data_path}/I"][()], dtype=float)
        sigma = np.asarray(handle[f"{data_path}/sigma_I"][()], dtype=float) if f"{data_path}/sigma_I" in handle else np.full_like(intensity, np.nan)
    if intensity.ndim == 1:
        intensity = intensity.reshape(1, -1)
    if sigma.ndim == 1:
        sigma = sigma.reshape(1, -1)
    return ReductionRows(path=path, q=q, intensity=intensity, sigma=sigma, mtime_ns=stat.st_mtime_ns, size=stat.st_size)


def stitch_one_row(low_q: ReductionRows, high_q: ReductionRows, row: int, overlap_q_max: float) -> tuple[np.ndarray, float, float, float, int]:
    low_q_data = np.column_stack([low_q.q, low_q.intensity[row], low_q.sigma[row]])
    high_q_data = np.column_stack([high_q.q, high_q.intensity[row], high_q.sigma[row]])
    scale, q_min, q_max, n_overlap = scale_high_q_to_low_q(low_q_data, high_q_data, overlap_q_max)
    high_q_scaled = high_q_data.copy()
    high_q_scaled[:, 1:] *= scale
    low_q_part = low_q_data[low_q_data[:, 0] < q_max]
    high_q_part = high_q_scaled[high_q_scaled[:, 0] >= q_max]
    stitched = np.vstack([low_q_part, high_q_part])
    return stitched[np.argsort(stitched[:, 0])], scale, q_min, q_max, n_overlap


def scale_high_q_to_low_q(low_q: np.ndarray, high_q: np.ndarray, overlap_q_max: float) -> tuple[float, float, float, int]:
    q_low = max(float(np.nanmin(low_q[:, 0])), float(np.nanmin(high_q[:, 0])))
    q_high = min(float(np.nanmax(low_q[:, 0])), float(overlap_q_max))
    overlap = high_q[(high_q[:, 0] >= q_low) & (high_q[:, 0] <= q_high) & (high_q[:, 1] > 0)]
    if overlap.shape[0] < 3:
        raise ValueError("Too few positive overlap points for detector stitching.")
    low_q_positive = low_q[(low_q[:, 0] > 0) & (low_q[:, 1] > 0)]
    low_q_at_high_q = np.exp(np.interp(np.log(overlap[:, 0]), np.log(low_q_positive[:, 0]), np.log(low_q_positive[:, 1])))
    ratios = low_q_at_high_q / overlap[:, 1]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size < 3:
        raise ValueError("Too few valid overlap ratios for detector stitching.")
    return float(np.median(ratios)), float(np.nanmin(overlap[:, 0])), float(np.nanmax(overlap[:, 0])), int(ratios.size)
