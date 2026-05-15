"""Reduce a structured ASAXS energy/group/frame HDF5 sequence.

This is the main multi-frame workflow: it maps source files into a manifest,
integrates each frame, rejects bad repeats, monitor-normalizes, averages by
energy/group, applies background/GC corrections, and writes analysis HDF5
provenance without modifying the source HDF5 files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import multiprocessing
import os
import queue
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from pyFAI import load as load_poni

from .analysis_h5 import (
    create_analysis_h5_from_data,
    file_sha256,
    write_background_subtraction_to_analysis_h5,
    write_glassy_carbon_normalization_to_analysis_h5,
    write_reduction_to_analysis_h5,
)
from .reduce_sequence import (
    build_sequence_map,
    collect_files,
    expected_count,
    remove_sequence_indices,
    resolve_sequence_files,
    write_manifest,
)
from .reduction_pipeline import (
    DEFAULT_DATASET_PATH,
    DEFAULT_FLUORESCENCE_Q_RANGE,
    DEFAULT_GC_Q_RANGE,
    DEFAULT_NPT,
    _integrated_intensity,
    _integrate_image,
    _load_mask,
    _load_reference_curve,
    _read_hdf5_image,
    energy_kev_to_wavelength_m,
    estimate_constant_fluorescence,
)

PROJECT_DIR = Path(__file__).resolve().parent


NDATTR_PREFIX = "entry/instrument/NDAttributes"


@dataclass
class ManifestItem:
    sequence_index: int
    energy_index: int
    group_index: int
    frame_index: int
    path: Path


@dataclass
class FrameCurve:
    item: ManifestItem
    energy_kev: float | None
    monitor_value: float
    q: np.ndarray
    intensity: np.ndarray
    total_intensity: float
    normalized_intensity: np.ndarray


@dataclass
class GroupAverage:
    energy_index: int
    group_index: int
    q: np.ndarray
    energy_kev: float | None
    avg_intensity: np.ndarray
    avg_error: np.ndarray
    frame_count: int
    kept_count: int
    dropped_count: int
    kept_sequence_indices: list[int]
    dropped_sequence_indices: list[int]
    avg_total_intensity: float
    avg_monitor_value: float


@dataclass
class FinalRecord:
    energy_index: int
    energy_kev: float | None
    q: np.ndarray
    final_before_fluorescence: np.ndarray
    final_error: np.ndarray
    component_columns: list[np.ndarray]
    component_names: list[str]
    metadata: dict[str, object]


@dataclass
class FinalOutput:
    path: Path
    component_path: Path
    energy_index: int
    energy_kev: float | None
    q: np.ndarray
    I: np.ndarray
    sigma_I: np.ndarray
    component_names: list[str]
    component_columns: list[np.ndarray]
    metadata: dict[str, object]


def stabilize_energy_kev(
    energy_kev: float | None,
    previous_energy_kev: float | None,
    delta_percent: float,
) -> float | None:
    """Suppress tiny monochromator readback jitter within one energy point."""
    if energy_kev is None:
        return previous_energy_kev
    if previous_energy_kev is None:
        return energy_kev
    if delta_percent < 0:
        raise ValueError("delta energy percent must be non-negative.")
    if previous_energy_kev == 0:
        return energy_kev
    percent_diff = abs(energy_kev - previous_energy_kev) / abs(previous_energy_kev) * 100.0
    if percent_diff <= delta_percent:
        return previous_energy_kev
    return energy_kev


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reduce an ASWAXS sequence manifest with frame rejection, monitor normalization, averaging, air subtraction, and GC scaling."
    )
    parser.add_argument("--manifest", help="Existing sequence manifest, or output manifest path when using --data-dir.")
    parser.add_argument("--data-dir", help="Directory containing the continuous HDF5 sequence. If provided, a manifest is created first.")
    parser.add_argument("--pattern", default="*.h5", help="HDF5 filename pattern when using --data-dir. Default: *.h5")
    parser.add_argument("--num-energies", type=int, help="Number of energies in the sequence.")
    parser.add_argument("--num-groups", type=int, help="Number of groups per energy.")
    parser.add_argument("--num-frames", type=int, help="Number of repeated frames per group.")
    parser.add_argument("--skip-files", type=int, default=0, help="Number of leading sorted files to ignore before sequence mapping.")
    parser.add_argument(
        "--skip-sequence-indices",
        nargs="*",
        type=int,
        default=[],
        help="One-based sorted file positions to skip after --skip-files, for known beamdown/repeated measurements.",
    )
    parser.add_argument(
        "--allow-extra-files",
        action="store_true",
        help="Allow more files than expected and use the first complete sequence.",
    )
    parser.add_argument(
        "--resume-mode",
        choices=("strict", "first", "last"),
        default="strict",
        help="How to handle extra files when creating a manifest. Default: strict.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not ask for beamdown indices interactively while creating a manifest.",
    )
    parser.add_argument("--poni", required=True, help="Calibration .poni file.")
    parser.add_argument("--mask", required=True, help="Mask file (.npy or EDF-readable).")
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs" / "sequence_reduction_output"), help="Output directory.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes for parallel reduction by contiguous energy batches. Default: 1",
    )
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="HDF5 detector image dataset path.")
    parser.add_argument("--npt", type=int, default=DEFAULT_NPT, help="Number of q bins.")
    parser.add_argument("--unit", default="q_A^-1", help="pyFAI radial unit.")
    parser.add_argument(
        "--delta-energy-percent",
        type=float,
        default=1e-3,
        help="Accept consecutive Mono_Energy fluctuations within this percent as the same previous energy. Default: 1e-3",
    )
    parser.add_argument(
        "--detector",
        choices=("auto", "Pil300K", "Eig1M"),
        default="auto",
        help="Detector used to choose default monitor normalization.",
    )
    parser.add_argument(
        "--monitor-key",
        help="Override monitor normalization key. Defaults: Pil300K -> SPD, Eig1M -> WPD.",
    )
    parser.add_argument("--outlier-zmax", type=float, default=3.5, help="MAD modified-z threshold for bad-frame rejection.")
    parser.add_argument("--sample-group", type=int, help="Group index for sample.")
    parser.add_argument("--air-group", type=int, help="Group index for air/background.")
    parser.add_argument("--empty-group", type=int, help="Group index for empty cell/capillary.")
    parser.add_argument("--water-group", type=int, help="Group index for water reference.")
    parser.add_argument("--gc-group", type=int, help="Group index for glassy carbon.")
    parser.add_argument(
        "--gc-reference-file",
        help="Optional two-column q,I glassy carbon reference curve. Default: built-in NIST SRM 3600 values.",
    )
    parser.add_argument(
        "--gc-q-range",
        nargs=2,
        type=float,
        default=DEFAULT_GC_Q_RANGE,
        metavar=("QMIN", "QMAX"),
        help="q range used to match glassy carbon to reference. Default: 0.03 0.20",
    )
    parser.add_argument(
        "--capillary-thickness",
        type=float,
        help="Sample/capillary thickness in the same units as --gc-thickness. Used for thickness normalization after GC scaling.",
    )
    parser.add_argument(
        "--gc-thickness",
        type=float,
        help="Measured glassy carbon thickness in the same units as --capillary-thickness. Used for thickness normalization after GC scaling.",
    )
    parser.add_argument("--subtract-fluorescence", action="store_true", help="Subtract constant fluorescence background.")
    parser.add_argument(
        "--fluorescence-q-range",
        nargs=2,
        type=float,
        default=DEFAULT_FLUORESCENCE_Q_RANGE,
        metavar=("QMIN", "QMAX"),
        help="q range used to estimate fluorescence. Default: 0.16 0.20",
    )
    parser.add_argument("--fluorescence-level", type=float, help="Fixed fluorescence level.")
    parser.add_argument(
        "--fluorescence-reference",
        choices=("latest", "each"),
        default="latest",
        help="Use the latest energy curve as one shared fluorescence background, or estimate each curve separately. Default: latest.",
    )
    parser.add_argument(
        "--analysis-h5",
        help="Analysis HDF5 output path. Default: output-dir/analysis.h5.",
    )
    return parser


def read_manifest(path: Path) -> list[ManifestItem]:
    items: list[ManifestItem] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence_index", "energy_index", "group_index", "frame_index", "hdf5_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            items.append(
                ManifestItem(
                    sequence_index=int(row["sequence_index"]),
                    energy_index=int(row["energy_index"]),
                    group_index=int(row["group_index"]),
                    frame_index=int(row["frame_index"]),
                    path=Path(row["hdf5_path"]).expanduser().resolve(),
                )
            )
    if not items:
        raise ValueError("Manifest has no sequence rows.")
    return items


def create_manifest_from_sequence(args: argparse.Namespace, output_dir: Path) -> Path:
    missing = [
        name
        for name in ("num_energies", "num_groups", "num_frames")
        if getattr(args, name) is None
    ]
    if missing:
        options = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(f"When using --data-dir, also provide: {options}")
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    expected = expected_count(args.num_energies, args.num_groups, args.num_frames)
    raw_files = collect_files(data_dir, args.pattern, args.skip_files)
    sequence_files, skip_indices = resolve_sequence_files(
        raw_files=raw_files,
        expected=expected,
        initial_skip_indices=args.skip_sequence_indices,
        allow_extra_files=args.allow_extra_files,
        resume_mode=args.resume_mode,
        no_prompt=args.no_prompt,
    )
    sequence_items = build_sequence_map(sequence_files, args.num_energies, args.num_groups, args.num_frames)
    manifest_path = Path(args.manifest).expanduser() if args.manifest else output_dir / "sequence_manifest.csv"
    manifest = write_manifest(sequence_items, manifest_path)

    print("ASWAXS sequence validated.")
    print(f"Data directory: {data_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Expected files: {expected}")
    print(f"Actual files found after leading skip: {len(raw_files)}")
    if skip_indices:
        print(f"Skipped sequence indices: {', '.join(str(index) for index in skip_indices)}")
    print(f"Actual files after beamdown skips: {len(remove_sequence_indices(raw_files, skip_indices))}")
    print(f"Actual files used: {len(sequence_files)}")
    print(f"Resume mode: {args.resume_mode}")
    print(f"Energies: {args.num_energies}")
    print(f"Groups per energy: {args.num_groups}")
    print(f"Frames per group: {args.num_frames}")
    print(f"Manifest: {manifest}")
    return manifest


def resolve_manifest(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.data_dir:
        return create_manifest_from_sequence(args, output_dir)
    if not args.manifest:
        raise ValueError("Provide either --manifest or --data-dir with --num-energies, --num-groups, and --num-frames.")
    return Path(args.manifest).expanduser().resolve()


def infer_detector(items: list[ManifestItem], requested: str) -> str:
    if requested != "auto":
        return requested
    text = " ".join(str(item.path) for item in items[: min(len(items), 10)])
    if "Pil300K" in text:
        return "Pil300K"
    if "Eig1M" in text:
        return "Eig1M"
    raise ValueError("Could not infer detector from manifest paths. Use --detector Pil300K or --detector Eig1M.")


def default_monitor_key(detector: str) -> str:
    if detector == "Pil300K":
        return "SPD"
    if detector == "Eig1M":
        return "WPD"
    raise ValueError(f"Unsupported detector for monitor normalization: {detector}")


def read_ndattr_scalar(path: Path, key: str) -> float | None:
    full_key = f"{NDATTR_PREFIX}/{key}"
    with h5py.File(path, "r") as handle:
        if full_key not in handle:
            return None
        value = np.asarray(handle[full_key][()])
    if value.size == 0:
        return None
    return float(value.reshape(-1)[0])


def reject_outliers(total_intensities: np.ndarray, zmax: float) -> np.ndarray:
    finite = np.isfinite(total_intensities)
    if np.count_nonzero(finite) == 0:
        return np.zeros_like(total_intensities, dtype=bool)
    median = np.median(total_intensities[finite])
    mad = np.median(np.abs(total_intensities[finite] - median))
    if mad == 0:
        return finite
    modified_z = np.full_like(total_intensities, np.inf, dtype=float)
    modified_z[finite] = 0.6745 * (total_intensities[finite] - median) / mad
    return np.abs(modified_z) <= zmax


def resample_to_q(q_target: np.ndarray, q_source: np.ndarray, values: np.ndarray) -> np.ndarray:
    mask = np.isfinite(q_source) & np.isfinite(values)
    if np.count_nonzero(mask) < 2:
        return np.full_like(q_target, np.nan, dtype=float)
    return np.interp(q_target, q_source[mask], values[mask], left=np.nan, right=np.nan)


def reduce_manifest_frames(
    items: list[ManifestItem],
    args: argparse.Namespace,
    monitor_key: str,
    progress_label: str | None = None,
    progress_queue: object | None = None,
) -> list[FrameCurve]:
    """Integrate every manifest item and keep per-frame values for later QC."""
    curves: list[FrameCurve] = []
    ai = load_poni(str(Path(args.poni).expanduser().resolve()))
    mask = _load_mask(Path(args.mask).expanduser().resolve())
    previous_energy_kev: float | None = None
    for idx, item in enumerate(items, start=1):
        if not item.path.exists():
            raise FileNotFoundError(f"Missing HDF5 file from manifest: {item.path}")
        raw_energy_kev = read_ndattr_scalar(item.path, "Mono_Energy")
        energy_kev = stabilize_energy_kev(raw_energy_kev, previous_energy_kev, args.delta_energy_percent)
        if energy_kev is not None:
            ai.wavelength = energy_kev_to_wavelength_m(energy_kev)
            previous_energy_kev = energy_kev
        image = _read_hdf5_image(item.path, args.dataset_path, frame=None)
        q, intensity = _integrate_image(ai, image, mask, args.npt, args.unit)
        monitor_value = read_ndattr_scalar(item.path, monitor_key)
        if monitor_value is None:
            raise KeyError(f"Monitor key {monitor_key} not found in {item.path}")
        if monitor_value == 0:
            raise ValueError(f"Monitor key {monitor_key} is zero in {item.path}")
        total_intensity = _integrated_intensity(q, intensity)
        curves.append(
            FrameCurve(
                item=item,
                energy_kev=energy_kev,
                monitor_value=monitor_value,
                q=q,
                intensity=intensity,
                total_intensity=total_intensity,
                normalized_intensity=intensity / monitor_value,
            )
        )
        if idx == 1 or idx % 50 == 0 or idx == len(items):
            prefix = f"[{progress_label}] " if progress_label else ""
            message = f"{prefix}Reduced frame {idx}/{len(items)}: {item.path.name}"
            print(message)
            if progress_queue is not None:
                progress_queue.put(message)
    return curves


def split_energy_batches(energy_indices: list[int], jobs: int) -> list[list[int]]:
    if jobs <= 1 or len(energy_indices) <= 1:
        return [energy_indices]
    chunk_size = max(1, math.ceil(len(energy_indices) / jobs))
    return [energy_indices[start : start + chunk_size] for start in range(0, len(energy_indices), chunk_size)]


def reduce_energy_batch_worker(
    batch_id: int,
    energy_indices: list[int],
    items: list[ManifestItem],
    args_dict: dict[str, object],
    monitor_key: str,
    progress_queue: object | None,
) -> tuple[int, list[int], list[GroupAverage]]:
    args = argparse.Namespace(**args_dict)
    if progress_queue is not None:
        progress_queue.put(
            f"[worker {batch_id}] Starting energies {energy_indices[0]}-{energy_indices[-1]} "
            f"({len(energy_indices)} energies, {len(items)} frames)"
        )
    curves = reduce_manifest_frames(
        items,
        args,
        monitor_key,
        progress_label=f"worker {batch_id}",
        progress_queue=progress_queue,
    )
    averages = average_groups(curves, args.outlier_zmax)
    if progress_queue is not None:
        progress_queue.put(
            f"[worker {batch_id}] Finished energies {energy_indices[0]}-{energy_indices[-1]} "
            f"with {len(averages)} group averages"
        )
    return batch_id, energy_indices, averages


def drain_progress_queue(progress_queue: object | None) -> None:
    if progress_queue is None:
        return
    while True:
        try:
            message = progress_queue.get_nowait()
        except queue.Empty:
            break
        else:
            print(message)


def reduce_manifest_frames_parallel(
    items: list[ManifestItem],
    args: argparse.Namespace,
    monitor_key: str,
) -> list[GroupAverage]:
    energy_indices = sorted({item.energy_index for item in items})
    max_workers = max(1, os.cpu_count() or 1)
    jobs = max(1, min(args.jobs, len(energy_indices), max_workers))
    if jobs == 1:
        curves = reduce_manifest_frames(items, args, monitor_key)
        return average_groups(curves, args.outlier_zmax)

    batches = split_energy_batches(energy_indices, jobs)
    items_by_batch: list[tuple[int, list[int], list[ManifestItem]]] = []
    for batch_id, batch_energies in enumerate(batches, start=1):
        batch_set = set(batch_energies)
        batch_items = [item for item in items if item.energy_index in batch_set]
        items_by_batch.append((batch_id, batch_energies, batch_items))

    print(f"Parallel reduction enabled: jobs={jobs}, cpu_max={max_workers}, total_energies={len(energy_indices)}")
    for batch_id, batch_energies, batch_items in items_by_batch:
        print(
            f"Worker {batch_id}: energies {batch_energies[0]}-{batch_energies[-1]} "
            f"({len(batch_energies)} energies, {len(batch_items)} frames)"
        )

    args_dict = vars(args).copy()
    averages: list[GroupAverage] = []
    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            future_map = {
                executor.submit(
                    reduce_energy_batch_worker,
                    batch_id,
                    batch_energies,
                    batch_items,
                    args_dict,
                    monitor_key,
                    progress_queue,
                ): (batch_id, batch_energies)
                for batch_id, batch_energies, batch_items in items_by_batch
            }
            pending = set(future_map)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=0.2,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                drain_progress_queue(progress_queue)
                for future in done:
                    batch_id, batch_energies = future_map[future]
                    _, _, batch_averages = future.result()
                    averages.extend(batch_averages)
                    print(
                        f"Completed worker {batch_id}: energies {batch_energies[0]}-{batch_energies[-1]}, "
                        f"group averages={len(batch_averages)}"
                    )
            drain_progress_queue(progress_queue)
    return sorted(averages, key=lambda avg: (avg.energy_index, avg.group_index))


def average_groups(curves: list[FrameCurve], zmax: float) -> list[GroupAverage]:
    """Reject repeated-frame outliers and average accepted frames by group."""
    grouped: dict[tuple[int, int], list[FrameCurve]] = {}
    for curve in curves:
        grouped.setdefault((curve.item.energy_index, curve.item.group_index), []).append(curve)

    averages: list[GroupAverage] = []
    for (energy_index, group_index), group_curves in sorted(grouped.items()):
        totals = np.asarray([curve.total_intensity for curve in group_curves], dtype=float)
        keep = reject_outliers(totals, zmax)
        if np.count_nonzero(keep) == 0:
            raise ValueError(f"All frames rejected for energy {energy_index}, group {group_index}")
        kept_curves = [curve for curve, keep_one in zip(group_curves, keep) if keep_one]
        q_ref = kept_curves[0].q
        kept_stack = np.vstack([resample_to_q(q_ref, curve.q, curve.normalized_intensity) for curve in kept_curves])
        avg_intensity = np.nanmean(kept_stack, axis=0)
        if len(kept_curves) > 1:
            avg_error = np.nanstd(kept_stack, axis=0, ddof=1) / np.sqrt(len(kept_curves))
        else:
            avg_error = np.full_like(avg_intensity, np.nan, dtype=float)
        energy_values = [curve.energy_kev for curve in kept_curves if curve.energy_kev is not None]
        monitor_values = [curve.monitor_value for curve in kept_curves]
        averages.append(
            GroupAverage(
                energy_index=energy_index,
                group_index=group_index,
                q=q_ref,
                energy_kev=float(np.mean(energy_values)) if energy_values else None,
                avg_intensity=avg_intensity,
                avg_error=avg_error,
                frame_count=len(group_curves),
                kept_count=len(kept_curves),
                dropped_count=len(group_curves) - len(kept_curves),
                kept_sequence_indices=[curve.item.sequence_index for curve in kept_curves],
                dropped_sequence_indices=[
                    curve.item.sequence_index for curve, keep_one in zip(group_curves, keep) if not keep_one
                ],
                avg_total_intensity=_integrated_intensity(q_ref, avg_intensity),
                avg_monitor_value=float(np.mean(monitor_values)),
            )
        )
    return averages


def group_lookup(averages: list[GroupAverage]) -> dict[tuple[int, int], GroupAverage]:
    return {(avg.energy_index, avg.group_index): avg for avg in averages}


def energy_header_lines(energy_kev: float | None) -> list[str]:
    if energy_kev is None:
        return []
    return [
        f"Energy={energy_kev}",
        f"Wavelength={12.398419843320026 / energy_kev}",
    ]


def write_group_average(avg: GroupAverage, output_dir: Path) -> Path:
    path = output_dir / "groups" / f"energy_{avg.energy_index:03d}_group_{avg.group_index:02d}_avg.dat"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "energy_index": avg.energy_index,
        "group_index": avg.group_index,
        "energy_kev": avg.energy_kev,
        "frame_count": avg.frame_count,
        "kept_count": avg.kept_count,
        "dropped_count": avg.dropped_count,
        "kept_sequence_indices": avg.kept_sequence_indices,
        "dropped_sequence_indices": avg.dropped_sequence_indices,
        "avg_monitor_value": avg.avg_monitor_value,
    }
    header = "\n".join(
        [
            "ASWAXS group average after bad-frame rejection and monitor normalization",
            *energy_header_lines(avg.energy_kev),
            "metadata_json=" + json.dumps(metadata, sort_keys=True),
            "columns=q I_avg_monitor_normalized I_err_standard_error",
        ]
    )
    np.savetxt(path, np.column_stack([avg.q, avg.avg_intensity, avg.avg_error]), header=header, comments="#")
    return path


def write_summary(averages: list[GroupAverage], output_dir: Path, monitor_key: str, detector: str) -> Path:
    path = output_dir / "group_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "energy_index",
                "group_index",
                "energy_kev",
                "frame_count",
                "kept_count",
                "dropped_count",
                "kept_sequence_indices",
                "dropped_sequence_indices",
                "avg_total_intensity_monitor_normalized",
                "avg_monitor_value",
                "monitor_key",
                "detector",
            ]
        )
        for avg in averages:
            writer.writerow(
                [
                    avg.energy_index,
                    avg.group_index,
                    avg.energy_kev,
                    avg.frame_count,
                    avg.kept_count,
                    avg.dropped_count,
                    " ".join(str(value) for value in avg.kept_sequence_indices),
                    " ".join(str(value) for value in avg.dropped_sequence_indices),
                    avg.avg_total_intensity,
                    avg.avg_monitor_value,
                    monitor_key,
                    detector,
                ]
            )
    return path


def compute_gc_scale(
    gc_curve: np.ndarray,
    q: np.ndarray,
    q_range: tuple[float, float],
    reference_file: Path | None,
) -> tuple[float, float, float]:
    reference_q, reference_i = _load_reference_curve(reference_file)
    qmin, qmax = q_range
    mask = np.isfinite(q) & np.isfinite(gc_curve) & (q >= qmin) & (q <= qmax)
    if np.count_nonzero(mask) < 2:
        raise ValueError("GC curve does not overlap calibration q range.")
    q_window = q[mask]
    gc_window = gc_curve[mask]
    reference_window = np.interp(q_window, reference_q, reference_i)
    measured_area = _integrated_intensity(q_window, gc_window)
    reference_area = _integrated_intensity(q_window, reference_window)
    if not np.isfinite(measured_area) or measured_area <= 0:
        raise ValueError("GC measured area is zero, negative, or invalid.")
    return float(reference_area / measured_area), measured_area, reference_area


def quadrature(*errors: np.ndarray) -> np.ndarray:
    finite_errors = [np.asarray(error, dtype=float) for error in errors if error is not None]
    if not finite_errors:
        raise ValueError("No error arrays were provided for quadrature.")
    variance = np.zeros_like(finite_errors[0], dtype=float)
    for error in finite_errors:
        variance = variance + np.square(error)
    return np.sqrt(variance)


def thickness_normalization_factor(args: argparse.Namespace) -> float:
    capillary_thickness = args.capillary_thickness
    gc_thickness = args.gc_thickness
    if capillary_thickness is None and gc_thickness is None:
        return 1.0
    if capillary_thickness is None or gc_thickness is None:
        raise ValueError("Provide both --capillary-thickness and --gc-thickness for thickness normalization.")
    if capillary_thickness <= 0 or gc_thickness <= 0:
        raise ValueError("Capillary thickness and glassy carbon thickness must be positive.")
    return float(gc_thickness / capillary_thickness)


def build_final_record(
    energy_index: int,
    lookup: dict[tuple[int, int], GroupAverage],
    args: argparse.Namespace,
    gc_reference_file: Path | None,
) -> FinalRecord:
    """Build one corrected sample curve for an energy from averaged role groups."""
    sample = lookup.get((energy_index, args.sample_group))
    if sample is None:
        raise ValueError(f"Missing sample group {args.sample_group} at energy {energy_index}")
    q = sample.q
    sample_curve = sample.avg_intensity
    sample_error = sample.avg_error

    air_curve = None
    air_error = None
    if args.air_group is not None:
        air = lookup.get((energy_index, args.air_group))
        if air is None:
            raise ValueError(f"Missing air group {args.air_group} at energy {energy_index}")
        air_curve = air.avg_intensity if np.allclose(q, air.q, rtol=1e-6, atol=1e-12) else resample_to_q(q, air.q, air.avg_intensity)
        air_error = air.avg_error if np.allclose(q, air.q, rtol=1e-6, atol=1e-12) else resample_to_q(q, air.q, air.avg_error)

    empty_curve = None
    empty_error = None
    if args.empty_group is not None:
        empty = lookup.get((energy_index, args.empty_group))
        if empty is None:
            raise ValueError(f"Missing empty group {args.empty_group} at energy {energy_index}")
        empty_curve = empty.avg_intensity if np.allclose(q, empty.q, rtol=1e-6, atol=1e-12) else resample_to_q(q, empty.q, empty.avg_intensity)
        empty_error = empty.avg_error if np.allclose(q, empty.q, rtol=1e-6, atol=1e-12) else resample_to_q(q, empty.q, empty.avg_error)

    sample_background_curve = np.zeros_like(sample_curve)
    sample_background_error_terms: list[np.ndarray] = []
    sample_background_names: list[str] = []
    if empty_curve is not None:
        sample_background_curve = sample_background_curve + empty_curve
        sample_background_error_terms.append(empty_error)
        sample_background_names.append("empty")
    sample_background_error = (
        quadrature(*sample_background_error_terms) if sample_background_error_terms else np.zeros_like(sample_curve)
    )
    sample_minus_background = sample_curve - sample_background_curve
    sample_minus_background_error = quadrature(sample_error, sample_background_error)

    water_curve = None
    water_error = None
    water_minus_background = None
    water_minus_background_error = None
    sample_minus_water = sample_minus_background
    sample_minus_water_error = sample_minus_background_error
    if args.water_group is not None:
        water = lookup.get((energy_index, args.water_group))
        if water is None:
            raise ValueError(f"Missing water group {args.water_group} at energy {energy_index}")
        water_curve = water.avg_intensity if np.allclose(q, water.q, rtol=1e-6, atol=1e-12) else resample_to_q(q, water.q, water.avg_intensity)
        water_error = water.avg_error if np.allclose(q, water.q, rtol=1e-6, atol=1e-12) else resample_to_q(q, water.q, water.avg_error)
        water_minus_background = water_curve - sample_background_curve
        water_minus_background_error = quadrature(water_error, sample_background_error)
        sample_minus_water = sample_minus_background - water_minus_background
        sample_minus_water_error = quadrature(sample_minus_background_error, water_minus_background_error)

    gc_background_curve = np.zeros_like(sample_curve)
    gc_background_error_terms: list[np.ndarray] = []
    gc_background_names: list[str] = []
    if air_curve is not None:
        gc_background_curve = gc_background_curve + air_curve
        gc_background_error_terms.append(air_error)
        gc_background_names.append("air")
    gc_background_error = quadrature(*gc_background_error_terms) if gc_background_error_terms else np.zeros_like(sample_curve)
    gc_curve = None
    gc_error = None
    gc_minus_background = None
    gc_minus_background_error = None
    scale_factor = None
    thickness_factor = 1.0
    absolute_curve = None
    absolute_error = None
    if args.gc_group is not None:
        thickness_factor = thickness_normalization_factor(args)
        gc = lookup.get((energy_index, args.gc_group))
        if gc is None:
            raise ValueError(f"Missing glassy carbon group {args.gc_group} at energy {energy_index}")
        gc_curve = gc.avg_intensity if np.allclose(q, gc.q, rtol=1e-6, atol=1e-12) else resample_to_q(q, gc.q, gc.avg_intensity)
        gc_error = gc.avg_error if np.allclose(q, gc.q, rtol=1e-6, atol=1e-12) else resample_to_q(q, gc.q, gc.avg_error)
        gc_minus_background = gc_curve - gc_background_curve
        gc_minus_background_error = quadrature(gc_error, gc_background_error)
        scale_factor, measured_area, reference_area = compute_gc_scale(
            gc_minus_background,
            q,
            tuple(args.gc_q_range),
            gc_reference_file,
        )
        combined_scale_factor = scale_factor * thickness_factor
        absolute_curve = sample_minus_water * combined_scale_factor
        absolute_error = np.abs(combined_scale_factor) * sample_minus_water_error
    else:
        measured_area = None
        reference_area = None
        combined_scale_factor = None

    final_before_fluorescence = absolute_curve if absolute_curve is not None else sample_minus_water
    final_error = absolute_error if absolute_error is not None else sample_minus_water_error

    component_columns = [q, sample_curve, sample_error]
    component_names = ["q", "I_sample_avg_norm", "I_sample_err"]
    if air_curve is not None:
        component_columns.extend([air_curve, air_error])
        component_names.extend(["I_air_avg_norm", "I_air_err"])
    if empty_curve is not None:
        component_columns.extend([empty_curve, empty_error])
        component_names.extend(["I_empty_avg_norm", "I_empty_err"])
    if water_curve is not None:
        component_columns.extend([water_curve, water_error])
        component_names.extend(["I_water_avg_norm", "I_water_err"])
    if sample_background_names:
        component_columns.extend([sample_background_curve, sample_background_error])
        component_names.extend(["I_sample_background", "I_sample_background_err"])
    component_columns.extend([sample_minus_background, sample_minus_background_error])
    component_names.extend(["I_sample_minus_empty", "I_sample_minus_empty_err"])
    if water_minus_background is not None:
        component_columns.extend([water_minus_background, water_minus_background_error])
        component_names.extend(["I_water_minus_empty", "I_water_minus_empty_err"])
        component_columns.extend([sample_minus_water, sample_minus_water_error])
        component_names.extend(["I_sample_minus_water_after_empty", "I_sample_minus_water_after_empty_err"])
    if gc_curve is not None:
        if gc_background_names:
            component_columns.extend([gc_background_curve, gc_background_error])
            component_names.extend(["I_gc_background", "I_gc_background_err"])
        component_columns.extend([gc_curve, gc_error, gc_minus_background, gc_minus_background_error])
        component_names.extend(["I_gc_avg_norm", "I_gc_err", "I_gc_minus_air", "I_gc_minus_air_err"])
    if absolute_curve is not None:
        component_columns.extend([absolute_curve, absolute_error])
        component_names.extend(["I_sample_minus_water_absolute_gc_thickness_norm", "I_sample_minus_water_absolute_gc_thickness_norm_err"])

    metadata = {
        "energy_index": energy_index,
        "energy_kev": sample.energy_kev,
        "sample_group": args.sample_group,
        "air_group": args.air_group,
        "empty_group": args.empty_group,
        "water_group": args.water_group,
        "gc_group": args.gc_group,
        "sample_background_terms": sample_background_names,
        "gc_background_terms": gc_background_names,
        "correction_order": "sample-empty; water-empty; sample_corrected-water_corrected; gc-air; GC scale; thickness normalization",
        "gc_scale_factor": scale_factor,
        "thickness_normalization_factor": thickness_factor,
        "combined_scale_factor": combined_scale_factor,
        "capillary_thickness": args.capillary_thickness,
        "gc_thickness": args.gc_thickness,
        "gc_measured_area": measured_area,
        "gc_reference_area": reference_area,
        "error_model": "standard error of kept normalized frames; subtraction errors by quadrature; sample and water are empty-subtracted; GC is air-subtracted; GC scale uncertainty not included",
    }
    return FinalRecord(
        energy_index=energy_index,
        energy_kev=sample.energy_kev,
        q=q,
        final_before_fluorescence=final_before_fluorescence,
        final_error=final_error,
        component_columns=component_columns,
        component_names=component_names,
        metadata=metadata,
    )


def write_final_sample_outputs(averages: list[GroupAverage], args: argparse.Namespace, output_dir: Path) -> list[FinalOutput]:
    if args.sample_group is None:
        return []
    lookup = group_lookup(averages)
    energy_indices = sorted({avg.energy_index for avg in averages})
    outputs: list[FinalOutput] = []
    gc_reference_file = Path(args.gc_reference_file).expanduser().resolve() if args.gc_reference_file else None
    records = [build_final_record(energy_index, lookup, args, gc_reference_file) for energy_index in energy_indices]

    fluorescence_level = None
    fluorescence_reference_energy_index = None
    fluorescence_reference_energy_kev = None
    if args.subtract_fluorescence:
        if args.fluorescence_level is not None:
            fluorescence_level = args.fluorescence_level
        elif args.fluorescence_reference == "latest":
            reference = records[-1]
            fluorescence_level = estimate_constant_fluorescence(
                reference.q,
                reference.final_before_fluorescence,
                tuple(args.fluorescence_q_range),
            )
            fluorescence_reference_energy_index = reference.energy_index
            fluorescence_reference_energy_kev = reference.energy_kev
        print(
            "Fluorescence subtraction: "
            f"reference={args.fluorescence_reference}, q_range={tuple(args.fluorescence_q_range)}, "
            f"level={fluorescence_level}"
        )

    for record in records:
        if args.subtract_fluorescence:
            if args.fluorescence_reference == "each" and args.fluorescence_level is None:
                fluorescence_level_one = estimate_constant_fluorescence(
                    record.q,
                    record.final_before_fluorescence,
                    tuple(args.fluorescence_q_range),
                )
                fluorescence_reference_energy_index_one = record.energy_index
                fluorescence_reference_energy_kev_one = record.energy_kev
            else:
                fluorescence_level_one = fluorescence_level
                fluorescence_reference_energy_index_one = fluorescence_reference_energy_index
                fluorescence_reference_energy_kev_one = fluorescence_reference_energy_kev
            final_curve = record.final_before_fluorescence - fluorescence_level_one
        else:
            fluorescence_level_one = None
            fluorescence_reference_energy_index_one = None
            fluorescence_reference_energy_kev_one = None
            final_curve = record.final_before_fluorescence

        component_columns = list(record.component_columns)
        component_names = list(record.component_names)
        if args.subtract_fluorescence:
            component_columns.append(np.full_like(record.q, fluorescence_level_one, dtype=float))
            component_names.append("I_fluorescence_background")
        component_columns.extend([final_curve, record.final_error])
        component_names.extend(["I_final", "I_final_err"])

        metadata = dict(record.metadata)
        metadata.update(
            {
                "fluorescence_background": fluorescence_level_one,
                "fluorescence_q_range": list(args.fluorescence_q_range),
                "fluorescence_reference": args.fluorescence_reference if args.subtract_fluorescence else None,
                "fluorescence_reference_energy_index": fluorescence_reference_energy_index_one,
                "fluorescence_reference_energy_kev": fluorescence_reference_energy_kev_one,
            }
        )

        path = output_dir / "final" / f"energy_{record.energy_index:03d}_sample_final.dat"
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "\n".join(
            [
                "ASWAXS final per-energy sample curve",
                *energy_header_lines(record.energy_kev),
                "metadata_json=" + json.dumps(metadata, sort_keys=True),
                "columns=q I_final I_final_err",
            ]
        )
        np.savetxt(path, np.column_stack([record.q, final_curve, record.final_error]), header=header, comments="#")
        component_path = output_dir / "components" / f"energy_{record.energy_index:03d}_sample_components.dat"
        component_path.parent.mkdir(parents=True, exist_ok=True)
        component_header = "\n".join(
            [
                "ASWAXS final per-energy sample curve with intermediate components",
                *energy_header_lines(record.energy_kev),
                "metadata_json=" + json.dumps(metadata, sort_keys=True),
                "columns=" + " ".join(component_names),
            ]
        )
        np.savetxt(component_path, np.column_stack(component_columns), header=component_header, comments="#")
        outputs.append(
            FinalOutput(
                path=path,
                component_path=component_path,
                energy_index=record.energy_index,
                energy_kev=record.energy_kev,
                q=record.q,
                I=final_curve,
                sigma_I=record.final_error,
                component_names=component_names,
                component_columns=component_columns,
                metadata=metadata,
            )
        )
    return outputs


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cpu_max = max(1, os.cpu_count() or 1)
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1.")
    if args.jobs > cpu_max:
        print(f"Requested jobs={args.jobs} exceeds system CPU count {cpu_max}; using {cpu_max}.")
        args.jobs = cpu_max

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_manifest(args, output_dir)
    args.manifest = str(manifest_path)
    items = read_manifest(manifest_path)
    detector = infer_detector(items, args.detector)
    monitor_key = args.monitor_key or default_monitor_key(detector)

    print("ASWAXS sequence reduction")
    print(f"Manifest: {manifest_path}")
    print(f"Detector: {detector}")
    print(f"Monitor normalization key: {monitor_key}")
    print(f"Jobs: {args.jobs}")
    print("Order: integrate frames -> reject bad frames by total intensity -> normalize kept frames -> average groups")

    averages = reduce_manifest_frames_parallel(items, args, monitor_key)
    for avg in averages:
        write_group_average(avg, output_dir)
    summary_path = write_summary(averages, output_dir, monitor_key, detector)
    final_outputs = write_final_sample_outputs(averages, args, output_dir)
    analysis_path = Path(args.analysis_h5).expanduser().resolve() if args.analysis_h5 else output_dir / "analysis.h5"
    _write_sequence_analysis_h5(
        analysis_path=analysis_path,
        manifest_path=manifest_path,
        items=items,
        averages=averages,
        final_outputs=final_outputs,
        args=args,
        monitor_key=monitor_key,
        detector=detector,
        summary_path=summary_path,
    )

    print(f"Wrote group summary: {summary_path}")
    print(f"Wrote {len(averages)} group-average files.")
    print(f"Wrote analysis HDF5: {analysis_path}")
    if final_outputs:
        print(f"Wrote {len(final_outputs)} final sample files.")
    else:
        print("No final sample files written because --sample-group was not provided.")
    return 0


def _write_sequence_analysis_h5(
    analysis_path: Path,
    manifest_path: Path,
    items: list[ManifestItem],
    averages: list[GroupAverage],
    final_outputs: list[FinalOutput],
    args: argparse.Namespace,
    monitor_key: str,
    detector: str,
    summary_path: Path,
) -> None:
    """Mirror the sequence reduction products into the structured analysis HDF5."""
    raw_paths = [item.path for item in items]
    data_reference_metadata = {
        "data_detector_path": args.dataset_path,
        "source_frame_indices": [item.sequence_index for item in items],
        "source_frame_count": len(items),
        "notes": f"ASWAXS sequence from manifest {manifest_path}; source data files opened read-only",
    }
    create_analysis_h5_from_data(raw_paths, analysis_path, data_reference_metadata=data_reference_metadata)

    q = averages[0].q if averages else np.asarray([])
    reduction_i = np.vstack([avg.avg_intensity for avg in averages]) if averages else np.empty((0, 0))
    reduction_sigma = np.vstack([avg.avg_error for avg in averages]) if averages else np.empty((0, 0))
    frame_log = _frame_filter_log_from_averages(averages, items, args.outlier_zmax)
    reduction_metadata = {
        "input_h5_file": str(manifest_path),
        "input_data_path": args.dataset_path,
        "output_h5_file": str(analysis_path),
        "output_data_path": "/entry/process_01_reduction/data",
        "n_total_frames": len(items),
        "n_accepted_frames": int(sum(avg.kept_count for avg in averages)),
        "n_rejected_frames": int(sum(avg.dropped_count for avg in averages)),
        "notes": "sequence reduction: integrate frames, monitor normalize, reject outliers, average by energy/group",
    }
    reduction_parameters = {
        "poni_file": str(args.poni),
        "poni_file_hash": file_sha256(args.poni),
        "mask_file": str(args.mask),
        "mask_file_hash": file_sha256(args.mask),
        "q_unit": args.unit,
        "q_min": float(np.nanmin(q)) if q.size else "unknown",
        "q_max": float(np.nanmax(q)) if q.size else "unknown",
        "n_q_bins": args.npt,
        "integration_method": "pyFAI.integrate1d",
        "normalization_method": f"monitor:{monitor_key}",
        "dark_subtraction": False,
        "flatfield_correction": False,
        "solid_angle_correction": "pyFAI_default",
        "polarization_correction": "unknown",
        "error_model": "standard error of accepted monitor-normalized frames",
        "detector": detector,
        "outlier_zmax": args.outlier_zmax,
    }
    write_reduction_to_analysis_h5(
        analysis_path,
        raw_paths[0],
        q,
        reduction_i,
        reduction_sigma,
        reduction_metadata,
        reduction_parameters,
        frame_filter_log=frame_log,
    )

    if not final_outputs:
        return

    final_q = final_outputs[0].q
    energies = np.asarray([np.nan if item.energy_kev is None else item.energy_kev for item in final_outputs], dtype=float)
    final_i = np.vstack([item.I for item in final_outputs])
    final_sigma = np.vstack([item.sigma_I for item in final_outputs])
    component_lookup = [_component_dict(item.component_names, item.component_columns) for item in final_outputs]

    corrected_data = {
        "q": final_q,
        "energy": energies,
        "I_sample_corrected": final_i,
        "sigma_sample_corrected": final_sigma,
        "I_gc_corrected": _stack_component(component_lookup, "I_gc_minus_air"),
        "sigma_gc_corrected": _stack_component(component_lookup, "I_gc_minus_air_err"),
    }
    subtraction_metadata = {
        "input_h5_file": str(analysis_path),
        "input_data_path": "/entry/process_01_reduction/data",
        "output_h5_file": str(analysis_path),
        "output_data_path": "/entry/process_02_background_subtraction/data",
        "notes": f"group summary CSV retained at {summary_path}",
    }
    subtraction_parameters = {
        "gc_background": "air" if args.air_group is not None else "unknown",
        "sample_background": "empty_cell/solvent" if args.empty_group or args.water_group else "unknown",
        "scale_by_I0": True,
        "scale_by_transmission": False,
        "scale_by_exposure_time": False,
        "subtraction_formula": "sample-empty; water-empty; sample_corrected-water_corrected; gc-air",
        "solvent_scale_factor": 1.0,
        "empty_cell_scale_factor": 1.0,
    }
    subtraction_map = {
        "energy": energies,
        "sample_id": args.sample_group,
        "air_id": args.air_group if args.air_group is not None else "unknown",
        "glassy_carbon_id": args.gc_group if args.gc_group is not None else "unknown",
        "empty_cell_id": args.empty_group if args.empty_group is not None else "unknown",
        "solvent_id": args.water_group if args.water_group is not None else "unknown",
    }
    write_background_subtraction_to_analysis_h5(
        analysis_path,
        corrected_data,
        subtraction_metadata,
        subtraction_parameters,
        subtraction_map,
    )

    if args.gc_group is None:
        return

    normalized_data = {
        "q": final_q,
        "energy": energies,
        "I_sample_normalized": final_i,
        "sigma_sample_normalized": final_sigma,
        "I_gc_normalized": _stack_component(component_lookup, "I_gc_minus_air"),
        "sigma_gc_normalized": _stack_component(component_lookup, "I_gc_minus_air_err"),
    }
    normalization_metadata = {
        "input_h5_file": str(analysis_path),
        "input_data_path": "/entry/process_02_background_subtraction/data",
        "output_h5_file": str(analysis_path),
        "output_data_path": "/entry/process_03_glassy_carbon_normalization/data",
    }
    normalization_parameters = {
        "gc_reference_file": str(args.gc_reference_file) if args.gc_reference_file else "NIST_SRM3600_builtin",
        "gc_reference_file_hash": file_sha256(args.gc_reference_file) if args.gc_reference_file else "builtin",
        "reference_units": "differential_scattering_cross_section",
        "q_range_used": list(args.gc_q_range),
        "scale_method": "integrated_area_ratio",
        "absolute_scale": True,
        "uncertainty_propagation": "GC scale uncertainty not propagated",
    }
    normalization_factors = {
        "energy": energies,
        "scale_factor": np.asarray([item.metadata.get("gc_scale_factor", np.nan) for item in final_outputs], dtype=float),
        "scale_uncertainty": np.full_like(energies, np.nan),
        "q_min_used": np.full_like(energies, args.gc_q_range[0]),
        "q_max_used": np.full_like(energies, args.gc_q_range[1]),
        "scale_factor_basis": "glassy_carbon_reference_area / measured_area",
    }
    write_glassy_carbon_normalization_to_analysis_h5(
        analysis_path,
        normalized_data,
        normalization_metadata,
        normalization_parameters,
        normalization_factors,
    )


def _frame_filter_log_from_averages(
    averages: list[GroupAverage],
    items: list[ManifestItem],
    outlier_zmax: float,
) -> dict[str, np.ndarray]:
    item_by_sequence = {item.sequence_index: item for item in items}
    frame_index: list[int] = []
    accepted: list[bool] = []
    reason: list[str] = []
    metric: list[float] = []
    low: list[float] = []
    high: list[float] = []
    for avg in averages:
        for sequence_index in avg.kept_sequence_indices:
            item = item_by_sequence.get(sequence_index)
            frame_index.append(item.frame_index if item else sequence_index)
            accepted.append(True)
            reason.append("")
            metric.append(avg.avg_total_intensity)
            low.append(np.nan)
            high.append(outlier_zmax)
        for sequence_index in avg.dropped_sequence_indices:
            item = item_by_sequence.get(sequence_index)
            frame_index.append(item.frame_index if item else sequence_index)
            accepted.append(False)
            reason.append("total_intensity_outlier")
            metric.append(np.nan)
            low.append(np.nan)
            high.append(outlier_zmax)
    return {
        "frame_index": np.asarray(frame_index, dtype=int),
        "accepted": np.asarray(accepted, dtype=bool),
        "rejection_reason": np.asarray(reason),
        "metric_value": np.asarray(metric, dtype=float),
        "threshold_low": np.asarray(low, dtype=float),
        "threshold_high": np.asarray(high, dtype=float),
    }


def _component_dict(names: list[str], columns: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(column) for name, column in zip(names, columns)}


def _stack_component(rows: list[dict[str, np.ndarray]], name: str) -> np.ndarray:
    if not rows:
        return np.asarray([])
    fallback = np.full_like(rows[0].get("q", np.asarray([])), np.nan, dtype=float)
    return np.vstack([row.get(name, fallback) for row in rows])


if __name__ == "__main__":
    raise SystemExit(main())

