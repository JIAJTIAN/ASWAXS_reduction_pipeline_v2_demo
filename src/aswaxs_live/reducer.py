"""Live-style ASWAXS pipeline v2 demo.

The current production reducer works as a batch program: read a sequence, reduce
all frames, average every group, then write final per-energy products. This demo
keeps that science code and changes only the orchestration. A manifest is replayed
one row at a time to mimic acquisition events:

    frame arrives -> 1D reduction -> group average -> per-energy correction

That makes the trigger behavior testable with existing data before we connect it
to a real file watcher or Bluesky/Kafka stream.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
PLAYGROUND_DIR = PROJECT_DIR.parent
DEFAULT_PIPELINE_ROOT = PLAYGROUND_DIR / "ASWAXS_reduction_pipeline"


def import_v1_pipeline(pipeline_root: Path) -> None:
    """Make the existing project importable without packaging it yet."""
    root = pipeline_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Pipeline root does not exist: {root}")
    sys.path.insert(0, str(root))


@dataclass(frozen=True)
class LiveEvent:
    """One audit record describing what the live scheduler just did."""

    time: str
    event: str
    energy_index: int | None = None
    group_index: int | None = None
    frame_index: int | None = None
    sequence_index: int | None = None
    path: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class SequencePosition:
    """Scientific meaning assigned to one file by acquisition order."""

    sequence_index: int
    energy_index: int
    group_index: int
    frame_index: int


class SequenceAssigner:
    """Map arriving files into energy -> group -> frame order.

    This is the watcher equivalent of the offline manifest. It assumes the
    acquisition writes files in this nested order:

        energy 1, group 1, frame 1
        energy 1, group 1, frame 2
        ...
        energy 1, group 2, frame 1
        ...
        energy 2, group 1, frame 1
    """

    def __init__(self, num_groups: int, num_frames: int, num_energies: int | None = None) -> None:
        if num_groups < 1:
            raise ValueError("--num-groups must be at least 1.")
        if num_frames < 1:
            raise ValueError("--num-frames must be at least 1.")
        if num_energies is not None and num_energies < 1:
            raise ValueError("--num-energies must be at least 1 when provided.")
        self.num_groups = num_groups
        self.num_frames = num_frames
        self.num_energies = num_energies
        self._next_sequence_index = 1

    @property
    def expected_total(self) -> int | None:
        if self.num_energies is None:
            return None
        return self.num_energies * self.num_groups * self.num_frames

    @property
    def assigned_count(self) -> int:
        return self._next_sequence_index - 1

    def is_complete(self) -> bool:
        return self.expected_total is not None and self.assigned_count >= self.expected_total

    def advance_to_sequence_index(self, next_sequence_index: int) -> None:
        """Resume assignment after frames already present in the analysis HDF5."""
        if next_sequence_index < 1:
            raise ValueError("next_sequence_index must be at least 1.")
        self._next_sequence_index = max(self._next_sequence_index, next_sequence_index)

    def next_position(self) -> SequencePosition:
        sequence_index = self._next_sequence_index
        if self.is_complete():
            raise StopIteration("All expected acquisition files have already been assigned.")

        zero_based = sequence_index - 1
        frames_per_energy = self.num_groups * self.num_frames
        energy_index = zero_based // frames_per_energy + 1
        group_index = (zero_based % frames_per_energy) // self.num_frames + 1
        frame_index = zero_based % self.num_frames + 1
        self._next_sequence_index += 1
        return SequencePosition(sequence_index, energy_index, group_index, frame_index)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ASWAXS v2 live orchestration from a manifest replay or watched acquisition folder.",
    )
    parser.add_argument("--manifest", help="Existing sequence_manifest.csv to replay.")
    parser.add_argument("--watch-dir", help="Directory receiving live HDF5 files from acquisition.")
    parser.add_argument(
        "--sample-name",
        default=None,
        help="Sample/run name used for the default analysis HDF5 filename.",
    )
    parser.add_argument("--pattern", default="*.h5", help="Watcher input filename pattern. Default: *.h5")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Watcher polling interval.")
    parser.add_argument("--settle-seconds", type=float, default=2.0, help="File-size stability wait.")
    parser.add_argument("--once", action="store_true", help="Watcher: process current files once, then exit.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing analysis HDF5/event log when present. Default: resume.",
    )
    parser.add_argument(
        "--restart",
        dest="resume",
        action="store_false",
        help="Restart from scratch: overwrite the analysis HDF5 and replace the live event log.",
    )
    parser.add_argument("--num-energies", type=int, default=None, help="Watcher: expected number of energies.")
    parser.add_argument("--num-groups", type=int, default=None, help="Watcher: expected groups per energy.")
    parser.add_argument("--num-frames", type=int, default=None, help="Watcher: expected frames per group.")
    parser.add_argument(
        "--pipeline-root",
        default="",
        help="Optional fallback path to an external older ASWAXS_reduction_pipeline project.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "outputs" / "live_v2_demo"),
        help="Demo output directory.",
    )
    parser.add_argument(
        "--analysis-h5",
        default=None,
        help="Analysis HDF5 output path. Default: output-dir/<sample_name>_analysis.h5.",
    )
    parser.add_argument(
        "--analysis-mode",
        default="asaxs",
        choices=["asaxs", "saxs"],
        help="asaxs: run background/GC/final correction after each energy. saxs: stop at 1D + group averages.",
    )
    parser.add_argument(
        "--write-text-output",
        action="store_true",
        help="Also write legacy .dat curve files. Default is HDF5-only for reduction curves.",
    )
    parser.add_argument("--poni", required=True, help="pyFAI PONI calibration file.")
    parser.add_argument("--mask", required=True, help="Detector mask file.")
    parser.add_argument("--dataset-path", default="entry/data/data", help="Detector image dataset inside each HDF5 file.")
    parser.add_argument("--npt", type=int, default=1000, help="Number of q bins for 1D integration.")
    parser.add_argument("--unit", default="q_A^-1", help="pyFAI radial unit.")
    parser.add_argument("--detector", default="auto", choices=["auto", "Pil300K", "Eig1M"])
    parser.add_argument("--monitor-key", default=None, help="NDAttribute monitor key. Default depends on detector.")
    parser.add_argument("--delta-energy-percent", type=float, default=1e-3)
    parser.add_argument("--outlier-zmax", type=float, default=3.5)
    parser.add_argument("--gc-group", type=int, default=None)
    parser.add_argument("--air-group", type=int, default=None)
    parser.add_argument("--empty-group", type=int, default=None)
    parser.add_argument("--water-group", type=int, default=None)
    parser.add_argument("--sample-group", type=int, default=None)
    parser.add_argument("--gc-reference-file", default=None)
    parser.add_argument("--gc-q-range", nargs=2, type=float, default=[0.03, 0.20])
    parser.add_argument("--capillary-thickness", type=float, default=None)
    parser.add_argument("--gc-thickness", type=float, default=None)
    parser.add_argument("--subtract-fluorescence", action="store_true")
    parser.add_argument("--fluorescence-level", type=float, default=None)
    parser.add_argument("--fluorescence-reference", default="latest", choices=["latest", "each"])
    parser.add_argument("--fluorescence-q-range", nargs=2, type=float, default=[0.8, 1.0])
    parser.add_argument("--limit-energies", type=int, default=None, help="Use only the first N energies for a quick test.")
    parser.add_argument(
        "--limit-frames-per-group",
        type=int,
        default=None,
        help="Use only the first N frames per energy/group for a quick test.",
    )
    return parser


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_event(handle, event: LiveEvent) -> None:
    """Append a machine-readable scheduler event and flush for live tailing."""
    handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
    handle.flush()


def file_is_ready(path: Path, dataset_path: str, settle_seconds: float) -> tuple[bool, str]:
    """Return True only when the detector file is stable and readable.

    The raw HDF5 file is opened read-only here. That is the same rule the
    reducer uses, and it avoids touching a file while the detector is still
    writing it.
    """
    if not path.exists() or not path.is_file():
        return False, "missing"
    first_size = path.stat().st_size
    if first_size <= 0:
        return False, "empty"
    if settle_seconds > 0:
        time.sleep(settle_seconds)
        if not path.exists() or path.stat().st_size != first_size:
            return False, "size-changing"
    try:
        with h5py.File(path, "r") as handle:
            if dataset_path not in handle:
                return False, f"missing dataset {dataset_path}"
            dataset = handle[dataset_path]
            if dataset.size == 0:
                return False, f"empty dataset {dataset_path}"
    except OSError as exc:
        return False, f"hdf5 not readable yet: {exc}"
    return True, "ready"


def inferred_sample_name(args: argparse.Namespace) -> str:
    """Pick a useful sample name for output filenames.

    In live mode the watched directory often points at a detector folder such as
    ``.../<sample>/Eig1M``. In that case the parent folder is the sample name.
    """
    if args.sample_name:
        return sanitize_name(args.sample_name)
    if args.watch_dir:
        watch_dir = Path(args.watch_dir)
        detector_names = {"eig1m", "pil300k", "saxs", "waxs"}
        if watch_dir.name.lower() in detector_names and watch_dir.parent.name:
            return sanitize_name(watch_dir.parent.name)
        if watch_dir.name:
            return sanitize_name(watch_dir.name)
    if args.manifest:
        return sanitize_name(Path(args.manifest).stem.replace("sequence_manifest", "analysis"))
    return "analysis"


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "analysis"


def default_analysis_h5_path(args: argparse.Namespace, output_dir: Path) -> Path:
    name = inferred_sample_name(args)
    if name.lower().endswith("_analysis"):
        filename = f"{name}.h5"
    else:
        filename = f"{name}_analysis.h5"
    return output_dir / filename


def analysis_h5_path_for_args(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.analysis_h5:
        return Path(args.analysis_h5).expanduser().resolve()
    return default_analysis_h5_path(args, output_dir)


def prepare_output_records_for_run(args: argparse.Namespace, output_dir: Path) -> Path:
    """Apply resume/restart behavior before the reducer writes any records.

    Resume keeps the existing analysis HDF5 and appends to the event log.
    Restart removes the old analysis record and old run sidecar records first,
    so the new run cannot mix with stale HDF5 rows or stale GUI monitor lines.
    """
    analysis_path = analysis_h5_path_for_args(args, output_dir)
    if not args.resume:
        for path in [
            analysis_path,
            output_dir / "live_events.jsonl",
            output_dir / "live_replay_manifest.csv",
            output_dir / "live_sequence_manifest.csv",
            output_dir / "group_summary.csv",
        ]:
            if path.exists():
                path.unlink()
                print(f"Restart: removed old run record: {path}")
    return analysis_path


def append_frame_curve_to_analysis_h5(
    analysis_path: Path,
    curve: object,
    monitor_key: str,
) -> int:
    """Append one live single-frame 1D curve to the analysis HDF5 file."""
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    item = curve.item
    q = np.asarray(curve.q, dtype=float)
    intensity = np.asarray(curve.normalized_intensity, dtype=float)
    sigma = np.full_like(intensity, np.nan, dtype=float)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(analysis_path, "a") as handle:
        entry = handle.require_group("entry")
        realtime = entry.require_group("realtime")
        process = realtime.require_group("process_01_reduction")
        process.attrs["NX_class"] = "NXprocess"
        frames = process.require_group("frames")
        frames.attrs["NX_class"] = "NXdata"
        frames.attrs["signal"] = "I_frame_q"
        frames.attrs["axes"] = np.asarray(["frame_number", "q"], dtype="S")
        frames.attrs["notes"] = "Single-frame 1D curves appended as files are reduced live."

        if "q" not in frames:
            frames.create_dataset("q", data=q)
        elif frames["q"].shape == q.shape and not np.allclose(frames["q"][()], q, rtol=1e-7, atol=1e-12):
            if "q_frame_q" not in frames:
                _create_extendable_dataset(frames, "q_frame_q", q.dtype, (0, q.size), (None, q.size))
            _append_row(frames["q_frame_q"], q)

        n_rows = frames["I_frame_q"].shape[0] if "I_frame_q" in frames else 0
        if "I_frame_q" not in frames:
            _create_extendable_dataset(frames, "I_frame_q", intensity.dtype, (0, intensity.size), (None, intensity.size))
            _create_extendable_dataset(frames, "sigma_frame_q", sigma.dtype, (0, sigma.size), (None, sigma.size))
            for name, dtype in [
                ("sequence_index", "i8"),
                ("energy_index", "i8"),
                ("group_index", "i8"),
                ("frame_index", "i8"),
            ]:
                _create_extendable_dataset(frames, name, np.dtype(dtype), (0,), (None,))
            for name in ["energy_kev", "monitor_value", "total_intensity"]:
                _create_extendable_dataset(frames, name, np.dtype("f8"), (0,), (None,))
            _create_extendable_dataset(frames, "source_file", string_dtype, (0,), (None,))
            _create_extendable_dataset(frames, "monitor_key", string_dtype, (0,), (None,))
            _create_extendable_dataset(frames, "qc_status", string_dtype, (0,), (None,))
            frames["qc_status"].attrs["pending_group_qc"] = (
                "Frame was reduced to 1D, but its energy/group has not reached "
                "the averaging trigger yet."
            )
            frames["qc_status"].attrs["accepted"] = (
                "Frame was kept by the group-average outlier filter."
            )
            frames["qc_status"].attrs["rejected_total_intensity"] = (
                "Frame was dropped from the group average by the total-intensity "
                "outlier filter."
            )

        _append_row(frames["I_frame_q"], intensity)
        _append_row(frames["sigma_frame_q"], sigma)
        _append_scalar(frames["sequence_index"], item.sequence_index)
        _append_scalar(frames["energy_index"], item.energy_index)
        _append_scalar(frames["group_index"], item.group_index)
        _append_scalar(frames["frame_index"], item.frame_index)
        _append_scalar(frames["energy_kev"], np.nan if curve.energy_kev is None else curve.energy_kev)
        _append_scalar(frames["monitor_value"], curve.monitor_value)
        _append_scalar(frames["total_intensity"], curve.total_intensity)
        _append_scalar(frames["source_file"], str(item.path))
        _append_scalar(frames["monitor_key"], monitor_key)
        _append_scalar(frames["qc_status"], "pending_group_qc")
        handle.flush()
    return n_rows


def update_frame_qc_status_in_analysis_h5(analysis_path: Path, avg: object) -> None:
    """Mark live frame QC status after group averaging.

    Status meanings:
    pending_group_qc: 1D frame exists, but the group-average trigger has not run.
    accepted: frame was kept by the group-average outlier filter.
    rejected_total_intensity: frame was dropped by the total-intensity filter.
    """
    if not analysis_path.exists():
        return
    frames_path = "/entry/realtime/process_01_reduction/frames"
    with h5py.File(analysis_path, "a") as handle:
        if frames_path not in handle:
            return
        frames = handle[frames_path]
        if "sequence_index" not in frames or "qc_status" not in frames:
            return
        sequence_indices = frames["sequence_index"][()]
        accepted = set(avg.kept_sequence_indices)
        rejected = set(avg.dropped_sequence_indices)
        qc = frames["qc_status"]
        for row, sequence_index in enumerate(sequence_indices):
            if int(sequence_index) in accepted:
                qc[row] = "accepted"
            elif int(sequence_index) in rejected:
                qc[row] = "rejected_total_intensity"
        handle.flush()


def existing_live_frame_paths_and_sequences(analysis_path: Path) -> tuple[set[Path], set[int], int]:
    """Return already-written source files and sequence indices from live HDF5 rows."""
    if not analysis_path.exists():
        return set(), set(), 0
    frames_path = "/entry/realtime/process_01_reduction/frames"
    with h5py.File(analysis_path, "r") as handle:
        if frames_path not in handle:
            return set(), set(), 0
        frames = handle[frames_path]
        source_values = frames["source_file"][()] if "source_file" in frames else []
        sequence_values = frames["sequence_index"][()] if "sequence_index" in frames else []
    paths = {Path(_decode_h5_text(value)).expanduser().resolve() for value in source_values if _decode_h5_text(value)}
    sequences = {int(value) for value in sequence_values}
    return paths, sequences, max(sequences) if sequences else 0


def read_live_frame_items_from_analysis_h5(analysis_path: Path, v1) -> list[object]:
    """Read lightweight manifest items for frames already present in analysis HDF5."""
    if not analysis_path.exists():
        return []
    frames_path = "/entry/realtime/process_01_reduction/frames"
    with h5py.File(analysis_path, "r") as handle:
        if frames_path not in handle:
            return []
        frames = handle[frames_path]
        required = ["sequence_index", "energy_index", "group_index", "frame_index", "source_file"]
        if any(name not in frames for name in required):
            return []
        items = [
            v1.ManifestItem(
                sequence_index=int(sequence_index),
                energy_index=int(energy_index),
                group_index=int(group_index),
                frame_index=int(frame_index),
                path=Path(_decode_h5_text(source_file)).expanduser().resolve(),
            )
            for sequence_index, energy_index, group_index, frame_index, source_file in zip(
                frames["sequence_index"][()],
                frames["energy_index"][()],
                frames["group_index"][()],
                frames["frame_index"][()],
                frames["source_file"][()],
            )
        ]
    return sorted(items, key=lambda item: item.sequence_index)


def restore_state_from_analysis_h5(state: "LivePipelineState") -> set[Path]:
    """Restore processed frame state from the live single-frame table.

    This makes watcher restarts continue after the last written frame instead of
    assigning old files again. It also rebuilds unfinished groups so the group
    average can still use frames that were reduced before the restart.
    """
    if not state.args.resume or not state.analysis_path.exists():
        return set()
    frames_path = "/entry/realtime/process_01_reduction/frames"
    with h5py.File(state.analysis_path, "r") as handle:
        if frames_path not in handle:
            return set()
        frames = handle[frames_path]
        required = ["q", "I_frame_q", "sequence_index", "energy_index", "group_index", "frame_index", "source_file"]
        if any(name not in frames for name in required):
            return set()
        q = np.asarray(frames["q"][()], dtype=float)
        normalized = np.asarray(frames["I_frame_q"][()], dtype=float)
        sequence_indices = np.asarray(frames["sequence_index"][()], dtype=int)
        energy_indices = np.asarray(frames["energy_index"][()], dtype=int)
        group_indices = np.asarray(frames["group_index"][()], dtype=int)
        frame_indices = np.asarray(frames["frame_index"][()], dtype=int)
        source_files = [_decode_h5_text(value) for value in frames["source_file"][()]]
        energy_kev = np.asarray(frames["energy_kev"][()] if "energy_kev" in frames else np.full(len(sequence_indices), np.nan), dtype=float)
        monitor_values = np.asarray(frames["monitor_value"][()] if "monitor_value" in frames else np.ones(len(sequence_indices)), dtype=float)
        total_values = np.asarray(frames["total_intensity"][()] if "total_intensity" in frames else np.full(len(sequence_indices), np.nan), dtype=float)

    restored_curves: list[object] = []
    processed_paths: set[Path] = set()
    for row, sequence_index in enumerate(sequence_indices):
        source_path = Path(source_files[row]).expanduser().resolve()
        processed_paths.add(source_path)
        item = state.v1.ManifestItem(
            sequence_index=int(sequence_index),
            energy_index=int(energy_indices[row]),
            group_index=int(group_indices[row]),
            frame_index=int(frame_indices[row]),
            path=source_path,
        )
        monitor_value = float(monitor_values[row]) if np.isfinite(monitor_values[row]) else 1.0
        curve = state.v1.FrameCurve(
            item=item,
            energy_kev=None if not np.isfinite(energy_kev[row]) else float(energy_kev[row]),
            monitor_value=monitor_value,
            q=q,
            intensity=normalized[row] * monitor_value,
            total_intensity=float(total_values[row]) if np.isfinite(total_values[row]) else float(np.nan),
            normalized_intensity=normalized[row],
        )
        restored_curves.append(curve)

    state.items = sorted([curve.item for curve in restored_curves], key=lambda item: item.sequence_index)
    grouped: dict[tuple[int, int], list[object]] = defaultdict(list)
    for curve in restored_curves:
        grouped[(curve.item.energy_index, curve.item.group_index)].append(curve)

    for key, curves in grouped.items():
        expected = state.expected_frame_counts.get(key)
        if expected is not None and len(curves) >= expected:
            averages = state.v1.average_groups(curves, state.runtime_args.outlier_zmax)
            if averages:
                state.completed_averages[key] = averages[0]
                update_frame_qc_status_in_analysis_h5(state.analysis_path, averages[0])
        else:
            state.pending_group_curves[key].extend(sorted(curves, key=lambda curve: curve.item.sequence_index))

    for energy_index, groups in state.expected_energy_groups.items():
        energy_key_set = {(energy_index, group) for group in groups}
        if energy_key_set and energy_key_set.issubset(state.completed_averages):
            state.completed_energies.add(energy_index)
            if state.args.analysis_mode == "asaxs":
                energy_averages = [state.completed_averages[group_key] for group_key in sorted(energy_key_set)]
                state.completed_final_outputs.extend(
                    build_final_outputs_for_h5(state.v1, energy_averages, state.runtime_args, state.output_dir)
                )

    state.last_analysis_path = state.analysis_path
    state.analysis_dirty = False
    return processed_paths


def _decode_h5_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _create_extendable_dataset(group: h5py.Group, name: str, dtype, shape: tuple[int, ...], maxshape: tuple[object, ...]) -> None:
    group.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype, chunks=True)


def _append_row(dataset: h5py.Dataset, values: np.ndarray) -> None:
    row = dataset.shape[0]
    dataset.resize((row + 1, *dataset.shape[1:]))
    dataset[row, ...] = values


def _append_scalar(dataset: h5py.Dataset, value: object) -> None:
    row = dataset.shape[0]
    dataset.resize((row + 1,))
    dataset[row] = value


def filter_manifest_items(items: list[object], limit_energies: int | None, limit_frames: int | None) -> list[object]:
    """Apply demo-size limits while preserving acquisition order."""
    selected_energy_indices = sorted({item.energy_index for item in items})
    if limit_energies is not None:
        selected_energy_indices = selected_energy_indices[:limit_energies]
    selected_energy_set = set(selected_energy_indices)

    frame_counts: dict[tuple[int, int], int] = defaultdict(int)
    filtered: list[object] = []
    for item in sorted(items, key=lambda value: value.sequence_index):
        if item.energy_index not in selected_energy_set:
            continue
        key = (item.energy_index, item.group_index)
        if limit_frames is not None and frame_counts[key] >= limit_frames:
            continue
        frame_counts[key] += 1
        filtered.append(item)
    return filtered


def expected_counts_by_group(items: Iterable[object]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for item in items:
        counts[(item.energy_index, item.group_index)] += 1
    return dict(counts)


def expected_groups_by_energy(items: Iterable[object]) -> dict[int, set[int]]:
    groups: dict[int, set[int]] = defaultdict(set)
    for item in items:
        groups[item.energy_index].add(item.group_index)
    return dict(groups)


def write_live_manifest(items: list[object], path: Path) -> Path:
    """Save the exact replay subset so the demo can be reproduced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence_index", "energy_index", "group_index", "frame_index", "hdf5_path"])
        for item in items:
            writer.writerow([item.sequence_index, item.energy_index, item.group_index, item.frame_index, item.path])
    return path


def make_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the small args namespace expected by the existing reducer functions."""
    return argparse.Namespace(
        poni=args.poni,
        mask=args.mask,
        dataset_path=args.dataset_path,
        npt=args.npt,
        unit=args.unit,
        delta_energy_percent=args.delta_energy_percent,
        outlier_zmax=args.outlier_zmax,
        gc_group=args.gc_group,
        air_group=args.air_group,
        empty_group=args.empty_group,
        water_group=args.water_group,
        sample_group=args.sample_group,
        gc_reference_file=args.gc_reference_file,
        gc_q_range=args.gc_q_range,
        capillary_thickness=args.capillary_thickness,
        gc_thickness=args.gc_thickness,
        subtract_fluorescence=args.subtract_fluorescence,
        fluorescence_level=args.fluorescence_level,
        fluorescence_reference=args.fluorescence_reference,
        fluorescence_q_range=args.fluorescence_q_range,
        write_text_output=args.write_text_output,
    )


def load_v1_functions(pipeline_root: Path | None = None) -> argparse.Namespace:
    """Import the reduction functions used by the live scheduler.

    The v2 project now carries a local copy of the reduction core so the GitHub
    repo can run by itself. ``pipeline_root`` remains as a fallback override for
    comparing against an external older project.
    """
    try:
        from aswaxs_live.core.reduce_aswaxs_sequence import (  # pylint: disable=import-outside-toplevel
            FinalOutput,
            FrameCurve,
            ManifestItem,
            _write_sequence_analysis_h5,
            average_groups,
            build_final_record,
            default_monitor_key,
            estimate_constant_fluorescence,
            infer_detector,
            read_manifest,
            reduce_manifest_frames,
            write_final_sample_outputs,
            write_group_average,
            write_summary,
        )
    except ImportError:
        if pipeline_root is None:
            raise
        import_v1_pipeline(pipeline_root)
        from reduce_aswaxs_sequence import (  # pylint: disable=import-error,import-outside-toplevel
            FinalOutput,
            FrameCurve,
            ManifestItem,
            _write_sequence_analysis_h5,
            average_groups,
            build_final_record,
            default_monitor_key,
            estimate_constant_fluorescence,
            infer_detector,
            read_manifest,
            reduce_manifest_frames,
            write_final_sample_outputs,
            write_group_average,
            write_summary,
        )

    return argparse.Namespace(
        FinalOutput=FinalOutput,
        FrameCurve=FrameCurve,
        ManifestItem=ManifestItem,
        _write_sequence_analysis_h5=_write_sequence_analysis_h5,
        average_groups=average_groups,
        build_final_record=build_final_record,
        default_monitor_key=default_monitor_key,
        estimate_constant_fluorescence=estimate_constant_fluorescence,
        infer_detector=infer_detector,
        read_manifest=read_manifest,
        reduce_manifest_frames=reduce_manifest_frames,
        write_final_sample_outputs=write_final_sample_outputs,
        write_group_average=write_group_average,
        write_summary=write_summary,
    )


def build_final_outputs_for_h5(
    v1,
    averages: list[object],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[object]:
    """Compute final ASAXS outputs without writing legacy text files."""
    if args.write_text_output:
        return v1.write_final_sample_outputs(averages, args, output_dir)
    if args.sample_group is None:
        return []

    lookup = {(avg.energy_index, avg.group_index): avg for avg in averages}
    energy_indices = sorted({avg.energy_index for avg in averages})
    gc_reference_file = Path(args.gc_reference_file).expanduser().resolve() if args.gc_reference_file else None
    records = [v1.build_final_record(energy_index, lookup, args, gc_reference_file) for energy_index in energy_indices]

    fluorescence_level = None
    fluorescence_reference_energy_index = None
    fluorescence_reference_energy_kev = None
    if args.subtract_fluorescence:
        if args.fluorescence_level is not None:
            fluorescence_level = args.fluorescence_level
        elif args.fluorescence_reference == "latest":
            reference = records[-1]
            fluorescence_level = v1.estimate_constant_fluorescence(
                reference.q,
                reference.final_before_fluorescence,
                tuple(args.fluorescence_q_range),
            )
            fluorescence_reference_energy_index = reference.energy_index
            fluorescence_reference_energy_kev = reference.energy_kev

    outputs: list[object] = []
    for record in records:
        if args.subtract_fluorescence:
            if args.fluorescence_reference == "each" and args.fluorescence_level is None:
                fluorescence_level_one = v1.estimate_constant_fluorescence(
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

        outputs.append(
            v1.FinalOutput(
                path=output_dir / "h5_only" / f"energy_{record.energy_index:03d}_sample_final.dat",
                component_path=output_dir / "h5_only" / f"energy_{record.energy_index:03d}_sample_components.dat",
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


class LivePipelineState:
    """Stateful trigger engine shared by manifest replay and folder watching."""

    def __init__(
        self,
        args: argparse.Namespace,
        output_dir: Path,
        event_log,
        v1,
        expected_frame_counts: dict[tuple[int, int], int],
        expected_energy_groups: dict[int, set[int]],
        monitor_key: str,
        detector: str,
    ) -> None:
        self.args = args
        self.output_dir = output_dir
        self.event_log = event_log
        self.v1 = v1
        self.expected_frame_counts = expected_frame_counts
        self.expected_energy_groups = expected_energy_groups
        self.monitor_key = monitor_key
        self.detector = detector
        self.runtime_args = make_runtime_args(args)
        self.pending_group_curves: dict[tuple[int, int], list[object]] = defaultdict(list)
        self.completed_averages: dict[tuple[int, int], object] = {}
        self.completed_energies: set[int] = set()
        self.completed_final_outputs: list[object] = []
        self.items: list[object] = []
        self.analysis_path = analysis_h5_path_for_args(self.args, self.output_dir)
        self.analysis_dirty = False
        self.last_analysis_path: Path | None = None

    def process_item(self, item: object) -> None:
        self.items.append(item)
        self.analysis_dirty = True
        key = (item.energy_index, item.group_index)
        write_event(
            self.event_log,
            LiveEvent(
                time=now_iso(),
                event="frame_arrived",
                energy_index=item.energy_index,
                group_index=item.group_index,
                frame_index=item.frame_index,
                sequence_index=item.sequence_index,
                path=str(item.path),
            ),
        )

        curves = self.v1.reduce_manifest_frames([item], self.runtime_args, self.monitor_key)
        frame_rows = [append_frame_curve_to_analysis_h5(self.analysis_path, curve, self.monitor_key) for curve in curves]
        self.pending_group_curves[key].extend(curves)
        frame_message = f"{len(self.pending_group_curves[key])}/{self.expected_frame_counts[key]} frames ready for group"
        if frame_rows:
            frame_message += f"; live H5 row {frame_rows[-1] + 1}"
        write_event(
            self.event_log,
            LiveEvent(
                time=now_iso(),
                event="frame_reduced_1d",
                energy_index=item.energy_index,
                group_index=item.group_index,
                frame_index=item.frame_index,
                sequence_index=item.sequence_index,
                path=str(self.analysis_path),
                message=frame_message,
            ),
        )

        if len(self.pending_group_curves[key]) == self.expected_frame_counts[key]:
            self._complete_group(key)
        self._complete_energy_if_ready(item.energy_index)

    def _complete_group(self, key: tuple[int, int]) -> None:
        averages = self.v1.average_groups(self.pending_group_curves[key], self.runtime_args.outlier_zmax)
        if len(averages) != 1:
            raise RuntimeError(f"Expected one completed average for {key}, got {len(averages)}")
        avg = averages[0]
        self.completed_averages[key] = avg
        avg_path = self.v1.write_group_average(avg, self.output_dir) if self.args.write_text_output else None
        update_frame_qc_status_in_analysis_h5(self.analysis_path, avg)
        write_event(
            self.event_log,
            LiveEvent(
                time=now_iso(),
                event="group_average_written",
                energy_index=avg.energy_index,
                group_index=avg.group_index,
                path=str(avg_path) if avg_path else None,
                message=f"kept={avg.kept_count}, dropped={avg.dropped_count}",
            ),
        )

    def _complete_energy_if_ready(self, energy_index: int) -> None:
        energy_key_set = {(energy_index, group) for group in self.expected_energy_groups[energy_index]}
        if energy_index in self.completed_energies or not energy_key_set.issubset(self.completed_averages):
            return
        self.completed_energies.add(energy_index)
        energy_averages = [self.completed_averages[group_key] for group_key in sorted(energy_key_set)]
        if self.args.analysis_mode == "saxs":
            self.v1.write_summary(list(self.completed_averages.values()), self.output_dir, self.monitor_key, self.detector)
            self.write_analysis_h5()
            write_event(
                self.event_log,
                LiveEvent(
                    time=now_iso(),
                    event="energy_batch_saxs_completed",
                    energy_index=energy_index,
                    message=f"{len(energy_averages)} groups averaged; ASAXS correction skipped",
                ),
            )
            return

        write_event(
            self.event_log,
            LiveEvent(
                time=now_iso(),
                event="energy_batch_asaxs_started",
                energy_index=energy_index,
                message=f"{len(energy_averages)} groups ready",
            ),
        )
        final_outputs = build_final_outputs_for_h5(self.v1, energy_averages, self.runtime_args, self.output_dir)
        self.completed_final_outputs.extend(final_outputs)
        self.analysis_dirty = True
        self.v1.write_summary(list(self.completed_averages.values()), self.output_dir, self.monitor_key, self.detector)
        self.write_analysis_h5()
        write_event(
            self.event_log,
            LiveEvent(
                time=now_iso(),
                event="energy_batch_asaxs_completed",
                energy_index=energy_index,
                message=f"final_outputs={len(final_outputs)}",
            ),
        )

    def write_analysis_h5(self) -> Path | None:
        if not self.completed_averages:
            return None
        if not self.analysis_dirty and self.last_analysis_path is not None:
            return self.last_analysis_path
        manifest_path = write_live_manifest(self.items, self.output_dir / "live_sequence_manifest.csv")
        summary_path = self.v1.write_summary(
            list(self.completed_averages.values()),
            self.output_dir,
            self.monitor_key,
            self.detector,
        )
        analysis_path = self.analysis_path
        self.v1._write_sequence_analysis_h5(
            analysis_path=analysis_path,
            manifest_path=manifest_path,
            items=self.items,
            averages=list(self.completed_averages.values()),
            final_outputs=self.completed_final_outputs,
            args=self.runtime_args,
            monitor_key=self.monitor_key,
            detector=self.detector,
            summary_path=summary_path,
        )
        self.analysis_dirty = False
        self.last_analysis_path = analysis_path
        return analysis_path


def replay_live_pipeline(args: argparse.Namespace) -> int:
    v1 = load_v1_functions(Path(args.pipeline_root) if args.pipeline_root else None)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = prepare_output_records_for_run(args, output_dir)
    event_log_path = output_dir / "live_events.jsonl"
    manifest_path = Path(args.manifest).expanduser().resolve()

    all_items = v1.read_manifest(manifest_path)
    items = filter_manifest_items(all_items, args.limit_energies, args.limit_frames_per_group)
    if not items:
        raise ValueError("No manifest rows remain after applying demo limits.")

    detector = v1.infer_detector(items, args.detector)
    monitor_key = args.monitor_key or v1.default_monitor_key(detector)
    replay_manifest_path = write_live_manifest(items, output_dir / "live_replay_manifest.csv")

    expected_frame_counts = expected_counts_by_group(items)
    expected_energy_groups = expected_groups_by_energy(items)

    print("ASWAXS live v2 demo")
    print(f"Replay manifest: {replay_manifest_path}")
    print(f"Detector: {detector}")
    print(f"Monitor normalization key: {monitor_key}")
    print(f"Frames in manifest: {len(all_items)}")
    print(f"Frames to replay: {len(items)}")
    if len(items) != len(all_items):
        print("Replay limits are active; only replayed frames are written to the live single-frame HDF5 table.")
    print(f"Output: {output_dir}")
    print(f"Run behavior: {'resume existing analysis HDF5' if args.resume else 'restart and overwrite analysis HDF5'}")

    log_mode = "a" if args.resume and event_log_path.exists() else "w"
    with event_log_path.open(log_mode, encoding="utf-8") as event_log:
        state = LivePipelineState(
            args=args,
            output_dir=output_dir,
            event_log=event_log,
            v1=v1,
            expected_frame_counts=expected_frame_counts,
            expected_energy_groups=expected_energy_groups,
            monitor_key=monitor_key,
            detector=detector,
        )
        restored_paths = restore_state_from_analysis_h5(state)
        restored_sequences = {item.sequence_index for item in state.items}
        if restored_paths or restored_sequences:
            print(f"Resume: restored {len(restored_sequences)} previously reduced frame rows from {state.analysis_path}")
        remaining_items = [
            item
            for item in sorted(items, key=lambda value: value.sequence_index)
            if item.sequence_index not in restored_sequences and item.path.expanduser().resolve() not in restored_paths
        ]
        if len(remaining_items) != len(items):
            print(f"Resume: skipping {len(items) - len(remaining_items)} already reduced manifest frames")
        for item in remaining_items:
            state.process_item(item)
        analysis_path = state.write_analysis_h5() or analysis_path

    print(f"Wrote live event log: {event_log_path}")
    print(f"Wrote analysis HDF5: {analysis_path}")
    print(f"Completed group averages: {len(state.completed_averages)}")
    print(f"Completed energy batches: {len(state.completed_energies)}")
    return 0


def watch_live_pipeline(args: argparse.Namespace) -> int:
    """Watch an acquisition folder and assign sequence meaning by file order."""
    if args.num_groups is None or args.num_frames is None:
        raise ValueError("Watcher mode requires --num-groups and --num-frames.")

    v1 = load_v1_functions(Path(args.pipeline_root) if args.pipeline_root else None)
    watch_dir = Path(args.watch_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not watch_dir.exists():
        raise FileNotFoundError(f"Missing watch directory: {watch_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = prepare_output_records_for_run(args, output_dir)

    assigner = SequenceAssigner(
        num_groups=args.num_groups,
        num_frames=args.num_frames,
        num_energies=args.num_energies,
    )
    expected_total = assigner.expected_total
    expected_group_set = set(range(1, args.num_groups + 1))
    processed_paths: set[Path] = set()
    detector: str | None = None
    monitor_key: str | None = None
    state: LivePipelineState | None = None
    event_log_path = output_dir / "live_events.jsonl"

    resume_items = read_live_frame_items_from_analysis_h5(analysis_path, v1) if args.resume else []
    if resume_items:
        detector = v1.infer_detector(resume_items, args.detector)
        monitor_key = args.monitor_key or v1.default_monitor_key(detector)
        expected_frame_counts = {
            (item.energy_index, item.group_index): args.num_frames
            for item in resume_items
        }
        if args.num_energies is not None:
            expected_energy_groups = {
                energy_index: expected_group_set
                for energy_index in range(1, args.num_energies + 1)
            }
        else:
            expected_energy_groups = {
                energy_index: expected_group_set
                for energy_index in sorted({item.energy_index for item in resume_items})
            }
        state = LivePipelineState(
            args=args,
            output_dir=output_dir,
            event_log=None,
            v1=v1,
            expected_frame_counts=expected_frame_counts,
            expected_energy_groups=expected_energy_groups,
            monitor_key=monitor_key,
            detector=detector,
        )
        processed_paths = restore_state_from_analysis_h5(state)
        if resume_items:
            assigner.advance_to_sequence_index(max(item.sequence_index for item in resume_items) + 1)

    print("ASWAXS live v2 watcher")
    print(f"Watching: {watch_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Sequence rule: energy -> group -> frame, groups={args.num_groups}, frames/group={args.num_frames}")
    if args.num_energies is not None:
        print(f"Expected total files: {expected_total}")
    print(f"Output: {output_dir}")
    print(f"Run behavior: {'resume existing analysis HDF5' if args.resume else 'restart and overwrite analysis HDF5'}")
    if state is not None:
        print(f"Resume: restored {len(processed_paths)} previously reduced frame rows from {analysis_path}")
        print(f"Resume: next sequence index is {assigner.assigned_count + 1}")

    log_mode = "a" if args.resume and event_log_path.exists() else "w"
    with event_log_path.open(log_mode, encoding="utf-8") as event_log:
        if state is not None:
            state.event_log = event_log
        while True:
            ready_count_this_poll = 0
            for h5_path in sorted(watch_dir.glob(args.pattern)):
                if assigner.is_complete():
                    break
                h5_path = h5_path.resolve()
                if h5_path in processed_paths:
                    continue
                settle_seconds = 0.0 if args.once else args.settle_seconds
                ready, reason = file_is_ready(h5_path, args.dataset_path, settle_seconds)
                if not ready:
                    write_event(
                        event_log,
                        LiveEvent(
                            time=now_iso(),
                            event="file_waiting",
                            path=str(h5_path),
                            message=reason,
                        ),
                    )
                    continue

                try:
                    position = assigner.next_position()
                except StopIteration:
                    break

                item = v1.ManifestItem(
                    sequence_index=position.sequence_index,
                    energy_index=position.energy_index,
                    group_index=position.group_index,
                    frame_index=position.frame_index,
                    path=h5_path,
                )
                if detector is None:
                    detector = v1.infer_detector([item], args.detector)
                    monitor_key = args.monitor_key or v1.default_monitor_key(detector)
                    state = LivePipelineState(
                        args=args,
                        output_dir=output_dir,
                        event_log=event_log,
                        v1=v1,
                        expected_frame_counts={},
                        expected_energy_groups={},
                        monitor_key=monitor_key,
                        detector=detector,
                    )
                    print(f"Detector: {detector}")
                    print(f"Monitor normalization key: {monitor_key}")

                assert state is not None
                state.expected_frame_counts[(position.energy_index, position.group_index)] = args.num_frames
                state.expected_energy_groups[position.energy_index] = expected_group_set
                write_event(
                    event_log,
                    LiveEvent(
                        time=now_iso(),
                        event="file_assigned_sequence",
                        energy_index=position.energy_index,
                        group_index=position.group_index,
                        frame_index=position.frame_index,
                        sequence_index=position.sequence_index,
                        path=str(h5_path),
                    ),
                )
                state.process_item(item)
                processed_paths.add(h5_path)
                ready_count_this_poll += 1

            if assigner.is_complete():
                break
            if args.once:
                break
            if ready_count_this_poll == 0:
                time.sleep(max(0.1, args.poll_seconds))

        analysis_path = state.write_analysis_h5() if state is not None else None

    print(f"Wrote live event log: {event_log_path}")
    if analysis_path is not None:
        print(f"Wrote analysis HDF5: {analysis_path}")
    if state is None:
        print("No HDF5 files were processed.")
    else:
        print(f"Processed files: {len(processed_paths)}")
        print(f"Completed group averages: {len(state.completed_averages)}")
        print(f"Completed energy batches: {len(state.completed_energies)}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.watch_dir and args.manifest:
        raise ValueError("Use either --watch-dir or --manifest, not both.")
    if args.watch_dir:
        return watch_live_pipeline(args)
    if args.manifest:
        return replay_live_pipeline(args)
    raise ValueError("Provide either --manifest for replay mode or --watch-dir for folder-watcher mode.")


if __name__ == "__main__":
    raise SystemExit(main())
