"""Live GUI for watching 1D SAXS/ASAXS output curves.

Run this in a separate terminal while the live reducer is reducing data. The
viewer never touches acquisition HDF5 files; it reads the reducer's analysis HDF5
file and, for older runs, optional legacy text curve outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import h5py
import numpy as np
from PyQt5 import QtCore, QtWidgets

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "live_v2_demo"
CURVE_PATTERNS = {
    "group averages": [("groups/*.dat", "group average")],
    "final curves": [("final/*.dat", "final sample")],
    "saxs": [
        ("saxs_reduction/groups/*.dat", "SAXS group"),
        ("saxs_reduction/final/*.dat", "SAXS final"),
        ("saxs_reduction/components/*.dat", "SAXS component"),
    ],
    "waxs": [
        ("waxs_reduction/groups/*.dat", "WAXS group"),
        ("waxs_reduction/final/*.dat", "WAXS final"),
        ("waxs_reduction/components/*.dat", "WAXS component"),
    ],
    "stitched": [
        ("stitched_sample_final/data/*.dat", "stitched final"),
        ("stitched_groupwise/stitched_groups/**/*.dat", "stitched group"),
        ("stitched_groupwise/scaled_waxs_groups/**/*.dat", "scaled WAXS"),
        ("stitched_groups/**/*.dat", "stitched group"),
        ("scaled_waxs_groups/**/*.dat", "scaled WAXS"),
        ("stitched_groupwise_reduced/final/*.dat", "stitched reduced"),
        ("stitched_groupwise_gc_normalized/final/*.dat", "stitched GC normalized"),
        ("stitched_sample/*.dat", "stitched sample"),
        ("sample_minus_empty_data/*.dat", "stitched diagnostic"),
    ],
    "all": [
        ("groups/*.dat", "group average"),
        ("final/*.dat", "final sample"),
        ("saxs_reduction/groups/*.dat", "SAXS group"),
        ("saxs_reduction/final/*.dat", "SAXS final"),
        ("waxs_reduction/groups/*.dat", "WAXS group"),
        ("waxs_reduction/final/*.dat", "WAXS final"),
        ("stitched_sample_final/data/*.dat", "stitched final"),
        ("stitched_groupwise/stitched_groups/**/*.dat", "stitched group"),
        ("stitched_groupwise/scaled_waxs_groups/**/*.dat", "scaled WAXS"),
        ("stitched_groups/**/*.dat", "stitched group"),
        ("scaled_waxs_groups/**/*.dat", "scaled WAXS"),
        ("stitched_groupwise_reduced/final/*.dat", "stitched reduced"),
        ("stitched_groupwise_gc_normalized/final/*.dat", "stitched GC normalized"),
    ],
}


@dataclass
class CurveFile:
    key: str
    path: str
    kind: str
    label: str
    mtime_ns: int
    size: int


@dataclass
class CurveData:
    q: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray | None
    label: str
    path: str
    metadata: dict[str, object]


@dataclass
class H5FrameTable:
    analysis_h5: Path
    mtime_ns: int
    size: int
    q: np.ndarray
    intensity: np.ndarray
    energy_index: np.ndarray
    group_index: np.ndarray
    frame_index: np.ndarray
    energy_kev: np.ndarray
    qc_status: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show live 1D curves written by the ASWAXS v2 reducer.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Reducer output directory containing groups/ and/or final/ subfolders.",
    )
    parser.add_argument("--refresh-ms", type=int, default=1500, help="Auto-refresh interval in milliseconds.")
    return parser


def parse_header_metadata(path: Path) -> dict[str, object]:
    """Extract the optional metadata_json header from reducer .dat files."""
    metadata: dict[str, object] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith("#"):
                    break
                text = line[1:].strip()
                if text.startswith("metadata_json="):
                    try:
                        metadata = json.loads(text.split("=", 1)[1])
                    except json.JSONDecodeError:
                        metadata = {}
                    break
    except OSError:
        return {}
    return metadata


def curve_label(path: Path, kind: str) -> str:
    """Build a compact plot label from filename and metadata when available."""
    metadata = parse_header_metadata(path)
    energy = metadata.get("energy_index")
    group = metadata.get("group_index")
    energy_kev = metadata.get("energy_kev")
    if energy is not None and group is not None:
        return f"{kind} E{int(energy):03d} G{int(group):02d}"
    if energy is not None:
        label = f"{kind} E{int(energy):03d}"
        if isinstance(energy_kev, int | float):
            label += f" {energy_kev:.4f} keV"
        return label

    match = re.search(r"energy_(\d+).*?group_(\d+)", path.name)
    if match:
        return f"{kind} E{int(match.group(1)):03d} G{int(match.group(2)):02d}"
    match = re.search(r"energy_(\d+)", path.name)
    if match:
        return f"{kind} E{int(match.group(1)):03d}"
    return f"{kind} {path.stem}"


def load_curve(path: Path, label: str) -> CurveData:
    """Read a reducer .dat curve. Invalid/partial files raise ValueError."""
    try:
        data = np.loadtxt(path)
    except Exception as exc:  # noqa: BLE001 - partial files can fail in several ways.
        raise ValueError(f"Could not read {path}: {exc}") from exc
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least q and I columns in {path}")
    sigma = data[:, 2] if data.shape[1] >= 3 else None
    return CurveData(
        q=data[:, 0],
        intensity=data[:, 1],
        sigma=sigma,
        label=label,
        path=str(path),
        metadata=parse_header_metadata(path),
    )


def latest_process_names(handle: h5py.File | h5py.Group, prefix: str) -> list[str]:
    """Return process groups matching a base prefix, sorted by version order."""
    entry = handle.get("entry") if isinstance(handle, h5py.File) else handle
    if entry is None:
        return []
    names = []
    for name in entry:
        if name == prefix or name.startswith(prefix + "_v"):
            names.append(name)

    def version_key(name: str) -> int:
        match = re.search(r"_v(\d+)$", name)
        return int(match.group(1)) if match else 1

    return sorted(names, key=version_key)


def analysis_entry_roots(handle: h5py.File) -> list[tuple[str, str, h5py.Group]]:
    """Return legacy and detector-scoped analysis roots.

    New dual-detector files keep detector records under names such as
    ``/entry/Pil300K`` and ``/entry/Eig1M``. Older single-detector files still
    store processes directly under ``/entry``.
    """
    if "entry" not in handle:
        return []
    entry = handle["entry"]
    roots: list[tuple[str, str, h5py.Group]] = [("", "/entry", entry)]
    for name in sorted(entry):
        if name in {"raw_reference", "realtime", "stitched_averages", "final"} or name.startswith("process_"):
            continue
        child = entry[name]
        if not isinstance(child, h5py.Group):
            continue
        detector = _decode_text(child.attrs.get("detector", name))
        roots.append((f"{detector} ", f"/entry/{name}", child))
    return roots


def h5_curve_records(analysis_h5: Path, category: str) -> list[CurveFile]:
    """Create curve records for known analysis.h5 result datasets."""
    if not analysis_h5.exists():
        return []
    stat = analysis_h5.stat()
    records: list[CurveFile] = []
    with h5py.File(analysis_h5, "r") as handle:
        if category in {"h5 single frames", "analysis h5"}:
            for detector_label, root_path, _root in analysis_entry_roots(handle):
                frames_path = f"{root_path}/realtime/process_01_reduction/frames"
                dataset_path = f"{frames_path}/I_frame_q"
                if f"{frames_path}/q" not in handle or dataset_path not in handle:
                    continue
                i_data = handle[dataset_path]
                rows = i_data.shape[0] if i_data.ndim > 1 else 1
                energy_indices = handle[f"{frames_path}/energy_index"][()] if f"{frames_path}/energy_index" in handle else None
                group_indices = handle[f"{frames_path}/group_index"][()] if f"{frames_path}/group_index" in handle else None
                frame_indices = handle[f"{frames_path}/frame_index"][()] if f"{frames_path}/frame_index" in handle else None
                energies = handle[f"{frames_path}/energy_kev"][()] if f"{frames_path}/energy_kev" in handle else None
                qc_status = handle[f"{frames_path}/qc_status"][()] if f"{frames_path}/qc_status" in handle else None
                for row in range(rows):
                    energy_index = _indexed_value(energy_indices, row, row + 1)
                    group_index = _indexed_value(group_indices, row, 1)
                    frame_index = _indexed_value(frame_indices, row, row + 1)
                    energy_value = _indexed_value(energies, row, np.nan)
                    energy_label = f" {float(energy_value):.4f} keV" if np.isfinite(float(energy_value)) else ""
                    qc_label = _decode_text(_indexed_value(qc_status, row, "pending_group_qc"))
                    records.append(
                        CurveFile(
                            key=f"h5://{analysis_h5}::{dataset_path}[{row}]",
                            path=str(analysis_h5),
                            kind=f"H5 {detector_label}single frame",
                            label=(
                                f"H5 {detector_label}frame{energy_label} E{int(energy_index):03d} "
                                f"G{int(group_index):02d} F{int(frame_index):03d} {qc_label}"
                            ),
                            mtime_ns=stat.st_mtime_ns,
                            size=stat.st_size,
                        )
                    )
        if category in {"h5 group averages", "h5 reduction", "analysis h5"}:
            for detector_label, root_path, root in analysis_entry_roots(handle):
                for process_name in latest_process_names(root, "process_01_reduction")[-1:]:
                    data_path = f"{root_path}/{process_name}/data"
                    if f"{data_path}/q" in handle and f"{data_path}/I" in handle:
                        i_data = handle[f"{data_path}/I"]
                        rows = i_data.shape[0] if i_data.ndim > 1 else 1
                        for row in range(rows):
                            records.append(
                                CurveFile(
                                    key=f"h5://{analysis_h5}::{data_path}/I[{row}]",
                                    path=str(analysis_h5),
                                    kind=f"H5 {detector_label}group average",
                                    label=f"H5 {detector_label}group average {process_name} row {row + 1:03d}",
                                    mtime_ns=stat.st_mtime_ns,
                                    size=stat.st_size,
                                )
                            )
        if category in {"h5 corrected", "analysis h5"}:
            specs = [
                ("process_02_background_subtraction", "I_sample_corrected", "sigma_sample_corrected", "H5 background corrected"),
                ("process_02_background_subtraction", "I_gc_corrected", "sigma_gc_corrected", "H5 GC corrected"),
            ]
            records.extend(_h5_process_dataset_records(handle, analysis_h5, stat, specs))
        if category in {"h5 normalized", "analysis h5"}:
            specs = [
                ("process_03_glassy_carbon_normalization", "I_sample_normalized", "sigma_sample_normalized", "H5 sample normalized"),
                ("process_03_glassy_carbon_normalization", "I_gc_normalized", "sigma_gc_normalized", "H5 GC normalized"),
            ]
            records.extend(_h5_process_dataset_records(handle, analysis_h5, stat, specs))
        if category in {"h5 final", "analysis h5"}:
            for detector_label, root_path, _root in analysis_entry_roots(handle):
                dataset_path = f"{root_path}/final/corrected_I_q_E/I"
                if dataset_path not in handle:
                    continue
                i_data = handle[dataset_path]
                rows = i_data.shape[0] if i_data.ndim > 1 else 1
                for row in range(rows):
                    records.append(
                        CurveFile(
                            key=f"h5://{analysis_h5}::{dataset_path}[{row}]",
                            path=str(analysis_h5),
                            kind=f"H5 {detector_label}final",
                            label=f"H5 {detector_label}final row {row + 1:03d}",
                            mtime_ns=stat.st_mtime_ns,
                            size=stat.st_size,
                        )
                    )
        if category in {"h5 stitched averages", "analysis h5"} and "/entry/stitched_averages/curves" in handle:
            curves = handle["/entry/stitched_averages/curves"]
            for name in sorted(curves):
                curve_path = f"/entry/stitched_averages/curves/{name}"
                if f"{curve_path}/q" not in handle or f"{curve_path}/I" not in handle:
                    continue
                records.append(
                    CurveFile(
                        key=f"h5curve://{analysis_h5}::{curve_path}",
                        path=str(analysis_h5),
                        kind="H5 stitched average",
                        label=f"H5 stitched average {name}",
                        mtime_ns=stat.st_mtime_ns,
                        size=stat.st_size,
                    )
                )
    return records


def _indexed_value(values, row: int, fallback):
    if values is None or np.size(values) <= row:
        return fallback
    return values[row]


def _decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def find_analysis_h5(output_path: Path) -> Path:
    """Find the current batch analysis HDF5 file for an output path."""
    if output_path.is_file() and output_path.suffix.lower() in {".h5", ".hdf5"}:
        return output_path
    candidates = sorted(
        [*output_path.glob("*_analysis.h5"), *output_path.glob("*_analysis.hdf5")],
        key=lambda path: path.stat().st_mtime_ns if path.exists() else 0,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return output_path / "analysis.h5"


def load_h5_frame_table(analysis_h5: Path) -> H5FrameTable | None:
    """Load the live single-frame table used by the raw-frame viewer."""
    if not analysis_h5.exists():
        return None
    frames_path = "/entry/realtime/process_01_reduction/frames"
    stat = analysis_h5.stat()
    with h5py.File(analysis_h5, "r") as handle:
        for _detector_label, root_path, _root in analysis_entry_roots(handle):
            candidate = f"{root_path}/realtime/process_01_reduction/frames"
            if f"{candidate}/q" in handle and f"{candidate}/I_frame_q" in handle:
                frames_path = candidate
                break
        else:
            return None
        frames = handle[frames_path]
        rows = frames["I_frame_q"].shape[0]
        fallback_index = np.arange(1, rows + 1, dtype=int)
        qc_values = frames["qc_status"][()] if "qc_status" in frames else np.asarray([b"pending_group_qc"] * rows)
        return H5FrameTable(
            analysis_h5=analysis_h5,
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            q=np.asarray(frames["q"][()], dtype=float),
            intensity=np.asarray(frames["I_frame_q"][()], dtype=float),
            energy_index=np.asarray(frames["energy_index"][()] if "energy_index" in frames else fallback_index, dtype=int),
            group_index=np.asarray(frames["group_index"][()] if "group_index" in frames else np.ones(rows, dtype=int), dtype=int),
            frame_index=np.asarray(frames["frame_index"][()] if "frame_index" in frames else fallback_index, dtype=int),
            energy_kev=np.asarray(frames["energy_kev"][()] if "energy_kev" in frames else np.full(rows, np.nan), dtype=float),
            qc_status=[_decode_text(value) for value in qc_values],
        )


def _h5_process_dataset_records(
    handle: h5py.File,
    analysis_h5: Path,
    stat,
    specs: list[tuple[str, str, str, str]],
) -> list[CurveFile]:
    records: list[CurveFile] = []
    for process_prefix, intensity_name, _sigma_name, label_prefix in specs:
        for detector_label, root_path, root in analysis_entry_roots(handle):
            for process_name in latest_process_names(root, process_prefix)[-1:]:
                data_path = f"{root_path}/{process_name}/data"
                dataset_path = f"{data_path}/{intensity_name}"
                if f"{data_path}/q" not in handle or dataset_path not in handle:
                    continue
                i_data = handle[dataset_path]
                rows = i_data.shape[0] if i_data.ndim > 1 else 1
                energies = handle[f"{data_path}/energy"][()] if f"{data_path}/energy" in handle else None
                for row in range(rows):
                    energy_label = ""
                    if energies is not None and np.size(energies) > row and np.isfinite(energies[row]):
                        energy_label = f" {float(energies[row]):.4f} keV"
                    records.append(
                        CurveFile(
                            key=f"h5://{analysis_h5}::{dataset_path}[{row}]",
                            path=str(analysis_h5),
                            kind=f"{label_prefix} {detector_label}".strip(),
                            label=f"{label_prefix} {detector_label}{energy_label} {process_name} row {row + 1:03d}".strip(),
                            mtime_ns=stat.st_mtime_ns,
                            size=stat.st_size,
                        )
                    )
    return records


def load_h5_curve(record: CurveFile) -> CurveData:
    """Load one curve from analysis.h5 using a record key."""
    stitched_match = re.match(r"h5curve://(.+)::(.+)$", record.key)
    if stitched_match:
        h5_path = Path(stitched_match.group(1))
        group_path = stitched_match.group(2)
        with h5py.File(h5_path, "r") as handle:
            group = handle[group_path]
            q = np.asarray(group["q"][()], dtype=float)
            intensity = np.asarray(group["I"][()], dtype=float)
            sigma = np.asarray(group["sigma_I"][()], dtype=float) if "sigma_I" in group else None
        return CurveData(q=q, intensity=intensity, sigma=sigma, label=record.label, path=record.key, metadata={})

    match = re.match(r"h5://(.+)::(.+)\[(\d+)\]$", record.key)
    if not match:
        raise ValueError(f"Invalid HDF5 curve key: {record.key}")
    h5_path = Path(match.group(1))
    dataset_path = match.group(2)
    row = int(match.group(3))
    data_group = str(Path(dataset_path).parent).replace("\\", "/")
    with h5py.File(h5_path, "r") as handle:
        q = np.asarray(handle[f"{data_group}/q"][()], dtype=float)
        i_data = np.asarray(handle[dataset_path][()], dtype=float)
        intensity = i_data[row] if i_data.ndim > 1 else i_data
        sigma = None
        sigma_name = _sigma_name_for_dataset(Path(dataset_path).name)
        sigma_path = f"{data_group}/{sigma_name}" if sigma_name else ""
        if sigma_path and sigma_path in handle:
            sigma_data = np.asarray(handle[sigma_path][()], dtype=float)
            sigma = sigma_data[row] if sigma_data.ndim > 1 else sigma_data
    return CurveData(q=q, intensity=intensity, sigma=sigma, label=record.label, path=record.key, metadata={})


def _sigma_name_for_dataset(name: str) -> str | None:
    return {
        "I": "sigma_I",
        "I_frame_q": "sigma_frame_q",
        "I_sample_corrected": "sigma_sample_corrected",
        "I_gc_corrected": "sigma_gc_corrected",
        "I_sample_normalized": "sigma_sample_normalized",
        "I_gc_normalized": "sigma_gc_normalized",
    }.get(name)


class LiveCurveViewer(QtWidgets.QMainWindow):
    """Small plotting GUI that follows output curves as they are written."""

    def __init__(self, output_dir: Path, refresh_ms: int) -> None:
        super().__init__()
        self.setWindowTitle("ASWAXS Live 1D Curves")
        self.resize(1160, 780)
        self.curve_files: dict[str, CurveFile] = {}
        self.curves: dict[str, CurveData] = {}
        self.frame_table: H5FrameTable | None = None
        self._build_ui(output_dir, refresh_ms)
        self._refresh_now()

    def _build_ui(self, output_dir: Path, refresh_ms: int) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QHBoxLayout()
        root.addLayout(controls)

        self.output_dir_edit = QtWidgets.QLineEdit(str(output_dir))
        controls.addWidget(QtWidgets.QLabel("Output"))
        controls.addWidget(self.output_dir_edit, 1)

        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self._browse_output_dir)
        controls.addWidget(browse)

        self.curve_kind_combo = QtWidgets.QComboBox()
        self.curve_kind_combo.addItems(
            ["h5 single frames", "h5 group averages", "h5 stitched averages", "h5 final"]
        )
        self.curve_kind_combo.currentTextChanged.connect(self._refresh_now)
        controls.addWidget(self.curve_kind_combo)

        self.max_curves_spin = QtWidgets.QSpinBox()
        self.max_curves_spin.setRange(1, 500)
        self.max_curves_spin.setValue(50)
        self.max_curves_spin.valueChanged.connect(self._plot_selected)
        controls.addWidget(QtWidgets.QLabel("Max"))
        controls.addWidget(self.max_curves_spin)

        self.log_x_check = QtWidgets.QCheckBox("log q")
        self.log_x_check.setChecked(True)
        self.log_x_check.stateChanged.connect(self._plot_selected)
        controls.addWidget(self.log_x_check)

        self.log_y_check = QtWidgets.QCheckBox("log I")
        self.log_y_check.setChecked(True)
        self.log_y_check.stateChanged.connect(self._plot_selected)
        controls.addWidget(self.log_y_check)

        self.auto_refresh_check = QtWidgets.QCheckBox("auto")
        self.auto_refresh_check.setChecked(True)
        controls.addWidget(self.auto_refresh_check)

        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_now)
        controls.addWidget(refresh)

        splitter = QtWidgets.QSplitter()
        root.addWidget(splitter, 1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.raw_controls = QtWidgets.QWidget()
        raw_form = QtWidgets.QFormLayout(self.raw_controls)
        raw_form.setContentsMargins(0, 0, 0, 0)
        self.energy_combo = QtWidgets.QComboBox()
        self.energy_combo.currentTextChanged.connect(self._raw_filter_changed)
        raw_form.addRow("Energy", self.energy_combo)
        self.group_combo = QtWidgets.QComboBox()
        self.group_combo.currentTextChanged.connect(self._raw_filter_changed)
        raw_form.addRow("Group", self.group_combo)
        self.raw_mode_combo = QtWidgets.QComboBox()
        self.raw_mode_combo.addItems(["latest", "single frame", "last N", "all in group", "average + frames", "heatmap"])
        self.raw_mode_combo.currentTextChanged.connect(self._raw_mode_changed)
        raw_form.addRow("Mode", self.raw_mode_combo)
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setMinimum(1)
        self.frame_slider.setMaximum(1)
        self.frame_slider.valueChanged.connect(self._plot_selected)
        raw_form.addRow("Frame", self.frame_slider)
        self.frame_label = QtWidgets.QLabel("No frame")
        raw_form.addRow("", self.frame_label)
        self.last_n_spin = QtWidgets.QSpinBox()
        self.last_n_spin.setRange(1, 500)
        self.last_n_spin.setValue(10)
        self.last_n_spin.valueChanged.connect(self._plot_selected)
        raw_form.addRow("Last N", self.last_n_spin)
        left_layout.addWidget(self.raw_controls)

        self.curve_list = QtWidgets.QListWidget()
        self.curve_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.curve_list.itemSelectionChanged.connect(self._plot_selected)
        left_layout.addWidget(self.curve_list, 1)
        self.status_label = QtWidgets.QLabel("No curves loaded")
        left_layout.addWidget(self.status_label)
        splitter.addWidget(left)

        plot_panel = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(7.8, 5.6), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)
        splitter.addWidget(plot_panel)
        splitter.setSizes([300, 860])

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(max(250, refresh_ms))
        self.timer.timeout.connect(self._refresh_if_auto)
        self.timer.start()

    def _browse_output_dir(self) -> None:
        current = Path(self.output_dir_edit.text() or str(DEFAULT_OUTPUT_DIR)).expanduser()
        start = str(current.parent if current.is_file() else current)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select analysis HDF5 file",
            start,
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if path:
            self.output_dir_edit.setText(path)
            self._refresh_now()

    def _refresh_if_auto(self) -> None:
        if self.auto_refresh_check.isChecked():
            self._refresh_now()

    def _analysis_h5_path(self) -> Path:
        return find_analysis_h5(Path(self.output_dir_edit.text()).expanduser())

    def _scan_curve_files(self) -> list[CurveFile]:
        output_dir = Path(self.output_dir_edit.text()).expanduser()
        kind = self.curve_kind_combo.currentText()
        if kind.startswith("h5") or kind == "analysis h5":
            return h5_curve_records(find_analysis_h5(output_dir), kind)
        specs = CURVE_PATTERNS[kind]

        files: list[CurveFile] = []
        for pattern, label_kind in specs:
            for path in output_dir.glob(pattern):
                if not path.is_file():
                    continue
                stat = path.stat()
                files.append(
                    CurveFile(
                        key=str(path.resolve()),
                        path=str(path.resolve()),
                        kind=label_kind,
                        label=curve_label(path, label_kind),
                        mtime_ns=stat.st_mtime_ns,
                        size=stat.st_size,
                    )
                )
        return sorted(files, key=lambda item: item.key)

    def _refresh_now(self) -> None:
        if self.curve_kind_combo.currentText() == "h5 single frames":
            self._refresh_raw_frames()
            return

        self.raw_controls.setVisible(False)
        self.curve_list.setVisible(True)
        try:
            files = self._scan_curve_files()
        except OSError as exc:
            self.status_label.setText(f"Cannot scan output directory: {exc}")
            return

        current_keys = {item.key for item in files}
        for old_key in list(self.curve_files):
            if old_key not in current_keys:
                self.curve_files.pop(old_key, None)
                self.curves.pop(old_key, None)

        changed = False
        for item in files:
            previous = self.curve_files.get(item.key)
            if previous and previous.mtime_ns == item.mtime_ns and previous.size == item.size:
                continue
            try:
                if item.key.startswith(("h5://", "h5curve://")):
                    self.curves[item.key] = load_h5_curve(item)
                else:
                    self.curves[item.key] = load_curve(Path(item.path), item.label)
            except ValueError:
                continue
            self.curve_files[item.key] = item
            changed = True

        if changed or self.curve_list.count() != len(self.curve_files):
            self._sync_curve_list()
            self._plot_selected()
        self.status_label.setText(
            f"{len(self.curves)} {self.curve_kind_combo.currentText()} curves loaded; "
            f"{len(self._selected_paths())} selected"
        )

    def _refresh_raw_frames(self) -> None:
        self.raw_controls.setVisible(True)
        self.curve_list.setVisible(False)
        analysis_h5 = self._analysis_h5_path()
        try:
            table = load_h5_frame_table(analysis_h5)
        except OSError as exc:
            self.status_label.setText(f"Cannot read analysis HDF5: {exc}")
            return
        if table is None:
            self.frame_table = None
            self.status_label.setText(f"No live single-frame table in {analysis_h5}")
            self.figure.clear()
            self.canvas.draw_idle()
            return
        changed = (
            self.frame_table is None
            or self.frame_table.analysis_h5 != table.analysis_h5
            or self.frame_table.mtime_ns != table.mtime_ns
            or self.frame_table.size != table.size
        )
        self.frame_table = table
        self._sync_raw_controls()
        if changed:
            self._plot_selected()

    def _sync_curve_list(self) -> None:
        self.curve_list.blockSignals(True)
        selected_paths = {
            self.curve_list.item(row).data(QtCore.Qt.UserRole)
            for row in range(self.curve_list.count())
            if self.curve_list.item(row).isSelected()
        }
        self.curve_list.clear()
        sorted_keys = sorted(self.curves)
        default_selected = {sorted_keys[-1]} if sorted_keys and not selected_paths else set()
        for key in sorted_keys:
            curve = self.curves[key]
            item = QtWidgets.QListWidgetItem(curve.label)
            item.setToolTip(str(curve.path))
            item.setData(QtCore.Qt.UserRole, key)
            self.curve_list.addItem(item)
            if key in selected_paths or key in default_selected:
                item.setSelected(True)
        self.curve_list.blockSignals(False)

    def _sync_raw_controls(self) -> None:
        table = self.frame_table
        if table is None or table.intensity.shape[0] == 0:
            return

        current_energy = self.energy_combo.currentData()
        current_group = self.group_combo.currentData()
        energies = sorted(set(map(int, table.energy_index)))
        newest_row = table.intensity.shape[0] - 1
        follow_latest = self.raw_mode_combo.currentText() == "latest"

        self.energy_combo.blockSignals(True)
        self.energy_combo.clear()
        for energy in energies:
            rows = np.flatnonzero(table.energy_index == energy)
            kev = table.energy_kev[rows[0]] if rows.size else np.nan
            label = f"E{energy:03d}"
            if np.isfinite(kev):
                label += f" {float(kev):.4f} keV"
            self.energy_combo.addItem(label, energy)
        if follow_latest and newest_row >= 0:
            energy_to_select = int(table.energy_index[newest_row])
        else:
            energy_to_select = current_energy if current_energy in energies else energies[-1]
        self.energy_combo.setCurrentIndex(max(0, self.energy_combo.findData(energy_to_select)))
        self.energy_combo.blockSignals(False)

        groups = sorted(set(map(int, table.group_index[table.energy_index == energy_to_select])))
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        for group in groups:
            self.group_combo.addItem(f"G{group:02d}", group)
        if follow_latest and newest_row >= 0 and int(table.energy_index[newest_row]) == int(energy_to_select):
            group_to_select = int(table.group_index[newest_row])
        else:
            group_to_select = current_group if current_group in groups else (groups[-1] if groups else 1)
        self.group_combo.setCurrentIndex(max(0, self.group_combo.findData(group_to_select)))
        self.group_combo.blockSignals(False)
        self._sync_frame_slider()

    def _raw_filter_changed(self) -> None:
        self._sync_frame_slider()
        self._plot_selected()

    def _raw_mode_changed(self) -> None:
        mode = self.raw_mode_combo.currentText()
        self.frame_slider.setEnabled(mode == "single frame")
        self.last_n_spin.setEnabled(mode == "last N")
        self._plot_selected()

    def _current_raw_rows(self) -> np.ndarray:
        table = self.frame_table
        if table is None:
            return np.asarray([], dtype=int)
        energy = self.energy_combo.currentData()
        group = self.group_combo.currentData()
        if energy is None or group is None:
            return np.asarray([], dtype=int)
        return np.flatnonzero((table.energy_index == int(energy)) & (table.group_index == int(group)))

    def _sync_frame_slider(self) -> None:
        rows = self._current_raw_rows()
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(1)
        self.frame_slider.setMaximum(max(1, rows.size))
        if rows.size:
            if self.raw_mode_combo.currentText() == "latest":
                self.frame_slider.setValue(rows.size)
            else:
                self.frame_slider.setValue(min(max(1, self.frame_slider.value()), rows.size))
            row = rows[self.frame_slider.value() - 1]
            table = self.frame_table
            self.frame_label.setText(f"F{int(table.frame_index[row]):03d} of {rows.size}")
        else:
            self.frame_slider.setValue(1)
            self.frame_label.setText("No frame")
        self.frame_slider.blockSignals(False)

    def _selected_paths(self) -> list[str]:
        selected = set()
        for item in self.curve_list.selectedItems():
            selected.add(item.data(QtCore.Qt.UserRole))
        paths: list[str] = [
            self.curve_list.item(row).data(QtCore.Qt.UserRole)
            for row in range(self.curve_list.count())
            if self.curve_list.item(row).data(QtCore.Qt.UserRole) in selected
        ]
        return paths[-self.max_curves_spin.value() :]

    def _plot_selected(self) -> None:
        if self.curve_kind_combo.currentText() == "h5 single frames":
            self._plot_raw_frames()
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        selected = self._selected_paths()
        for path in selected:
            curve = self.curves.get(path)
            if curve is None:
                continue
            q = np.asarray(curve.q, dtype=float)
            intensity = np.asarray(curve.intensity, dtype=float)
            mask = np.isfinite(q) & np.isfinite(intensity)
            if self.log_x_check.isChecked():
                mask &= q > 0
            if self.log_y_check.isChecked():
                mask &= intensity > 0
            if np.count_nonzero(mask) < 2:
                continue
            ax.plot(q[mask], intensity[mask], linewidth=1.2, label=curve.label)

        ax.set_xlabel("q")
        ax.set_ylabel("I")
        ax.set_xscale("log" if self.log_x_check.isChecked() else "linear")
        ax.set_yscale("log" if self.log_y_check.isChecked() else "linear")
        ax.grid(True, which="both", alpha=0.22)
        if selected:
            ax.legend(loc="best", fontsize=8)
        ax.set_title("Live 1D Curves")
        self.canvas.draw_idle()

    def _plot_raw_frames(self) -> None:
        self.figure.clear()
        table = self.frame_table
        rows = self._current_raw_rows()
        if table is None or rows.size == 0:
            ax = self.figure.add_subplot(111)
            ax.set_title("No single-frame curves")
            self.canvas.draw_idle()
            return

        mode = self.raw_mode_combo.currentText()
        if mode == "latest":
            plot_rows = rows[-1:]
        elif mode == "single frame":
            index = min(max(1, self.frame_slider.value()), rows.size) - 1
            plot_rows = rows[index : index + 1]
        elif mode == "last N":
            plot_rows = rows[-min(self.last_n_spin.value(), rows.size) :]
        else:
            plot_rows = rows

        if mode == "heatmap":
            self._plot_raw_heatmap(table, plot_rows, rows)
        else:
            self._plot_raw_curves(table, plot_rows, rows, mode)
        self.status_label.setText(self._raw_status_text(table, plot_rows, rows, mode))
        self.canvas.draw_idle()

    def _plot_raw_curves(self, table: H5FrameTable, plot_rows: np.ndarray, group_rows: np.ndarray, mode: str) -> None:
        ax = self.figure.add_subplot(111)
        q = np.asarray(table.q, dtype=float)
        if mode == "average + frames":
            for row in plot_rows:
                self._plot_one_raw_curve(ax, q, table.intensity[row], table.qc_status[row], label=None, alpha=0.20, linewidth=0.8)
            avg = np.nanmean(table.intensity[group_rows], axis=0)
            self._plot_one_raw_curve(ax, q, avg, "accepted", label="group average", alpha=1.0, linewidth=2.2, color="black")
        else:
            many = plot_rows.size > 5
            for row in plot_rows:
                label = None if many else self._frame_label(table, row)
                alpha = 0.28 if many else 0.9
                linewidth = 0.8 if many else 1.4
                self._plot_one_raw_curve(ax, q, table.intensity[row], table.qc_status[row], label=label, alpha=alpha, linewidth=linewidth)

        ax.set_xlabel("q")
        ax.set_ylabel("I")
        ax.set_xscale("log" if self.log_x_check.isChecked() else "linear")
        ax.set_yscale("log" if self.log_y_check.isChecked() else "linear")
        ax.grid(True, which="both", alpha=0.22)
        ax.set_title(self._raw_title(table, group_rows, mode, plot_rows.size))
        if plot_rows.size <= 5 or mode == "average + frames":
            ax.legend(loc="best", fontsize=8)
        elif plot_rows.size:
            self._add_status_legend(ax, table, plot_rows)

    def _plot_one_raw_curve(
        self,
        ax,
        q: np.ndarray,
        intensity: np.ndarray,
        status: str,
        label: str | None,
        alpha: float,
        linewidth: float,
        color: str | None = None,
    ) -> None:
        mask = np.isfinite(q) & np.isfinite(intensity)
        if self.log_x_check.isChecked():
            mask &= q > 0
        if self.log_y_check.isChecked():
            mask &= intensity > 0
        if np.count_nonzero(mask) < 2:
            return
        ax.plot(q[mask], intensity[mask], color=color or self._status_color(status), alpha=alpha, linewidth=linewidth, label=label)

    def _plot_raw_heatmap(self, table: H5FrameTable, plot_rows: np.ndarray, group_rows: np.ndarray) -> None:
        ax = self.figure.add_subplot(111)
        q = np.asarray(table.q, dtype=float)
        data = np.asarray(table.intensity[plot_rows], dtype=float)
        x = np.log10(q) if self.log_x_check.isChecked() else q
        z = np.log10(np.where(data > 0, data, np.nan)) if self.log_y_check.isChecked() else data
        image = ax.imshow(
            z,
            aspect="auto",
            origin="lower",
            extent=[float(np.nanmin(x)), float(np.nanmax(x)), 1, max(1, plot_rows.size)],
            interpolation="nearest",
        )
        ax.set_xlabel("log10(q)" if self.log_x_check.isChecked() else "q")
        ax.set_ylabel("frame order")
        ax.set_title(self._raw_title(table, group_rows, "heatmap", plot_rows.size))
        self.figure.colorbar(image, ax=ax, label="log10(I)" if self.log_y_check.isChecked() else "I")

    def _status_color(self, status: str) -> str:
        return {
            "accepted": "#2f7ed8",
            "rejected_total_intensity": "#d84a3a",
            "pending_group_qc": "#777777",
        }.get(status, "#777777")

    def _add_status_legend(self, ax, table: H5FrameTable, rows: np.ndarray) -> None:
        from matplotlib.lines import Line2D

        statuses = []
        for status in ["accepted", "rejected_total_intensity", "pending_group_qc"]:
            if any(table.qc_status[row] == status for row in rows):
                statuses.append(status)
        handles = [
            Line2D([0], [0], color=self._status_color(status), lw=2, label=status)
            for status in statuses
        ]
        if handles:
            ax.legend(handles=handles, loc="best", fontsize=8)

    def _frame_label(self, table: H5FrameTable, row: int) -> str:
        return f"E{int(table.energy_index[row]):03d} G{int(table.group_index[row]):02d} F{int(table.frame_index[row]):03d}"

    def _raw_title(self, table: H5FrameTable, group_rows: np.ndarray, mode: str, plotted_count: int) -> str:
        if group_rows.size == 0:
            return "Live Single Frames"
        row = group_rows[-1]
        energy = f"E{int(table.energy_index[row]):03d}"
        if np.isfinite(table.energy_kev[row]):
            energy += f" {float(table.energy_kev[row]):.4f} keV"
        return f"{energy} G{int(table.group_index[row]):02d} | {mode} | {plotted_count} plotted"

    def _raw_status_text(self, table: H5FrameTable, plot_rows: np.ndarray, group_rows: np.ndarray, mode: str) -> str:
        counts = {status: 0 for status in ["accepted", "rejected_total_intensity", "pending_group_qc"]}
        for row in group_rows:
            counts[table.qc_status[row]] = counts.get(table.qc_status[row], 0) + 1
        return (
            f"{group_rows.size} frames in group; {plot_rows.size} plotted by {mode}; "
            f"accepted {counts.get('accepted', 0)} | "
            f"rejected {counts.get('rejected_total_intensity', 0)} | "
            f"pending {counts.get('pending_group_qc', 0)}"
        )


def main() -> int:
    args = build_parser().parse_args()
    app = QtWidgets.QApplication(sys.argv)
    window = LiveCurveViewer(Path(args.output_dir), args.refresh_ms)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
