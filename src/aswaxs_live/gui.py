"""GUI launcher for the ASWAXS v2 live pipeline demo.

Window 0 is the setup/control window. It builds the command for the live reducer
launcher and can launch/stop the reducer.

Window 1 is the acquisition/reduction monitor. It shows reducer stdout/stderr
and tails ``live_events.jsonl`` so the user can see frame, group, and energy
triggers as they happen.

Window 2 is the curve browser/plotter. It reuses ``LiveCurveViewer`` and lets
the user select available 1D output files to plot.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from aswaxs_live.stitcher import update_live_stitched_averages
from aswaxs_live.viewer import LiveCurveViewer


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
PLAYGROUND_DIR = PROJECT_DIR.parent
LEGACY_PIPELINE_ROOT = PLAYGROUND_DIR / "ASWAXS_reduction_pipeline"
DEFAULT_PIPELINE_ROOT = ""
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "live_gui_run"
DEFAULT_MANIFEST = (
    DEFAULT_PIPELINE_ROOT
    / "outputs"
    / "FC_AuSiO2NP_60uL_min_Eig1M_reduced_no_fluorescence"
    / "sequence_manifest.csv"
)
DEFAULT_EIGER_WATCH_DIR = (
    r"\\chemmat-c51\data_rw\aswaxs\bera\Apr2026\Commissioning"
    r"\FC_AuSiO2NP_ASWAXS_60uL_min\Eig1M"
)
DEFAULT_PIL300K_WATCH_DIR = DEFAULT_EIGER_WATCH_DIR.replace("Eig1M", "Pil300K")
DEFAULT_WATCH_DIR = DEFAULT_PIL300K_WATCH_DIR
DEFAULT_SAMPLE_NAME = "FC_AuSiO2NP_ASWAXS_60uL_min"
DEFAULT_SAXS_PONI = PLAYGROUND_DIR / "FC_ASWAXS" / "FC_AgBH_11_919keV" / "Pil300K" / "calib.poni"
DEFAULT_SAXS_MASK = PLAYGROUND_DIR / "FC_ASWAXS" / "FC_AgBH_11_919keV" / "Pil300K" / "mask.msk"
DEFAULT_WAXS_PONI = PLAYGROUND_DIR / "FC_ASWAXS" / "FC_AgBH_11_919keV" / "Eig1M" / "calib.poni"
DEFAULT_WAXS_MASK = PLAYGROUND_DIR / "FC_ASWAXS" / "FC_AgBH_11_919keV" / "Eig1M" / "mask.msk"
DEFAULT_PONI = DEFAULT_SAXS_PONI
DEFAULT_MASK = DEFAULT_SAXS_MASK
SETTINGS_PATH = PROJECT_DIR / ".aswaxs_live_gui_settings.json"


class NoWheelSpinBox(QtWidgets.QSpinBox):
    """QSpinBox that ignores mouse-wheel changes."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt method name.
        event.ignore()


class NoWheelDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """QDoubleSpinBox that ignores mouse-wheel changes."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt method name.
        event.ignore()


class ProcessMonitorWindow(QtWidgets.QMainWindow):
    """Show reducer text output and structured live event records."""

    def __init__(self, output_dir: Path, title: str = "ASWAXS Live Process Monitor") -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1120, 760)
        self.process: QtCore.QProcess | None = None
        self.output_dir = output_dir
        self.event_log_path = self.output_dir / "live_events.jsonl"
        self._event_offset = 0
        self.expected_frames: int | None = None
        self.frames_reduced = 0
        self.groups_done = 0
        self.energies_done = 0
        self.run_started_at: float | None = None
        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tail_event_log)
        self.timer.start()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        self.status_label = QtWidgets.QLabel("Reducer is not running")
        root.addWidget(self.status_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        root.addWidget(splitter, 1)

        self.event_table = QtWidgets.QTableWidget(0, 7)
        self.event_table.setHorizontalHeaderLabels(
            ["time", "event", "energy", "group", "frame", "sequence", "message/path"]
        )
        header = self.event_table.horizontalHeader()
        header.setStretchLastSection(True)
        for column, width in enumerate([220, 230, 70, 70, 70, 90, 420]):
            self.event_table.setColumnWidth(column, width)
        for column in range(6):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.Interactive)
        header.setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)
        splitter.addWidget(self.event_table)

        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        splitter.addWidget(self.log_edit)
        splitter.setSizes([430, 300])

        progress_panel = QtWidgets.QWidget()
        progress_layout = QtWidgets.QVBoxLayout(progress_panel)
        progress_layout.setContentsMargins(0, 6, 0, 0)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Waiting for frames")
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #9aa4b2;
                border-radius: 6px;
                background: #eef1f5;
                min-height: 18px;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 6px;
                background-color: #2f7ed8;
            }
            """
        )
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QtWidgets.QLabel("Frames 0 | Groups 0 | Energies 0")
        progress_layout.addWidget(self.progress_label)
        root.addWidget(progress_panel)

    def set_output_dir(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.event_log_path = self.output_dir / "live_events.jsonl"
        self.clear_run_display()

    def set_expected_frames(self, expected_frames: int | None) -> None:
        self.expected_frames = expected_frames if expected_frames and expected_frames > 0 else None
        self._update_progress()

    def clear_run_display(self) -> None:
        self._event_offset = 0
        self.frames_reduced = 0
        self.groups_done = 0
        self.energies_done = 0
        self.run_started_at = None
        self.event_table.setRowCount(0)
        self.log_edit.clear()
        self.status_label.setText("Reducer is not running")
        self._update_progress()

    def attach_process(self, process: QtCore.QProcess) -> None:
        self.process = process
        process.readyReadStandardOutput.connect(self.read_stdout)
        process.readyReadStandardError.connect(self.read_stderr)
        process.started.connect(self._process_started)
        process.finished.connect(self._process_finished)

    def _process_started(self) -> None:
        self.status_label.setText("Reducer running")
        self.run_started_at = time.monotonic()
        self._update_progress()

    def append_log(self, text: str) -> None:
        self.log_edit.appendPlainText(text.rstrip())

    def read_stdout(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        if text:
            self.append_log(text)

    def read_stderr(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode(errors="replace")
        if text:
            self.append_log(text)

    def _process_finished(self, code: int, status: QtCore.QProcess.ExitStatus) -> None:
        state = "crashed" if status == QtCore.QProcess.CrashExit else "finished"
        self.status_label.setText(f"Reducer {state} with exit code {code}")
        self.tail_event_log()

    def tail_event_log(self) -> None:
        if not self.event_log_path.exists():
            return
        try:
            with self.event_log_path.open("r", encoding="utf-8") as handle:
                handle.seek(self._event_offset)
                lines = handle.readlines()
                self._event_offset = handle.tell()
        except OSError:
            return
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._add_event_row(event)

    def _add_event_row(self, event: dict[str, object]) -> None:
        row = self.event_table.rowCount()
        self.event_table.insertRow(row)
        message = event.get("message") or event.get("path") or ""
        values = [
            event.get("time", ""),
            event.get("event", ""),
            event.get("energy_index", ""),
            event.get("group_index", ""),
            event.get("frame_index", ""),
            event.get("sequence_index", ""),
            message,
        ]
        for column, value in enumerate(values):
            self.event_table.setItem(row, column, QtWidgets.QTableWidgetItem("" if value is None else str(value)))
        self.event_table.scrollToBottom()
        self._update_progress_from_event(event)

    def _update_progress_from_event(self, event: dict[str, object]) -> None:
        event_name = str(event.get("event", ""))
        if event_name == "frame_reduced_1d":
            if self.run_started_at is None:
                self.run_started_at = time.monotonic()
            self.frames_reduced += 1
        elif event_name == "group_average_written":
            self.groups_done += 1
        elif event_name in {"energy_batch_asaxs_completed", "energy_batch_saxs_completed"}:
            self.energies_done += 1
        self._update_progress()

    def _update_progress(self) -> None:
        if self.expected_frames:
            self.progress_bar.setMaximum(self.expected_frames)
            self.progress_bar.setValue(min(self.frames_reduced, self.expected_frames))
            percent = 100.0 * min(self.frames_reduced, self.expected_frames) / self.expected_frames
            self.progress_bar.setFormat(
                f"{self.frames_reduced}/{self.expected_frames} frames ({percent:.1f}%) | "
                f"{self._eta_text()}"
            )
        else:
            self.progress_bar.setMaximum(0)
            self.progress_bar.setFormat(f"{self.frames_reduced} frames reduced | {self._elapsed_text()}")
        self.progress_label.setText(
            f"Frames {self.frames_reduced}"
            + (f" / {self.expected_frames}" if self.expected_frames else "")
            + f" | Groups {self.groups_done} | Energies {self.energies_done}"
            + f" | {self._elapsed_text()}"
            + (f" | {self._eta_text()}" if self.expected_frames else "")
        )

    def _elapsed_text(self) -> str:
        if self.run_started_at is None:
            return "elapsed --"
        return f"elapsed {self._format_duration(time.monotonic() - self.run_started_at)}"

    def _eta_text(self) -> str:
        if self.run_started_at is None or not self.expected_frames or self.frames_reduced <= 0:
            return "remaining estimating"
        elapsed = time.monotonic() - self.run_started_at
        rate = self.frames_reduced / elapsed if elapsed > 0 else 0
        if rate <= 0:
            return "remaining estimating"
        remaining_frames = max(0, self.expected_frames - self.frames_reduced)
        return f"remaining {self._format_duration(remaining_frames / rate)}"

    def _format_duration(self, seconds: float) -> str:
        total = max(0, int(round(seconds)))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m {secs:02d}s"
        if minutes:
            return f"{minutes:d}m {secs:02d}s"
        return f"{secs:d}s"


class SetupWindow(QtWidgets.QMainWindow):
    """Parameter window for launching and monitoring the v2 live reducer."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ASWAXS V2 Live Pipeline Setup")
        self.resize(1180, 860)
        self.process: QtCore.QProcess | None = None
        self.detector_processes: dict[str, QtCore.QProcess] = {}
        self.monitor_window = ProcessMonitorWindow(DEFAULT_OUTPUT_DIR)
        self.dual_monitor_windows = {
            "pil300k": ProcessMonitorWindow(DEFAULT_OUTPUT_DIR / "Pil300K", "Pil300K Process Monitor"),
            "eig1m": ProcessMonitorWindow(DEFAULT_OUTPUT_DIR / "Eig1M", "Eig1M Process Monitor"),
        }
        self.curve_window = LiveCurveViewer(DEFAULT_OUTPUT_DIR, refresh_ms=1500)
        self.dual_curve_windows = {
            "pil300k": LiveCurveViewer(DEFAULT_OUTPUT_DIR / "Pil300K", refresh_ms=1500),
            "eig1m": LiveCurveViewer(DEFAULT_OUTPUT_DIR / "Eig1M", refresh_ms=1500),
            "stitched": LiveCurveViewer(DEFAULT_OUTPUT_DIR / f"{DEFAULT_SAMPLE_NAME}_analysis.h5", refresh_ms=1500),
        }
        self.stitch_timer = QtCore.QTimer(self)
        self.stitch_timer.setInterval(2000)
        self.stitch_timer.timeout.connect(self._update_live_stitched_outputs)
        self._build_ui()
        self._load_settings()
        self._update_source_visibility()
        self.refresh_command()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = QtWidgets.QWidget()
        self.form_layout = QtWidgets.QVBoxLayout(form_widget)
        self.form_layout.setSpacing(10)
        scroll.setWidget(form_widget)
        root.addWidget(scroll, 1)

        self._build_source_group()
        self._build_paths_group()
        self._build_reduction_group()
        self._build_sequence_group()
        self._build_asaxs_group()

        self.command_preview = QtWidgets.QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMaximumHeight(120)
        root.addWidget(self.command_preview)

        actions = QtWidgets.QHBoxLayout()
        root.addLayout(actions)
        self.start_button = QtWidgets.QPushButton("Start Reducer")
        self.start_button.clicked.connect(self.start_reducer)
        actions.addWidget(self.start_button)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_reducer)
        actions.addWidget(self.stop_button)
        monitor_button = QtWidgets.QPushButton("Show Process Monitor")
        monitor_button.clicked.connect(self.show_monitor)
        actions.addWidget(monitor_button)
        curves_button = QtWidgets.QPushButton("Show Curves")
        curves_button.clicked.connect(self.show_curves)
        actions.addWidget(curves_button)
        actions.addStretch(1)

        self.statusBar().showMessage("Ready")

    def _settings_widgets(self) -> dict[str, QtWidgets.QWidget]:
        return {
            "source_mode": self.source_mode_combo,
            "detector_run_mode": self.detector_run_mode_combo,
            "analysis_mode": self.analysis_mode_combo,
            "sample_name": self.sample_name_edit,
            "write_text_output": self.write_text_output_check,
            "restart_behavior": self.restart_behavior_combo,
            "once": self.once_check,
            "watch_dir": self.watch_dir_edit,
            "manifest": self.manifest_edit,
            "output_dir": self.output_dir_edit,
            "analysis_h5": self.analysis_h5_edit,
            "pipeline_root": self.pipeline_root_edit,
            "poni": self.poni_edit,
            "mask": self.mask_edit,
            "pil300k_watch_dir": self.saxs_watch_dir_edit,
            "eig1m_watch_dir": self.waxs_watch_dir_edit,
            "pil300k_poni": self.saxs_poni_edit,
            "pil300k_mask": self.saxs_mask_edit,
            "eig1m_poni": self.waxs_poni_edit,
            "eig1m_mask": self.waxs_mask_edit,
            "pil300k_monitor_key": self.saxs_monitor_key_edit,
            "eig1m_monitor_key": self.waxs_monitor_key_edit,
            "dataset_path": self.dataset_path_edit,
            "detector": self.detector_combo,
            "monitor_key": self.monitor_key_edit,
            "npt": self.npt_spin,
            "unit": self.unit_edit,
            "outlier_zmax": self.outlier_spin,
            "delta_energy_percent": self.delta_energy_spin,
            "pattern": self.pattern_edit,
            "num_energies": self.num_energies_spin,
            "num_groups": self.num_groups_spin,
            "num_frames": self.num_frames_spin,
            "limit_energies": self.limit_energies_spin,
            "limit_frames": self.limit_frames_spin,
            "poll_seconds": self.poll_spin,
            "settle_seconds": self.settle_spin,
            "gc_group": self.gc_group_spin,
            "air_group": self.air_group_spin,
            "empty_group": self.empty_group_spin,
            "water_group": self.water_group_spin,
            "sample_group": self.sample_group_spin,
            "gc_reference_file": self.gc_ref_edit,
            "gc_q_min": self.gc_q_min_spin,
            "gc_q_max": self.gc_q_max_spin,
            "capillary_thickness": self.capillary_thickness_spin,
            "gc_thickness": self.gc_thickness_spin,
            "subtract_fluorescence": self.subtract_fluorescence_check,
            "fluorescence_reference": self.fluorescence_reference_combo,
            "fluorescence_level": self.fluorescence_level_spin,
            "fluorescence_q_min": self.fluorescence_q_min_spin,
            "fluorescence_q_max": self.fluorescence_q_max_spin,
        }

    def _load_settings(self) -> None:
        if not SETTINGS_PATH.exists():
            return
        try:
            settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for key, widget in self._settings_widgets().items():
            if key in settings:
                self._set_widget_value(widget, settings[key])
        legacy_detector_keys = {
            "saxs_watch_dir": self.saxs_watch_dir_edit,
            "waxs_watch_dir": self.waxs_watch_dir_edit,
            "saxs_poni": self.saxs_poni_edit,
            "saxs_mask": self.saxs_mask_edit,
            "waxs_poni": self.waxs_poni_edit,
            "waxs_mask": self.waxs_mask_edit,
            "saxs_monitor_key": self.saxs_monitor_key_edit,
            "waxs_monitor_key": self.waxs_monitor_key_edit,
        }
        for key, widget in legacy_detector_keys.items():
            if key in settings:
                self._set_widget_value(widget, settings[key])
        if settings.get("detector_run_mode") == "SAXS + WAXS":
            self.detector_run_mode_combo.setCurrentText("Pil300K + Eig1M")
        if Path(self.pipeline_root_edit.text()).expanduser() == LEGACY_PIPELINE_ROOT:
            self.pipeline_root_edit.clear()
        self._migrate_dual_detector_settings()
        self.statusBar().showMessage(f"Loaded previous settings from {SETTINGS_PATH.name}")

    def _save_settings(self) -> None:
        settings = {
            key: self._widget_value(widget)
            for key, widget in self._settings_widgets().items()
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            self.statusBar().showMessage(f"Could not save GUI settings: {exc}")

    def _widget_value(self, widget: QtWidgets.QWidget) -> object:
        if isinstance(widget, QtWidgets.QLineEdit):
            return widget.text()
        if isinstance(widget, QtWidgets.QComboBox):
            return widget.currentText()
        if isinstance(widget, QtWidgets.QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QtWidgets.QSpinBox | QtWidgets.QDoubleSpinBox):
            return widget.value()
        return None

    def _set_widget_value(self, widget: QtWidgets.QWidget, value: object) -> None:
        widget.blockSignals(True)
        try:
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.setText("" if value is None else str(value))
            elif isinstance(widget, QtWidgets.QComboBox):
                text = "" if value is None else str(value)
                if widget.findText(text) >= 0:
                    widget.setCurrentText(text)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QtWidgets.QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.setValue(float(value))
        except (TypeError, ValueError):
            pass
        finally:
            widget.blockSignals(False)

    def _migrate_dual_detector_settings(self) -> None:
        """Swap older saved GUI defaults to SAXS=Pil300K and WAXS=Eig1M."""
        saxs_text = " ".join([self.saxs_watch_dir_edit.text(), self.saxs_poni_edit.text(), self.saxs_mask_edit.text()])
        waxs_text = " ".join([self.waxs_watch_dir_edit.text(), self.waxs_poni_edit.text(), self.waxs_mask_edit.text()])
        if "Eig1M" not in saxs_text or "Pil300K" not in waxs_text:
            return
        saxs_watch, waxs_watch = self.saxs_watch_dir_edit.text(), self.waxs_watch_dir_edit.text()
        saxs_poni, waxs_poni = self.saxs_poni_edit.text(), self.waxs_poni_edit.text()
        saxs_mask, waxs_mask = self.saxs_mask_edit.text(), self.waxs_mask_edit.text()
        self.saxs_watch_dir_edit.setText(waxs_watch)
        self.waxs_watch_dir_edit.setText(saxs_watch)
        self.saxs_poni_edit.setText(waxs_poni)
        self.waxs_poni_edit.setText(saxs_poni)
        self.saxs_mask_edit.setText(waxs_mask)
        self.waxs_mask_edit.setText(saxs_mask)

    def _line(self, value: str = "") -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(value)
        edit.textChanged.connect(self.refresh_command)
        return edit

    def _spin(self, value: int, minimum: int = 0, maximum: int = 1_000_000) -> QtWidgets.QSpinBox:
        spin = NoWheelSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.valueChanged.connect(self.refresh_command)
        return spin

    def _double(self, value: float, minimum: float = -1e9, maximum: float = 1e9) -> QtWidgets.QDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(6)
        spin.setValue(value)
        spin.valueChanged.connect(self.refresh_command)
        return spin

    def _combo(self, values: list[str], current: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItems(values)
        combo.setCurrentText(current)
        combo.currentTextChanged.connect(self.refresh_command)
        return combo

    def _browse_button(self, target: QtWidgets.QLineEdit, mode: str, caption: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton("Browse")

        def browse() -> None:
            start = target.text() or str(PROJECT_DIR)
            if mode == "dir":
                path = QtWidgets.QFileDialog.getExistingDirectory(self, caption, start)
            elif mode == "save_file":
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    caption,
                    start,
                    "HDF5 files (*.h5 *.hdf5);;All files (*)",
                )
            elif mode == "hdf5":
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self,
                    caption,
                    start,
                    "HDF5 files (*.h5 *.hdf5);;All files (*)",
                )
            else:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption, start, mode)
            if path:
                target.setText(path)

        button.clicked.connect(browse)
        return button

    def _path_row(self, form: QtWidgets.QFormLayout, label: str, edit: QtWidgets.QLineEdit, mode: str) -> None:
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        layout.addWidget(self._browse_button(edit, mode, f"Select {label}"))
        form.addRow(label, row)

    def _build_source_group(self) -> None:
        box = QtWidgets.QGroupBox("Window 0: Run Mode")
        form = QtWidgets.QFormLayout(box)
        self.source_mode_combo = self._combo(["watch folder", "manifest replay"], "watch folder")
        self.source_mode_combo.currentTextChanged.connect(self._update_source_visibility)
        form.addRow("Source", self.source_mode_combo)
        self.detector_run_mode_combo = self._combo(["single detector", "Pil300K + Eig1M"], "single detector")
        self.detector_run_mode_combo.currentTextChanged.connect(self._update_source_visibility)
        form.addRow("Detector jobs", self.detector_run_mode_combo)
        self.analysis_mode_combo = self._combo(["asaxs", "saxs"], "asaxs")
        form.addRow("Analysis mode", self.analysis_mode_combo)
        self.sample_name_edit = self._line(DEFAULT_SAMPLE_NAME)
        form.addRow("Sample name", self.sample_name_edit)
        self.write_text_output_check = QtWidgets.QCheckBox("write legacy .dat curve files")
        self.write_text_output_check.setChecked(False)
        self.write_text_output_check.stateChanged.connect(self.refresh_command)
        form.addRow("Text output", self.write_text_output_check)
        self.restart_behavior_combo = self._combo(["resume", "restart"], "resume")
        form.addRow("Existing output", self.restart_behavior_combo)
        self.once_check = QtWidgets.QCheckBox("process current files once")
        self.once_check.stateChanged.connect(self._update_source_visibility)
        form.addRow("Watcher once", self.once_check)
        self.form_layout.addWidget(box)

    def _build_paths_group(self) -> None:
        box = QtWidgets.QGroupBox("Paths")
        form = QtWidgets.QFormLayout(box)
        self.watch_dir_edit = self._line(DEFAULT_WATCH_DIR)
        self.watch_dir_edit.editingFinished.connect(self._maybe_update_sample_name_from_watch_dir)
        self._path_row(form, "Watch directory", self.watch_dir_edit, "dir")
        self.manifest_edit = self._line(str(DEFAULT_MANIFEST))
        self._path_row(form, "Manifest", self.manifest_edit, "CSV files (*.csv);;All files (*)")
        self.output_dir_edit = self._line(str(DEFAULT_OUTPUT_DIR))
        self._path_row(form, "Output directory", self.output_dir_edit, "dir")
        self.analysis_h5_edit = self._line("")
        self._path_row(form, "Analysis HDF5", self.analysis_h5_edit, "save_file")
        self.pipeline_root_edit = self._line(str(DEFAULT_PIPELINE_ROOT))
        self._path_row(form, "External pipeline root (optional)", self.pipeline_root_edit, "dir")
        self.poni_edit = self._line(str(DEFAULT_PONI))
        self._path_row(form, "PONI", self.poni_edit, "PONI files (*.poni);;All files (*)")
        self.mask_edit = self._line(str(DEFAULT_MASK))
        self._path_row(form, "Mask", self.mask_edit, "Mask files (*.msk *.npy *.edf);;HDF5 files (*.h5 *.hdf5);;All files (*)")
        self.form_layout.addWidget(box)

        dual_box = QtWidgets.QGroupBox("Parallel Detector Inputs")
        dual_form = QtWidgets.QFormLayout(dual_box)
        self.dual_detector_box = dual_box
        self.saxs_watch_dir_edit = self._line(DEFAULT_PIL300K_WATCH_DIR)
        self.saxs_watch_dir_edit.editingFinished.connect(self._maybe_update_sample_name_from_watch_dir)
        self._path_row(dual_form, "Pil300K watch directory", self.saxs_watch_dir_edit, "dir")
        self.waxs_watch_dir_edit = self._line(DEFAULT_EIGER_WATCH_DIR)
        self._path_row(dual_form, "Eig1M watch directory", self.waxs_watch_dir_edit, "dir")
        self.saxs_poni_edit = self._line(str(DEFAULT_SAXS_PONI))
        self._path_row(dual_form, "Pil300K PONI", self.saxs_poni_edit, "PONI files (*.poni);;All files (*)")
        self.saxs_mask_edit = self._line(str(DEFAULT_SAXS_MASK))
        self._path_row(dual_form, "Pil300K mask", self.saxs_mask_edit, "Mask files (*.msk *.npy *.edf);;HDF5 files (*.h5 *.hdf5);;All files (*)")
        self.saxs_monitor_key_edit = self._line("")
        dual_form.addRow("Pil300K monitor key", self.saxs_monitor_key_edit)
        self.waxs_poni_edit = self._line(str(DEFAULT_WAXS_PONI))
        self._path_row(dual_form, "Eig1M PONI", self.waxs_poni_edit, "PONI files (*.poni);;All files (*)")
        self.waxs_mask_edit = self._line(str(DEFAULT_WAXS_MASK))
        self._path_row(dual_form, "Eig1M mask", self.waxs_mask_edit, "Mask files (*.msk *.npy *.edf);;HDF5 files (*.h5 *.hdf5);;All files (*)")
        self.waxs_monitor_key_edit = self._line("")
        dual_form.addRow("Eig1M monitor key", self.waxs_monitor_key_edit)
        self.form_layout.addWidget(dual_box)

    def _build_reduction_group(self) -> None:
        box = QtWidgets.QGroupBox("Reduction Parameters")
        form = QtWidgets.QFormLayout(box)
        self.reduction_form = form
        self.dataset_path_edit = self._line("entry/data/data")
        form.addRow("Detector dataset path", self.dataset_path_edit)
        self.detector_combo = self._combo(["auto", "Eig1M", "Pil300K"], "auto")
        form.addRow("Detector override", self.detector_combo)
        self.monitor_key_edit = self._line("")
        form.addRow("Monitor key", self.monitor_key_edit)
        self.npt_spin = self._spin(1000, 1)
        form.addRow("q bins", self.npt_spin)
        self.unit_edit = self._line("q_A^-1")
        form.addRow("q unit", self.unit_edit)
        self.outlier_spin = self._double(3.5, 0.0, 100.0)
        form.addRow("Outlier z max", self.outlier_spin)
        self.delta_energy_spin = self._double(0.001, 0.0, 100.0)
        form.addRow("Delta energy %", self.delta_energy_spin)
        self.form_layout.addWidget(box)

    def _build_sequence_group(self) -> None:
        box = QtWidgets.QGroupBox("Acquisition Sequence")
        form = QtWidgets.QFormLayout(box)
        self.pattern_edit = self._line("*.h5")
        form.addRow("File pattern", self.pattern_edit)
        self.num_energies_spin = self._spin(1, 0)
        form.addRow("Number of energies", self.num_energies_spin)
        self.num_groups_spin = self._spin(5, 1)
        form.addRow("Groups per energy", self.num_groups_spin)
        self.num_frames_spin = self._spin(100, 1)
        form.addRow("Frames per group", self.num_frames_spin)
        self.limit_energies_spin = self._spin(0, 0)
        form.addRow("Replay limit energies", self.limit_energies_spin)
        self.limit_frames_spin = self._spin(0, 0)
        form.addRow("Replay limit frames/group", self.limit_frames_spin)
        self.poll_spin = self._double(2.0, 0.1, 3600.0)
        form.addRow("Poll seconds", self.poll_spin)
        self.settle_spin = self._double(2.0, 0.0, 3600.0)
        form.addRow("Settle seconds", self.settle_spin)
        self.form_layout.addWidget(box)

    def _build_asaxs_group(self) -> None:
        box = QtWidgets.QGroupBox("ASAXS Roles and Corrections")
        form = QtWidgets.QFormLayout(box)
        self.gc_group_spin = self._spin(1, 0)
        form.addRow("GC group", self.gc_group_spin)
        self.air_group_spin = self._spin(2, 0)
        form.addRow("Air group", self.air_group_spin)
        self.empty_group_spin = self._spin(3, 0)
        form.addRow("Empty group", self.empty_group_spin)
        self.water_group_spin = self._spin(4, 0)
        form.addRow("Water/solvent group", self.water_group_spin)
        self.sample_group_spin = self._spin(5, 0)
        form.addRow("Sample group", self.sample_group_spin)
        self.gc_ref_edit = self._line("")
        self._path_row(form, "GC reference file", self.gc_ref_edit, "Data files (*.dat *.txt *.csv);;All files (*)")
        self.gc_q_min_spin = self._double(0.03, 0.0, 1000.0)
        self.gc_q_max_spin = self._double(0.20, 0.0, 1000.0)
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.gc_q_min_spin)
        layout.addWidget(self.gc_q_max_spin)
        form.addRow("GC q range", row)
        self.capillary_thickness_spin = self._double(0.0, 0.0, 1e9)
        form.addRow("Capillary thickness", self.capillary_thickness_spin)
        self.gc_thickness_spin = self._double(0.0, 0.0, 1e9)
        form.addRow("GC thickness", self.gc_thickness_spin)
        self.subtract_fluorescence_check = QtWidgets.QCheckBox("subtract fluorescence")
        self.subtract_fluorescence_check.stateChanged.connect(self.refresh_command)
        form.addRow("Fluorescence", self.subtract_fluorescence_check)
        self.fluorescence_reference_combo = self._combo(["latest", "each"], "latest")
        form.addRow("Fluorescence reference", self.fluorescence_reference_combo)
        self.fluorescence_level_spin = self._double(0.0, -1e12, 1e12)
        form.addRow("Fluorescence fixed level", self.fluorescence_level_spin)
        self.fluorescence_q_min_spin = self._double(0.8, 0.0, 1000.0)
        self.fluorescence_q_max_spin = self._double(1.0, 0.0, 1000.0)
        fluorescence_q_row = QtWidgets.QWidget()
        fluorescence_q_layout = QtWidgets.QHBoxLayout(fluorescence_q_row)
        fluorescence_q_layout.setContentsMargins(0, 0, 0, 0)
        fluorescence_q_layout.addWidget(self.fluorescence_q_min_spin)
        fluorescence_q_layout.addWidget(self.fluorescence_q_max_spin)
        form.addRow("Fluorescence q range", fluorescence_q_row)
        self.form_layout.addWidget(box)

    def _update_source_visibility(self) -> None:
        if self._dual_detector_enabled() and self.source_mode_combo.currentText() != "watch folder":
            self.source_mode_combo.blockSignals(True)
            self.source_mode_combo.setCurrentText("watch folder")
            self.source_mode_combo.blockSignals(False)
        manifest_mode = self.source_mode_combo.currentText() == "manifest replay"
        once_mode = self.once_check.isChecked()
        dual_mode = self._dual_detector_enabled()
        self.manifest_edit.setEnabled(manifest_mode)
        self.watch_dir_edit.setEnabled(not manifest_mode and not dual_mode)
        self.detector_combo.setEnabled(not dual_mode)
        self.detector_combo.setVisible(not dual_mode)
        detector_label = self.reduction_form.labelForField(self.detector_combo)
        if detector_label is not None:
            detector_label.setVisible(not dual_mode)
        self.monitor_key_edit.setEnabled(not dual_mode)
        self.poni_edit.setEnabled(not dual_mode)
        self.mask_edit.setEnabled(not dual_mode)
        self.dual_detector_box.setVisible(dual_mode)
        self.once_check.setEnabled(not manifest_mode)
        self.poll_spin.setEnabled(not manifest_mode and not once_mode)
        self.settle_spin.setEnabled(not manifest_mode and not once_mode)
        self.refresh_command()

    def _dual_detector_enabled(self) -> bool:
        return self.detector_run_mode_combo.currentText() in {"Pil300K + Eig1M", "SAXS + WAXS"}

    def _maybe_update_sample_name_from_watch_dir(self) -> None:
        current = self.sample_name_edit.text().strip()
        if current and current != DEFAULT_SAMPLE_NAME:
            return
        watch_dir = Path(self.watch_dir_edit.text())
        detector_names = {"eig1m", "pil300k", "saxs", "waxs"}
        if watch_dir.name.lower() in detector_names and watch_dir.parent.name:
            self.sample_name_edit.setText(watch_dir.parent.name)
        elif watch_dir.name:
            self.sample_name_edit.setText(watch_dir.name)

    def _append_arg(self, args: list[str], name: str, value: str | int | float | None) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text:
            args.extend([name, text])

    def command_args(self) -> list[str]:
        args = [str(PROJECT_DIR / "scripts" / "run_reducer.py")]
        if self.source_mode_combo.currentText() == "manifest replay":
            self._append_arg(args, "--manifest", self.manifest_edit.text())
            if self.limit_energies_spin.value() > 0:
                self._append_arg(args, "--limit-energies", self.limit_energies_spin.value())
            if self.limit_frames_spin.value() > 0:
                self._append_arg(args, "--limit-frames-per-group", self.limit_frames_spin.value())
        else:
            self._append_arg(args, "--watch-dir", self.watch_dir_edit.text())
            self._append_arg(args, "--pattern", self.pattern_edit.text())
            if self.num_energies_spin.value() > 0:
                self._append_arg(args, "--num-energies", self.num_energies_spin.value())
            self._append_arg(args, "--num-groups", self.num_groups_spin.value())
            self._append_arg(args, "--num-frames", self.num_frames_spin.value())
            if self.once_check.isChecked():
                args.append("--once")
            else:
                self._append_arg(args, "--poll-seconds", self.poll_spin.value())
                self._append_arg(args, "--settle-seconds", self.settle_spin.value())

        self._append_arg(args, "--pipeline-root", self.pipeline_root_edit.text())
        self._append_arg(args, "--output-dir", self.output_dir_edit.text())
        self._append_arg(args, "--sample-name", self.sample_name_edit.text())
        self._append_arg(args, "--analysis-h5", self.analysis_h5_edit.text())
        self._append_arg(args, "--analysis-mode", self.analysis_mode_combo.currentText())
        if self.write_text_output_check.isChecked():
            args.append("--write-text-output")
        if self.restart_behavior_combo.currentText() == "restart":
            args.append("--restart")
        self._append_arg(args, "--poni", self.poni_edit.text())
        self._append_arg(args, "--mask", self.mask_edit.text())
        self._append_arg(args, "--dataset-path", self.dataset_path_edit.text())
        self._append_arg(args, "--npt", self.npt_spin.value())
        self._append_arg(args, "--unit", self.unit_edit.text())
        self._append_arg(args, "--detector", self.detector_combo.currentText())
        self._append_arg(args, "--monitor-key", self.monitor_key_edit.text())
        self._append_arg(args, "--delta-energy-percent", self.delta_energy_spin.value())
        self._append_arg(args, "--outlier-zmax", self.outlier_spin.value())

        if self.analysis_mode_combo.currentText() == "asaxs":
            self._append_optional_group(args, "--gc-group", self.gc_group_spin.value())
            self._append_optional_group(args, "--air-group", self.air_group_spin.value())
            self._append_optional_group(args, "--empty-group", self.empty_group_spin.value())
            self._append_optional_group(args, "--water-group", self.water_group_spin.value())
            self._append_optional_group(args, "--sample-group", self.sample_group_spin.value())
            self._append_arg(args, "--gc-reference-file", self.gc_ref_edit.text())
            args.extend(["--gc-q-range", str(self.gc_q_min_spin.value()), str(self.gc_q_max_spin.value())])
            if self.capillary_thickness_spin.value() > 0 or self.gc_thickness_spin.value() > 0:
                self._append_arg(args, "--capillary-thickness", self.capillary_thickness_spin.value())
                self._append_arg(args, "--gc-thickness", self.gc_thickness_spin.value())
            if self.subtract_fluorescence_check.isChecked():
                args.append("--subtract-fluorescence")
                self._append_arg(args, "--fluorescence-reference", self.fluorescence_reference_combo.currentText())
                if self.fluorescence_level_spin.value() != 0:
                    self._append_arg(args, "--fluorescence-level", self.fluorescence_level_spin.value())
                args.extend(
                    [
                        "--fluorescence-q-range",
                        str(self.fluorescence_q_min_spin.value()),
                        str(self.fluorescence_q_max_spin.value()),
                    ]
                )
        return args

    def detector_command_args(self, role: str) -> list[str]:
        """Build one reducer command for a parallel detector job."""
        role = role.lower()
        if role not in {"pil300k", "eig1m"}:
            raise ValueError(f"Unknown detector role: {role}")
        args = self.command_args()
        for flag, has_value in [
            ("--manifest", True),
            ("--watch-dir", True),
            ("--output-dir", True),
            ("--sample-name", True),
            ("--analysis-h5", True),
            ("--detector", True),
            ("--monitor-key", True),
            ("--poni", True),
            ("--mask", True),
        ]:
            args = self._remove_arg(args, flag, has_value)

        detector = "Pil300K" if role == "pil300k" else "Eig1M"
        output_dir = Path(self.output_dir_edit.text()).expanduser().resolve() / detector
        sample_name = f"{self.sample_name_edit.text().strip() or DEFAULT_SAMPLE_NAME}_{detector}"
        if role == "pil300k":
            args.extend(
                [
                    "--watch-dir",
                    self.saxs_watch_dir_edit.text(),
                    "--output-dir",
                    str(output_dir),
                    "--sample-name",
                    sample_name,
                    "--detector",
                    detector,
                    "--poni",
                    self.saxs_poni_edit.text(),
                    "--mask",
                    self.saxs_mask_edit.text(),
                ]
            )
            self._append_arg(args, "--monitor-key", self.saxs_monitor_key_edit.text())
        else:
            args.extend(
                [
                    "--watch-dir",
                    self.waxs_watch_dir_edit.text(),
                    "--output-dir",
                    str(output_dir),
                    "--sample-name",
                    sample_name,
                    "--detector",
                    detector,
                    "--poni",
                    self.waxs_poni_edit.text(),
                    "--mask",
                    self.waxs_mask_edit.text(),
                ]
            )
            self._append_arg(args, "--monitor-key", self.waxs_monitor_key_edit.text())
        return args

    def _remove_arg(self, args: list[str], flag: str, has_value: bool) -> list[str]:
        cleaned: list[str] = []
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if arg == flag:
                skip_next = has_value
                continue
            cleaned.append(arg)
        return cleaned

    def _append_optional_group(self, args: list[str], flag: str, value: int) -> None:
        if value > 0:
            args.extend([flag, str(value)])

    def refresh_command(self) -> None:
        if not hasattr(self, "command_preview"):
            return
        if self._dual_detector_enabled():
            lines = []
            for role in ("pil300k", "eig1m"):
                args = [sys.executable, *self.detector_command_args(role)]
                command = " ".join(f'"{arg}"' if " " in arg else arg for arg in args)
                label = "Pil300K" if role == "pil300k" else "Eig1M"
                lines.append(f"[{label}] {command}")
            self.command_preview.setPlainText("\n".join(lines))
        else:
            args = [sys.executable, *self.command_args()]
            self.command_preview.setPlainText(" ".join(f'"{arg}"' if " " in arg else arg for arg in args))

    def start_reducer(self) -> None:
        if self._dual_detector_enabled():
            self.start_dual_reducers()
            return
        if self.process is not None and self.process.state() != QtCore.QProcess.NotRunning:
            QtWidgets.QMessageBox.information(self, "Reducer already running", "Stop the current reducer first.")
            return
        self._save_settings()
        output_dir = Path(self.output_dir_edit.text()).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.restart_behavior_combo.currentText() == "restart":
            event_log_path = output_dir / "live_events.jsonl"
            if event_log_path.exists():
                try:
                    event_log_path.unlink()
                except OSError as exc:
                    QtWidgets.QMessageBox.warning(self, "Could not clear old monitor log", str(exc))
                    return
        self.monitor_window.set_output_dir(output_dir)
        self.monitor_window.set_expected_frames(self._expected_monitor_frames())
        self.monitor_window.clear_run_display()
        self.curve_window.output_dir_edit.setText(str(output_dir))
        self.curve_window._refresh_now()

        self.process = QtCore.QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(self.command_args())
        self.process.setWorkingDirectory(str(PLAYGROUND_DIR))
        self.monitor_window.attach_process(self.process)
        self.monitor_window.show()
        self.curve_window.show()
        self.process.start()
        if not self.process.waitForStarted(3000):
            QtWidgets.QMessageBox.critical(self, "Reducer failed to start", self.process.errorString())
            return
        self.statusBar().showMessage("Reducer started")

    def start_dual_reducers(self) -> None:
        if any(process.state() != QtCore.QProcess.NotRunning for process in self.detector_processes.values()):
            QtWidgets.QMessageBox.information(self, "Reducers already running", "Stop the current reducers first.")
            return
        self._save_settings()
        base_output = Path(self.output_dir_edit.text()).expanduser().resolve()
        base_output.mkdir(parents=True, exist_ok=True)
        self.detector_processes.clear()
        combined_h5 = self._combined_analysis_h5_path()
        if self.restart_behavior_combo.currentText() == "restart" and combined_h5.exists():
            try:
                combined_h5.unlink()
            except OSError as exc:
                QtWidgets.QMessageBox.warning(self, "Could not clear old combined analysis HDF5", str(exc))
                return
        for role in ("pil300k", "eig1m"):
            detector = "Pil300K" if role == "pil300k" else "Eig1M"
            output_dir = base_output / detector
            output_dir.mkdir(parents=True, exist_ok=True)
            if self.restart_behavior_combo.currentText() == "restart":
                event_log_path = output_dir / "live_events.jsonl"
                if event_log_path.exists():
                    try:
                        event_log_path.unlink()
                    except OSError as exc:
                        QtWidgets.QMessageBox.warning(self, f"Could not clear old {detector} monitor log", str(exc))
                        return
            monitor = self.dual_monitor_windows[role]
            viewer = self.dual_curve_windows[role]
            monitor.set_output_dir(output_dir)
            monitor.set_expected_frames(self._expected_monitor_frames())
            monitor.clear_run_display()
            viewer.output_dir_edit.setText(str(output_dir))
            viewer._refresh_now()

            process = QtCore.QProcess(self)
            process.setProgram(sys.executable)
            process.setArguments(self.detector_command_args(role))
            process.setWorkingDirectory(str(PLAYGROUND_DIR))
            monitor.attach_process(process)
            monitor.show()
            viewer.show()
            process.start()
            self.detector_processes[role] = process
        failed = [role for role, process in self.detector_processes.items() if not process.waitForStarted(3000)]
        if failed:
            QtWidgets.QMessageBox.critical(self, "Reducer failed to start", ", ".join(failed))
            return
        self.dual_curve_windows["stitched"].output_dir_edit.setText(str(combined_h5))
        self.stitch_timer.start()
        self.statusBar().showMessage("Pil300K and Eig1M reducers started")

    def stop_reducer(self) -> None:
        if self._dual_detector_enabled() or self.detector_processes:
            self.stop_dual_reducers()
            return
        if self.process is None or self.process.state() == QtCore.QProcess.NotRunning:
            return
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()
        self.statusBar().showMessage("Reducer stopped")

    def stop_dual_reducers(self) -> None:
        stopped = False
        for process in self.detector_processes.values():
            if process.state() == QtCore.QProcess.NotRunning:
                continue
            process.terminate()
            if not process.waitForFinished(3000):
                process.kill()
            stopped = True
        if stopped:
            self.statusBar().showMessage("Pil300K/Eig1M reducers stopped")
        self._update_live_stitched_outputs()
        self.stitch_timer.stop()

    def show_monitor(self) -> None:
        if self._dual_detector_enabled():
            base_output = Path(self.output_dir_edit.text()).expanduser().resolve()
            for role, monitor in self.dual_monitor_windows.items():
                detector = "Pil300K" if role == "pil300k" else "Eig1M"
                monitor.set_output_dir(base_output / detector)
                monitor.set_expected_frames(self._expected_monitor_frames())
                monitor.show()
                monitor.raise_()
            return
        self.monitor_window.set_output_dir(Path(self.output_dir_edit.text()).expanduser().resolve())
        self.monitor_window.set_expected_frames(self._expected_monitor_frames())
        self.monitor_window.show()
        self.monitor_window.raise_()

    def show_curves(self) -> None:
        if self._dual_detector_enabled():
            base_output = Path(self.output_dir_edit.text()).expanduser().resolve()
            for role, window in self.dual_curve_windows.items():
                if role == "stitched":
                    window.output_dir_edit.setText(str(self._combined_analysis_h5_path()))
                else:
                    detector = "Pil300K" if role == "pil300k" else "Eig1M"
                    window.output_dir_edit.setText(str(base_output / detector))
                window._refresh_now()
                window.show()
                window.raise_()
            return
        self.curve_window.output_dir_edit.setText(str(Path(self.output_dir_edit.text()).expanduser().resolve()))
        self.curve_window._refresh_now()
        self.curve_window.show()
        self.curve_window.raise_()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt method name.
        self._save_settings()
        super().closeEvent(event)

    def _expected_monitor_frames(self) -> int | None:
        if self.source_mode_combo.currentText() == "watch folder":
            energies = self.num_energies_spin.value()
            if energies <= 0:
                return None
            return energies * self.num_groups_spin.value() * self.num_frames_spin.value()

        # Manifest replay may use limit controls for quick tests. If the energy
        # limit is unknown/zero, leave the monitor in activity mode.
        limit_energies = self.limit_energies_spin.value()
        limit_frames = self.limit_frames_spin.value()
        if limit_energies <= 0:
            return None
        frames_per_group = limit_frames if limit_frames > 0 else self.num_frames_spin.value()
        return limit_energies * self.num_groups_spin.value() * frames_per_group

    def _update_live_stitched_outputs(self) -> None:
        if not self._dual_detector_enabled():
            return
        base_output = Path(self.output_dir_edit.text()).expanduser().resolve()
        stitched_h5 = update_live_stitched_averages(
            base_output / "Pil300K",
            base_output / "Eig1M",
            self._combined_analysis_h5_path(),
        )
        if stitched_h5 is not None:
            self.dual_curve_windows["stitched"].output_dir_edit.setText(str(stitched_h5))
            self.dual_curve_windows["stitched"]._refresh_now()

    def _combined_analysis_h5_path(self) -> Path:
        """Return the one public analysis HDF5 path for the current batch."""
        explicit = self.analysis_h5_edit.text().strip()
        if explicit:
            return Path(explicit).expanduser().resolve()
        sample = self.sample_name_edit.text().strip() or DEFAULT_SAMPLE_NAME
        safe_sample = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample).strip("._") or "analysis"
        return Path(self.output_dir_edit.text()).expanduser().resolve() / f"{safe_sample}_analysis.h5"


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = SetupWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
