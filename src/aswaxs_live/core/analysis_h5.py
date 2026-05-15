"""Helpers for writing the analysis HDF5 file.

The central rule in this module is separation of concerns:
source beamline HDF5 files are read-only inputs, while analysis HDF5 files
hold all reduction outputs, copied/summarized source metadata, and provenance.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


UNKNOWN = "unknown"
METADATA_COPY_MAX_ITEMS = 4096
METADATA_COPY_MAX_BYTES = 1024 * 1024
PROCESS_GROUPS = {
    "reduction": "process_01_reduction",
    "background_subtraction": "process_02_background_subtraction",
    "glassy_carbon_normalization": "process_03_glassy_carbon_normalization",
    "asaxs_component_extraction": "process_04_asaxs_component_extraction",
}


def file_sha256(path: str | Path | None, chunk_size: int = 1024 * 1024) -> str:
    if path is None:
        return UNKNOWN
    resolved = Path(path).expanduser()
    if not resolved.exists() or not resolved.is_file():
        return UNKNOWN
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_sha256(dataset: h5py.Dataset, max_bytes: int = 100 * 1024 * 1024) -> str:
    if dataset.size * dataset.dtype.itemsize > max_bytes:
        return UNKNOWN
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(dataset[()]).tobytes())
    return digest.hexdigest()


def current_process_metadata(
    process_name: str,
    process_stage: str,
    input_h5_file: str | Path | None = None,
    input_data_path: str | None = None,
    output_h5_file: str | Path | None = None,
    output_data_path: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "process_name": process_name,
        "process_stage": process_stage,
        "program": Path(sys.argv[0]).name or UNKNOWN,
        "version": UNKNOWN,
        "date": datetime.now(timezone.utc).isoformat(),
        "user": getpass.getuser() or UNKNOWN,
        "hostname": socket.gethostname() or platform.node() or UNKNOWN,
        "command": " ".join(sys.argv) or UNKNOWN,
        "git_commit": _git_commit(),
        "input_h5_file": str(input_h5_file) if input_h5_file is not None else UNKNOWN,
        "input_data_path": input_data_path or UNKNOWN,
        "output_h5_file": str(output_h5_file) if output_h5_file is not None else UNKNOWN,
        "output_data_path": output_data_path or UNKNOWN,
        "notes": notes or UNKNOWN,
    }


def create_analysis_h5_from_data(
    data_h5_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    analysis_h5_path: str | Path,
    data_reference_metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Create or refresh the analysis file's source-data reference section."""
    analysis_path = Path(analysis_h5_path).expanduser().resolve()
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    if analysis_path.exists() and overwrite:
        analysis_path.unlink()

    raw_paths = _as_path_list(data_h5_path)
    for raw_path in raw_paths:
        with h5py.File(raw_path, "r"):
            pass

    mode = "w" if overwrite or not analysis_path.exists() else "a"
    with h5py.File(analysis_path, mode) as handle:
        entry = handle.require_group("entry")
        data_ref = _replace_group(entry, "data_reference")
        metadata = _data_reference_defaults(raw_paths, data_reference_metadata)
        _write_mapping(data_ref, metadata)
        _write_original_data_metadata(
            data_ref,
            raw_paths,
            metadata.get("data_metadata_path", "entry/instrument/NDAttributes"),
        )
    return analysis_path


def create_analysis_h5_from_raw(
    raw_h5_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    analysis_h5_path: str | Path,
    raw_reference_metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    return create_analysis_h5_from_data(
        raw_h5_path,
        analysis_h5_path,
        data_reference_metadata=raw_reference_metadata,
        overwrite=overwrite,
    )


def write_process_group(
    analysis_h5_path: str | Path,
    process_name: str,
    process_stage: str,
    metadata: Mapping[str, Any] | None,
    parameters: Mapping[str, Any] | None,
    data: Mapping[str, Any] | None,
    previous_metadata: Mapping[str, Any] | None = None,
    input_paths: Mapping[str, Any] | None = None,
    output_path: str | None = None,
) -> str:
    """Write one versioned NXprocess group without overwriting previous runs."""
    analysis_path = Path(analysis_h5_path).expanduser().resolve()
    with h5py.File(analysis_path, "a") as handle:
        entry = handle.require_group("entry")
        base_name = PROCESS_GROUPS.get(process_stage, process_name)
        group_name = _next_versioned_name(entry, base_name)
        process_group = entry.create_group(group_name)
        process_group.attrs["NX_class"] = "NXprocess"

        output_data_path = output_path or f"/entry/{group_name}/data"
        merged_metadata = current_process_metadata(
            process_name=process_name,
            process_stage=process_stage,
            output_h5_file=analysis_path,
            output_data_path=output_data_path,
        )
        if metadata:
            merged_metadata.update(dict(metadata))
        if input_paths:
            merged_metadata.update(dict(input_paths))

        _write_mapping(process_group.create_group("metadata"), merged_metadata)
        _write_mapping(process_group.create_group("previous_metadata"), previous_metadata or {})
        _write_mapping(process_group.create_group("parameters"), parameters or {})
        data_group = process_group.create_group("data")
        _write_mapping(data_group, data or {})
        _mark_nxdata(data_group)
        return f"/entry/{group_name}"


def write_reduction_to_analysis_h5(
    analysis_h5_path: str | Path,
    raw_h5_path: str | Path,
    q: Any,
    I: Any,
    sigma_I: Any,
    reduction_metadata: Mapping[str, Any],
    reduction_parameters: Mapping[str, Any],
    frame_filter_log: Mapping[str, Any] | None = None,
    I_frame_q: Any | None = None,
    sigma_frame_q: Any | None = None,
) -> str:
    if not Path(analysis_h5_path).exists():
        create_analysis_h5_from_data(raw_h5_path, analysis_h5_path)
    previous = {
        "data_reference_summary": read_data_reference_summary(analysis_h5_path),
        **summarize_raw_acquisition(raw_h5_path, reduction_metadata.get("input_data_path", "entry/data/data")),
    }
    data = {
        "q": q,
        "I": I,
        "sigma_I": sigma_I,
        "n_total_frames": reduction_metadata.get("n_total_frames", UNKNOWN),
        "n_accepted_frames": reduction_metadata.get("n_accepted_frames", UNKNOWN),
        "n_rejected_frames": reduction_metadata.get("n_rejected_frames", UNKNOWN),
    }
    if I_frame_q is not None:
        data["I_frame_q"] = I_frame_q
    if sigma_frame_q is not None:
        data["sigma_frame_q"] = sigma_frame_q
    if frame_filter_log is not None:
        data["frame_filter_log"] = frame_filter_log
    group_path = write_process_group(
        analysis_h5_path,
        "reduction",
        "reduction",
        reduction_metadata,
        reduction_parameters,
        data,
        previous_metadata=previous,
        output_path="/entry/process_01_reduction/data",
    )
    _move_frame_filter_log(analysis_h5_path, group_path)
    return group_path


def write_background_subtraction_to_analysis_h5(
    analysis_h5_path: str | Path,
    corrected_data: Mapping[str, Any],
    subtraction_metadata: Mapping[str, Any],
    subtraction_parameters: Mapping[str, Any],
    subtraction_map: Mapping[str, Any],
) -> str:
    previous = _previous_for_stage(analysis_h5_path, "background_subtraction")
    data = dict(corrected_data)
    data["subtraction_map"] = subtraction_map
    group_path = write_process_group(
        analysis_h5_path,
        "background_subtraction",
        "background_subtraction",
        subtraction_metadata,
        subtraction_parameters,
        data,
        previous_metadata=previous,
        output_path="/entry/process_02_background_subtraction/data",
    )
    _move_child_to_process_group(analysis_h5_path, group_path, "data/subtraction_map", "subtraction_map")
    return group_path


def write_glassy_carbon_normalization_to_analysis_h5(
    analysis_h5_path: str | Path,
    normalized_data: Mapping[str, Any],
    normalization_metadata: Mapping[str, Any],
    normalization_parameters: Mapping[str, Any],
    normalization_factors: Mapping[str, Any],
) -> str:
    previous = _previous_for_stage(analysis_h5_path, "glassy_carbon_normalization")
    data = dict(normalized_data)
    data["normalization_factors"] = normalization_factors
    group_path = write_process_group(
        analysis_h5_path,
        "glassy_carbon_normalization",
        "glassy_carbon_normalization",
        normalization_metadata,
        normalization_parameters,
        data,
        previous_metadata=previous,
        output_path="/entry/process_03_glassy_carbon_normalization/data",
    )
    _move_child_to_process_group(analysis_h5_path, group_path, "data/normalization_factors", "normalization_factors")
    return group_path


def write_asaxs_components_to_analysis_h5(
    analysis_h5_path: str | Path,
    corrected_I_q_E: Mapping[str, Any],
    components: Mapping[str, Any],
    energy_table: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    fit_diagnostics: Mapping[str, Any],
    asaxs_metadata: Mapping[str, Any],
) -> str:
    previous = _previous_for_stage(analysis_h5_path, "asaxs_component_extraction")
    data = {
        "corrected_I_q_E": corrected_I_q_E,
        "components": components,
        "energy_table": energy_table,
        "model": model_metadata,
        "fit_diagnostics": fit_diagnostics,
    }
    group_path = write_process_group(
        analysis_h5_path,
        "asaxs_component_extraction",
        "asaxs_component_extraction",
        asaxs_metadata,
        {},
        data,
        previous_metadata=previous,
        output_path="/entry/process_04_asaxs_component_extraction/components",
    )
    with h5py.File(analysis_h5_path, "a") as handle:
        group = handle[group_path]
        for name in ("corrected_I_q_E", "components", "energy_table", "model", "fit_diagnostics"):
            if f"data/{name}" in group:
                handle.move(f"{group_path}/data/{name}", f"{group_path}/{name}")
        if "corrected_I_q_E" in group:
            group["corrected_I_q_E"].attrs["NX_class"] = "NXdata"
            group["corrected_I_q_E"].attrs["signal"] = "I"
            group["corrected_I_q_E"].attrs["axes"] = np.asarray(["energy", "q"], dtype=h5py.string_dtype("utf-8"))
        if "components" in group:
            group["components"].attrs["NX_class"] = "NXdata"
            group["components"].attrs["signal"] = "I_resonant"
            group["components"].attrs["axes"] = "q"
        _write_final_group(handle, group)
    return group_path


def validate_analysis_h5(analysis_h5_path: str | Path) -> list[str]:
    errors: list[str] = []
    with h5py.File(analysis_h5_path, "r") as handle:
        if "entry/data_reference" not in handle:
            errors.append("missing /entry/data_reference")
        process_groups = [name for name in handle.get("entry", {}) if name.startswith("process_")]
        for name in process_groups:
            group = handle[f"entry/{name}"]
            if group.attrs.get("NX_class") != "NXprocess":
                errors.append(f"/entry/{name} missing NX_class=NXprocess")
            for child in ("metadata", "parameters", "previous_metadata"):
                if child not in group:
                    errors.append(f"/entry/{name} missing {child}")
    return errors


def read_data_reference_summary(analysis_h5_path: str | Path) -> dict[str, Any]:
    with h5py.File(analysis_h5_path, "r") as handle:
        if "entry/data_reference" not in handle:
            return {}
        return _group_to_summary(handle["entry/data_reference"])


def read_raw_reference_summary(analysis_h5_path: str | Path) -> dict[str, Any]:
    return read_data_reference_summary(analysis_h5_path)


def summarize_raw_acquisition(raw_h5_path: str | Path, detector_path: str = "entry/data/data") -> dict[str, Any]:
    summary = {
        "acquisition_summary": {},
        "detector_summary": {},
        "monitor_summary": {},
        "sample_summary": {},
        "scan_summary": {},
    }
    with h5py.File(raw_h5_path, "r") as handle:
        if detector_path in handle:
            dataset = handle[detector_path]
            summary["detector_summary"] = {
                "input_h5_file": str(raw_h5_path),
                "input_data_path": detector_path,
                "input_dataset_shape": list(dataset.shape),
                "input_dataset_dtype": str(dataset.dtype),
                "input_dataset_hash": dataset_sha256(dataset),
            }
        if "entry/instrument/NDAttributes" in handle:
            ndattrs = handle["entry/instrument/NDAttributes"]
            summary["acquisition_summary"]["original_metadata_path"] = ndattrs.name
            summary["acquisition_summary"]["all_pvs"] = _metadata_group_to_mapping(ndattrs)
            summary["monitor_summary"] = _metadata_group_to_mapping(ndattrs)
        for key in ("entry/start_time", "entry/title", "entry/sample/name", "entry/scan_id", "entry/run_uid"):
            if key in handle:
                target = "sample_summary" if "sample" in key else "scan_summary"
                summary[target][key.rsplit("/", 1)[-1]] = _read_small_dataset(handle[key])
    return summary


def _write_original_data_metadata(data_reference_group: h5py.Group, raw_paths: list[Path], data_metadata_path: Any) -> None:
    # Store two views of the source metadata:
    # tree = the source HDF5 hierarchy with large datasets summarized;
    # pvs = the configured PV/NDAttributes path for quick human inspection.
    metadata_group = data_reference_group.create_group("original_metadata")
    metadata_paths = _metadata_paths_for_data_files(data_metadata_path, len(raw_paths))
    for index, raw_path in enumerate(raw_paths):
        data_group = metadata_group.create_group(f"data_{index + 1:06d}")
        data_group.create_dataset("data_file", data=str(raw_path), dtype=h5py.string_dtype("utf-8"))
        metadata_path = metadata_paths[index]
        data_group.create_dataset("data_metadata_path", data=metadata_path, dtype=h5py.string_dtype("utf-8"))
        with h5py.File(raw_path, "r") as raw_handle:
            tree = data_group.create_group("tree")
            _copy_metadata_tree(raw_handle, tree)
            if metadata_path in raw_handle:
                pvs = data_group.create_group("pvs")
                _copy_metadata_tree(raw_handle[metadata_path], pvs)
            else:
                data_group.create_dataset("notes", data=f"metadata path not found: {metadata_path}", dtype=h5py.string_dtype("utf-8"))


def _metadata_paths_for_data_files(data_metadata_path: Any, count: int) -> list[str]:
    if isinstance(data_metadata_path, (list, tuple, np.ndarray)):
        values = [str(value) for value in data_metadata_path]
        if len(values) == count:
            return values
        if values:
            return [values[0]] * count
    return [str(data_metadata_path or "entry/instrument/NDAttributes")] * count


def _copy_metadata_tree(source: h5py.Group | h5py.Dataset, dest: h5py.Group) -> None:
    _copy_attrs(source, dest)
    if isinstance(source, h5py.Dataset):
        _copy_or_summarize_dataset(source, dest, "value")
        return
    for key, value in source.items():
        if isinstance(value, h5py.Dataset):
            _copy_or_summarize_dataset(value, dest, key)
        elif isinstance(value, h5py.Group):
            child = dest.create_group(key)
            _copy_metadata_tree(value, child)


def _copy_or_summarize_dataset(source: h5py.Dataset, dest: h5py.Group, key: str) -> None:
    # Detector images and other large arrays should remain in the source file.
    # The analysis file keeps their location and shape so provenance is complete
    # without duplicating acquisition data.
    byte_count = source.size * source.dtype.itemsize
    if source.size <= METADATA_COPY_MAX_ITEMS and byte_count <= METADATA_COPY_MAX_BYTES:
        dest.file.copy(source, dest, name=key)
        return
    summary = dest.create_group(key)
    _copy_attrs(source, summary)
    summary.attrs["summarized_instead_of_copied"] = True
    summary.create_dataset("source_path", data=source.name, dtype=h5py.string_dtype("utf-8"))
    summary.create_dataset("shape", data=np.asarray(source.shape, dtype=np.int64))
    summary.create_dataset("dtype", data=str(source.dtype), dtype=h5py.string_dtype("utf-8"))
    summary.create_dataset("notes", data="dataset omitted from analysis file because it exceeds metadata copy limits", dtype=h5py.string_dtype("utf-8"))


def _copy_attrs(source: h5py.Group | h5py.Dataset, dest: h5py.Group | h5py.Dataset) -> None:
    for key, value in source.attrs.items():
        dest.attrs[key] = value


def _metadata_group_to_mapping(group: h5py.Group) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, h5py.Dataset):
            if value.size <= METADATA_COPY_MAX_ITEMS and value.size * value.dtype.itemsize <= METADATA_COPY_MAX_BYTES:
                metadata[key] = _read_small_dataset(value, max_items=METADATA_COPY_MAX_ITEMS)
            else:
                metadata[key] = {
                    "source_path": value.name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "summarized_instead_of_copied": True,
                }
        elif isinstance(value, h5py.Group):
            metadata[key] = _metadata_group_to_mapping(value)
    return metadata


def _previous_for_stage(analysis_h5_path: str | Path, process_stage: str) -> dict[str, Any]:
    # Previous-step metadata is intentionally compact: parameters, metadata, and
    # output paths/shapes are carried forward, but large arrays are not copied.
    previous: dict[str, Any] = {"data_reference_summary": read_data_reference_summary(analysis_h5_path)}
    with h5py.File(analysis_h5_path, "r") as handle:
        entry = handle["entry"]
        if process_stage == "background_subtraction":
            _add_process_summary(previous, entry, "process_01_reduction")
        elif process_stage == "glassy_carbon_normalization":
            _add_process_summary(previous, entry, "process_01_reduction")
            _add_process_summary(previous, entry, "process_02_background_subtraction")
        elif process_stage == "asaxs_component_extraction":
            _add_process_summary(previous, entry, "process_01_reduction")
            _add_process_summary(previous, entry, "process_02_background_subtraction")
            _add_process_summary(previous, entry, "process_03_glassy_carbon_normalization")
    return previous


def _add_process_summary(target: dict[str, Any], entry: h5py.Group, base_name: str) -> None:
    name = _latest_versioned_name(entry, base_name)
    if not name:
        return
    process = entry[name]
    simple_name = base_name
    if "metadata" in process:
        target[f"{simple_name}_metadata"] = _group_to_summary(process["metadata"])
    if "parameters" in process:
        target[f"{simple_name}_parameters"] = _group_to_summary(process["parameters"])
    output_summary = {"output_data_path": f"/entry/{name}/data"}
    if "data" in process:
        output_summary["datasets"] = _dataset_shapes(process["data"])
    target[f"{simple_name}_output_summary"] = output_summary


def _data_reference_defaults(raw_paths: list[Path], extra: Mapping[str, Any] | None) -> dict[str, Any]:
    first = raw_paths[0] if raw_paths else None
    metadata = {
        "data_file": [str(path) for path in raw_paths] if len(raw_paths) > 1 else str(first),
        "data_entry": "entry",
        "data_detector_path": "entry/data/data",
        "data_metadata_path": "entry/instrument/NDAttributes",
        "data_file_hash": [file_sha256(path) for path in raw_paths] if len(raw_paths) > 1 else file_sha256(first),
        "source_run_uid": UNKNOWN,
        "source_scan_id": UNKNOWN,
        "source_frame_indices": UNKNOWN,
        "source_frame_count": len(raw_paths) if raw_paths else UNKNOWN,
        "notes": "source data HDF5 file is treated as read-only; all processing products live in this analysis file",
    }
    if extra:
        metadata.update(_normalize_data_reference_keys(dict(extra)))
    return metadata


def _normalize_data_reference_keys(metadata: dict[str, Any]) -> dict[str, Any]:
    replacements = {
        "raw_file": "data_file",
        "raw_entry": "data_entry",
        "raw_detector_path": "data_detector_path",
        "raw_metadata_path": "data_metadata_path",
        "raw_file_hash": "data_file_hash",
    }
    for old, new in replacements.items():
        if old in metadata and new not in metadata:
            metadata[new] = metadata.pop(old)
    return metadata


def _write_mapping(group: h5py.Group, values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        safe_key = str(key).replace("/", "_")
        _write_value(group, safe_key, value)


def _write_value(group: h5py.Group, key: str, value: Any) -> None:
    if value is None:
        value = UNKNOWN
    if isinstance(value, Mapping):
        child = group.create_group(key)
        _write_mapping(child, value)
        return
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, Mapping) for item in value):
        child = group.create_group(key)
        for index, item in enumerate(value):
            _write_mapping(child.create_group(f"item_{index:04d}"), item)
        return
    if isinstance(value, (str, Path)):
        group.create_dataset(key, data=str(value), dtype=h5py.string_dtype("utf-8"))
        return
    if isinstance(value, bytes):
        group.create_dataset(key, data=value.decode("utf-8", errors="replace"), dtype=h5py.string_dtype("utf-8"))
        return
    array = np.asarray(value)
    if array.dtype.kind in {"U", "O"}:
        try:
            string_array = np.asarray([str(item) for item in array.reshape(-1)], dtype=h5py.string_dtype("utf-8"))
            string_array = string_array.reshape(array.shape)
            group.create_dataset(key, data=string_array)
        except TypeError:
            group.create_dataset(key, data=json.dumps(value, default=str), dtype=h5py.string_dtype("utf-8"))
    else:
        group.create_dataset(key, data=array)


def _mark_nxdata(group: h5py.Group) -> None:
    if "I" in group and "q" in group:
        group.attrs["NX_class"] = "NXdata"
        group.attrs["signal"] = "I"
        group.attrs["axes"] = "q"


def _replace_group(parent: h5py.Group, name: str) -> h5py.Group:
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def _next_versioned_name(parent: h5py.Group, base_name: str) -> str:
    if base_name not in parent:
        return base_name
    version = 2
    while f"{base_name}_v{version:03d}" in parent:
        version += 1
    return f"{base_name}_v{version:03d}"


def _latest_versioned_name(parent: h5py.Group, base_name: str) -> str | None:
    candidates = [name for name in parent if name == base_name or name.startswith(f"{base_name}_v")]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _as_path_list(raw_h5_path: str | Path | list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    if isinstance(raw_h5_path, (list, tuple)):
        return [Path(path).expanduser().resolve() for path in raw_h5_path]
    return [Path(raw_h5_path).expanduser().resolve()]


def _read_small_dataset(dataset: h5py.Dataset, max_items: int = 16) -> Any:
    if dataset.size > max_items:
        return {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    array = np.asarray(value)
    if array.shape == ():
        scalar = array.item()
        return scalar.decode("utf-8", errors="replace") if isinstance(scalar, bytes) else scalar
    return array.tolist()


def _group_to_summary(group: h5py.Group) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, h5py.Dataset):
            summary[key] = _read_small_dataset(value)
        elif isinstance(value, h5py.Group):
            summary[key] = _group_to_summary(value)
    return summary


def _dataset_shapes(group: h5py.Group) -> dict[str, Any]:
    shapes: dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, h5py.Dataset):
            shapes[key] = {"path": value.name, "shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, h5py.Group):
            shapes[key] = _dataset_shapes(value)
    return shapes


def _move_frame_filter_log(analysis_h5_path: str | Path, group_path: str) -> None:
    _move_child_to_process_group(analysis_h5_path, group_path, "data/frame_filter_log", "frame_filter_log")


def _move_child_to_process_group(analysis_h5_path: str | Path, group_path: str, source: str, target: str) -> None:
    with h5py.File(analysis_h5_path, "a") as handle:
        group = handle[group_path]
        if source in group:
            if target in group:
                del group[target]
            handle.move(f"{group_path}/{source}", f"{group_path}/{target}")


def _write_final_group(handle: h5py.File, process_group: h5py.Group) -> None:
    entry = handle["entry"]
    final = _replace_group(entry, "final")
    if "corrected_I_q_E" in process_group:
        _copy_group(process_group["corrected_I_q_E"], final.create_group("corrected_I_q_E"))
    if "components" in process_group:
        _copy_group(process_group["components"], final.create_group("asaxs_components"))


def _copy_group(source: h5py.Group, dest: h5py.Group) -> None:
    for attr_key, attr_value in source.attrs.items():
        dest.attrs[attr_key] = attr_value
    for key, value in source.items():
        if isinstance(value, h5py.Dataset):
            source.file.copy(value, dest, name=key)
        elif isinstance(value, h5py.Group):
            _copy_group(value, dest.create_group(key))


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.getcwd(),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return UNKNOWN
    return result.stdout.strip() or UNKNOWN
