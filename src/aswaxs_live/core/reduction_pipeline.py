"""Single-file or small-list SAXS/ASAXS image reduction.

This script is the simple reducer used by the watcher and by ad hoc runs. It
opens source HDF5 files read-only, performs pyFAI 1D integration, optionally
subtracts background/fluorescence and applies glassy-carbon scaling, then writes
both legacy text/NPZ outputs and the structured analysis HDF5 record.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import fabio
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

PROJECT_DIR = Path(__file__).resolve().parent


DEFAULT_DATASET_PATH = "entry/data/data"
DEFAULT_NPT = 1000
DEFAULT_GC_Q_RANGE = (0.03, 0.20)
DEFAULT_FLUORESCENCE_Q_RANGE = (0.16, 0.20)
HC_ANGSTROM_KEV = 12.398419843320026
NDATTR_PREFIX = "entry/instrument/NDAttributes"

# NIST SRM 3600 glassy carbon certified reference curve used for absolute scaling.
# Values are q (A^-1) and differential scattering cross section.
NIST_SRM3600_Q = np.array(
    [
        0.00827568, 0.00888450, 0.00954735, 0.01026900, 0.01105780, 0.01191830,
        0.01286110, 0.01389340, 0.01502510, 0.01626850, 0.01763650, 0.01914320,
        0.02080510, 0.02264220, 0.02467500, 0.02692890, 0.02943170, 0.03221560,
        0.03531810, 0.03878270, 0.04265880, 0.04700390, 0.05188580, 0.05738140,
        0.06358290, 0.07059620, 0.07854840, 0.08758630, 0.09788540, 0.10965500,
        0.11431200, 0.11839500, 0.12262400, 0.12314200, 0.12700400, 0.13154000,
        0.13623900, 0.13864300, 0.14110500, 0.14614500, 0.15136500, 0.15651300,
        0.15677100, 0.16237100, 0.16817000, 0.17417700, 0.17718100, 0.18039800,
        0.18684100, 0.19351500, 0.20042700, 0.20116500, 0.20758600, 0.21500000,
        0.22267900, 0.22909500, 0.23063300, 0.23887100, 0.24740200,
    ],
    dtype=float,
)
NIST_SRM3600_I = np.array(
    [
        34.933380, 34.427156, 34.042170, 33.698553, 33.352529, 33.027533,
        32.665045, 32.306665, 31.970485, 31.559099, 31.183763, 30.861805,
        30.514300, 30.084982, 29.690414, 29.249965, 28.889970, 28.449341,
        28.065980, 27.704965, 27.331304, 26.974065, 26.676952, 26.401158,
        26.177427, 25.904683, 25.528734, 24.917743, 23.946472, 22.472101,
        21.777228, 21.112938, 20.401110, 20.287060, 19.685107, 18.909809,
        18.089242, 17.679572, 17.264117, 16.372848, 15.458350, 14.587700,
        14.563071, 13.616671, 12.668549, 11.752287, 11.311460, 10.862157,
        9.961979, 9.116906, 8.325578, 8.224897, 7.541931, 6.854391, 6.216070,
        5.715911, 5.582366, 4.999113, 4.463604,
    ],
    dtype=float,
)


@dataclass
class ReductionInputs:
    input_hdf5: list[Path]
    poni_file: Path
    mask_file: Path
    output_dir: Path
    background_hdf5: Path | None
    glassy_carbon_hdf5: Path | None
    glassy_carbon_background_hdf5: Path | None
    gc_reference_file: Path | None


@dataclass
class IntegrationResult:
    q: np.ndarray
    intensity: np.ndarray
    background: np.ndarray | None
    corrected: np.ndarray
    absolute: np.ndarray | None = None
    fluorescence_background: float | None = None
    fluorescence_corrected: np.ndarray | None = None


@dataclass
class GlassyCarbonCalibration:
    scale_factor: float
    q_range: tuple[float, float]
    measured_area: float
    reference_area: float
    gc_result: IntegrationResult
    reference_q: np.ndarray
    reference_i: np.ndarray


def _read_hdf5_image(path: Path, dataset_path: str, frame: int | None) -> np.ndarray:
    """Read a 2D image or average/select from a 3D frame stack."""
    with h5py.File(path, "r") as handle:
        if dataset_path not in handle:
            raise KeyError(f"Dataset '{dataset_path}' was not found in {path}")
        data = np.asarray(handle[dataset_path][()], dtype=np.float32)

    if data.ndim == 2:
        return data
    if data.ndim == 3:
        if frame is None:
            return np.nanmean(data, axis=0).astype(np.float32)
        return np.asarray(data[frame], dtype=np.float32)
    raise ValueError(f"Expected a 2D image or 3D frame stack in {path}; found shape {data.shape}")


def _load_mask(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        mask = np.load(path)
    else:
        with fabio.open(str(path)) as image:
            mask = np.asarray(image.data)
    return mask.astype(bool)


def _load_reference_curve(path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if path is None:
        return NIST_SRM3600_Q, NIST_SRM3600_I
    data = np.loadtxt(path, comments="#", delimiter=None)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Reference curve must contain at least two columns: {path}")
    return np.asarray(data[:, 0], dtype=float), np.asarray(data[:, 1], dtype=float)


def _integrated_intensity(q: np.ndarray, intensity: np.ndarray, q_range: tuple[float, float] | None = None) -> float:
    mask = np.isfinite(q) & np.isfinite(intensity)
    if q_range is not None:
        qmin, qmax = q_range
        mask &= (q >= qmin) & (q <= qmax)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.trapezoid(intensity[mask], q[mask]))


def _optional_path_text(path: str | Path | None) -> str | None:
    return str(path) if path else None


def energy_kev_to_wavelength_m(energy_kev: float) -> float:
    if energy_kev <= 0:
        raise ValueError("Energy must be positive to convert to wavelength.")
    return (HC_ANGSTROM_KEV / energy_kev) * 1e-10


def _read_ndattr_scalar(path: Path, key: str) -> float | None:
    full_key = f"{NDATTR_PREFIX}/{key}"
    with h5py.File(path, "r") as handle:
        if full_key not in handle:
            return None
        value = np.asarray(handle[full_key][()])
    if value.size == 0:
        return None
    return float(value.reshape(-1)[0])


def _set_ai_wavelength_from_hdf5(ai, path: Path) -> float | None:
    energy_kev = _read_ndattr_scalar(path, "Mono_Energy")
    if energy_kev is None:
        return None
    ai.wavelength = energy_kev_to_wavelength_m(energy_kev)
    return energy_kev


def estimate_constant_fluorescence(
    q: np.ndarray,
    intensity: np.ndarray,
    q_range: tuple[float, float],
) -> float:
    qmin, qmax = q_range
    if qmin >= qmax:
        raise ValueError("Fluorescence q-range minimum must be smaller than maximum.")
    mask = np.isfinite(q) & np.isfinite(intensity) & (q >= qmin) & (q <= qmax)
    if np.count_nonzero(mask) == 0:
        raise ValueError("No finite data points overlap the fluorescence q range.")
    return float(np.mean(intensity[mask]))


def apply_fluorescence_subtraction(
    result: IntegrationResult,
    q_range: tuple[float, float],
    fixed_level: float | None = None,
) -> None:
    source = result.absolute if result.absolute is not None else result.corrected
    background_level = fixed_level
    if background_level is None:
        background_level = estimate_constant_fluorescence(result.q, source, q_range)
    result.fluorescence_background = float(background_level)
    result.fluorescence_corrected = source - result.fluorescence_background


def _integrate_image(ai, image: np.ndarray, mask: np.ndarray, npt: int, unit: str) -> tuple[np.ndarray, np.ndarray]:
    if image.shape != mask.shape:
        raise ValueError(f"Image shape {image.shape} does not match mask shape {mask.shape}")
    result = ai.integrate1d(image, npt, mask=mask, unit=unit)
    return np.asarray(result.radial), np.asarray(result.intensity)


def reduce_one_file(
    input_hdf5: Path,
    poni_file: Path,
    mask_file: Path,
    dataset_path: str,
    npt: int,
    unit: str,
    frame: int | None = None,
    background_hdf5: Path | None = None,
    background_scale: float = 1.0,
) -> IntegrationResult:
    """Integrate one source HDF5 file and optionally subtract a background file."""
    ai = load_poni(str(poni_file))
    mask = _load_mask(mask_file)

    _set_ai_wavelength_from_hdf5(ai, input_hdf5)
    sample_image = _read_hdf5_image(input_hdf5, dataset_path, frame)
    q, sample_i = _integrate_image(ai, sample_image, mask, npt, unit)

    background_i = None
    corrected_i = sample_i.copy()
    if background_hdf5 is not None:
        _set_ai_wavelength_from_hdf5(ai, background_hdf5)
        background_image = _read_hdf5_image(background_hdf5, dataset_path, frame)
        background_q, background_i = _integrate_image(ai, background_image, mask, npt, unit)
        if not np.allclose(q, background_q, rtol=1e-6, atol=1e-12):
            raise ValueError("Sample and background q grids do not match.")
        corrected_i = sample_i - background_scale * background_i

    return IntegrationResult(q=q, intensity=sample_i, background=background_i, corrected=corrected_i)


def calibrate_with_glassy_carbon(
    glassy_carbon_hdf5: Path,
    poni_file: Path,
    mask_file: Path,
    dataset_path: str,
    npt: int,
    unit: str,
    q_range: tuple[float, float],
    frame: int | None = None,
    glassy_carbon_background_hdf5: Path | None = None,
    background_scale: float = 1.0,
    reference_file: Path | None = None,
) -> GlassyCarbonCalibration:
    """Calculate an absolute scale factor from measured glassy carbon."""
    gc_result = reduce_one_file(
        input_hdf5=glassy_carbon_hdf5,
        poni_file=poni_file,
        mask_file=mask_file,
        dataset_path=dataset_path,
        npt=npt,
        unit=unit,
        frame=frame,
        background_hdf5=glassy_carbon_background_hdf5,
        background_scale=background_scale,
    )
    reference_q, reference_i = _load_reference_curve(reference_file)
    qmin, qmax = q_range
    if qmin >= qmax:
        raise ValueError("Glassy carbon q-range minimum must be smaller than maximum.")
    mask = (
        np.isfinite(gc_result.q)
        & np.isfinite(gc_result.corrected)
        & (gc_result.q >= qmin)
        & (gc_result.q <= qmax)
    )
    if np.count_nonzero(mask) < 2:
        raise ValueError("Measured glassy carbon curve does not overlap the requested q range.")

    measured_q = gc_result.q[mask]
    measured_i = gc_result.corrected[mask]
    reference_on_measured_q = np.interp(measured_q, reference_q, reference_i)
    measured_area = _integrated_intensity(measured_q, measured_i)
    reference_area = _integrated_intensity(measured_q, reference_on_measured_q)
    if not np.isfinite(measured_area) or measured_area <= 0:
        raise ValueError(
            "Measured glassy carbon calibration area is zero, negative, or invalid. "
            "Check the glassy carbon file, background file, and q calibration range."
        )

    scale_factor = float(reference_area / measured_area)
    gc_result.absolute = gc_result.corrected * scale_factor
    return GlassyCarbonCalibration(
        scale_factor=scale_factor,
        q_range=q_range,
        measured_area=measured_area,
        reference_area=reference_area,
        gc_result=gc_result,
        reference_q=reference_q,
        reference_i=reference_i,
    )


def _write_outputs(
    result: IntegrationResult,
    input_hdf5: Path,
    output_dir: Path,
    args: argparse.Namespace,
    gc_calibration: GlassyCarbonCalibration | None = None,
) -> Path:
    stem = input_hdf5.stem
    output_path = output_dir / f"{stem}_1d.dat"

    columns = [result.q, result.intensity]
    column_names = [args.unit, "I_sample"]
    if result.background is not None:
        columns.append(result.background)
        column_names.append("I_background")
    columns.append(result.corrected)
    column_names.append("I_corrected")
    if result.absolute is not None:
        columns.append(result.absolute)
        column_names.append("I_absolute_gc")
    if result.fluorescence_corrected is not None:
        fluorescence_background = np.full_like(result.q, result.fluorescence_background, dtype=float)
        columns.append(fluorescence_background)
        column_names.append("I_fluorescence_background")
        columns.append(result.fluorescence_corrected)
        column_names.append("I_final")

    metadata = {
        "input_hdf5": str(input_hdf5),
        "background_hdf5": str(args.background_hdf5) if args.background_hdf5 else None,
        "background_scale": args.background_scale,
        "glassy_carbon_hdf5": str(args.glassy_carbon_hdf5) if args.glassy_carbon_hdf5 else None,
        "glassy_carbon_background_hdf5": _optional_path_text(
            args.glassy_carbon_background_hdf5 or args.background_hdf5
        ),
        "gc_reference_file": _optional_path_text(args.gc_reference_file) or "NIST_SRM3600_builtin",
        "gc_scale_factor": gc_calibration.scale_factor if gc_calibration else None,
        "gc_q_range": list(gc_calibration.q_range) if gc_calibration else None,
        "subtract_fluorescence": args.subtract_fluorescence,
        "fluorescence_q_range": list(args.fluorescence_q_range) if args.subtract_fluorescence else None,
        "fluorescence_fixed_level": args.fluorescence_level,
        "fluorescence_background": result.fluorescence_background,
        "fluorescence_source": "I_absolute_gc" if result.absolute is not None else "I_corrected",
        "dataset_path": args.dataset_path,
        "poni": str(args.poni),
        "mask": str(args.mask),
        "npt": args.npt,
        "unit": args.unit,
        "frame": args.frame,
    }
    header = "\n".join(
        [
            "ASWAXS 1D reduction",
            "metadata_json=" + json.dumps(metadata, sort_keys=True),
            "columns=" + " ".join(column_names),
        ]
    )
    np.savetxt(output_path, np.column_stack(columns), header=header)
    np.savez(
        output_dir / f"{stem}_1d.npz",
        q=result.q,
        intensity_sample=result.intensity,
        intensity_background=result.background,
        intensity_corrected=result.corrected,
        intensity_absolute_gc=result.absolute,
        intensity_fluorescence_corrected=result.fluorescence_corrected,
        fluorescence_background=result.fluorescence_background,
        metadata=json.dumps(metadata, sort_keys=True),
    )
    return output_path


def _write_gc_calibration_outputs(
    calibration: GlassyCarbonCalibration,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    output_path = output_dir / "glassy_carbon_calibration.dat"
    reference_interp = np.interp(calibration.gc_result.q, calibration.reference_q, calibration.reference_i)
    columns = [
        calibration.gc_result.q,
        calibration.gc_result.intensity,
        calibration.gc_result.corrected,
        reference_interp,
        calibration.gc_result.absolute,
    ]
    metadata = {
        "glassy_carbon_hdf5": str(args.glassy_carbon_hdf5),
        "glassy_carbon_background_hdf5": _optional_path_text(
            args.glassy_carbon_background_hdf5 or args.background_hdf5
        ),
        "gc_reference_file": _optional_path_text(args.gc_reference_file) or "NIST_SRM3600_builtin",
        "gc_q_range": list(calibration.q_range),
        "gc_scale_factor": calibration.scale_factor,
        "measured_gc_area": calibration.measured_area,
        "reference_gc_area": calibration.reference_area,
        "unit": args.unit,
    }
    header = "\n".join(
        [
            "ASWAXS glassy carbon calibration",
            "metadata_json=" + json.dumps(metadata, sort_keys=True),
            "columns=" + " ".join(["q", "I_gc_sample", "I_gc_corrected", "I_gc_reference", "I_gc_absolute"]),
        ]
    )
    np.savetxt(output_path, np.column_stack(columns), header=header)
    (output_dir / "glassy_carbon_calibration.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reduce ASWAXS HDF5 detector images to masked, azimuthally averaged 1D data."
    )
    parser.add_argument(
        "--input-hdf5",
        nargs="+",
        required=True,
        help="One or more measurement HDF5 files to reduce.",
    )
    parser.add_argument("--poni", required=True, help="Calibration .poni file generated during pyFAI setup.")
    parser.add_argument("--mask", required=True, help="Mask file generated during pyFAI setup (.npy or EDF-readable).")
    parser.add_argument(
        "--background-hdf5",
        help="Optional background/empty-cell HDF5 file. Its 1D curve is subtracted from each sample curve.",
    )
    parser.add_argument(
        "--background-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the background 1D curve before subtraction.",
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help=f"HDF5 dataset containing detector image data. Default: {DEFAULT_DATASET_PATH}",
    )
    parser.add_argument("--npt", type=int, default=DEFAULT_NPT, help="Number of radial bins for azimuthal integration.")
    parser.add_argument("--unit", default="q_A^-1", help="pyFAI radial unit, for example q_A^-1, q_nm^-1, or 2th_deg.")
    parser.add_argument(
        "--glassy-carbon-hdf5",
        help="Optional glassy carbon HDF5 file used to compute an absolute intensity scale factor.",
    )
    parser.add_argument(
        "--glassy-carbon-background-hdf5",
        help="Optional background/air HDF5 file for glassy carbon. If omitted, --background-hdf5 is reused.",
    )
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
        help="q range used to match glassy carbon to the reference curve. Default: 0.03 0.20",
    )
    parser.add_argument(
        "--subtract-fluorescence",
        action="store_true",
        help="Subtract a constant fluorescence background from the final 1D curve.",
    )
    parser.add_argument(
        "--fluorescence-q-range",
        nargs=2,
        type=float,
        default=DEFAULT_FLUORESCENCE_Q_RANGE,
        metavar=("QMIN", "QMAX"),
        help="q range used to estimate constant fluorescence background. Default: 0.16 0.20",
    )
    parser.add_argument(
        "--fluorescence-level",
        type=float,
        help="Fixed fluorescence background level to subtract. If omitted, the level is estimated from --fluorescence-q-range.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Frame index for a 3D HDF5 stack. If omitted, 3D stacks are averaged before integration.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "outputs" / "reduction_output"),
        help="Output directory for reduced 1D data products.",
    )
    parser.add_argument(
        "--analysis-h5",
        help=(
            "Optional analysis HDF5 output path. If omitted, one input writes output-dir/analysis.h5; "
            "multiple inputs write output-dir/<input_stem>_analysis.h5."
        ),
    )
    return parser


def validate_inputs(args: argparse.Namespace) -> ReductionInputs:
    input_hdf5 = [Path(path).expanduser().resolve() for path in args.input_hdf5]
    poni_file = Path(args.poni).expanduser().resolve()
    mask_file = Path(args.mask).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    background_hdf5 = Path(args.background_hdf5).expanduser().resolve() if args.background_hdf5 else None
    glassy_carbon_hdf5 = Path(args.glassy_carbon_hdf5).expanduser().resolve() if args.glassy_carbon_hdf5 else None
    glassy_carbon_background_hdf5 = (
        Path(args.glassy_carbon_background_hdf5).expanduser().resolve()
        if args.glassy_carbon_background_hdf5
        else background_hdf5
    )
    gc_reference_file = Path(args.gc_reference_file).expanduser().resolve() if args.gc_reference_file else None

    paths_to_check = [(path, "input HDF5") for path in input_hdf5]
    paths_to_check.extend([(poni_file, "PONI"), (mask_file, "mask")])
    if background_hdf5 is not None:
        paths_to_check.append((background_hdf5, "background HDF5"))
    if glassy_carbon_hdf5 is not None:
        paths_to_check.append((glassy_carbon_hdf5, "glassy carbon HDF5"))
    if glassy_carbon_background_hdf5 is not None:
        paths_to_check.append((glassy_carbon_background_hdf5, "glassy carbon background HDF5"))
    if gc_reference_file is not None:
        paths_to_check.append((gc_reference_file, "glassy carbon reference file"))

    for path, label in paths_to_check:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} file: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return ReductionInputs(
        input_hdf5=input_hdf5,
        poni_file=poni_file,
        mask_file=mask_file,
        output_dir=output_dir,
        background_hdf5=background_hdf5,
        glassy_carbon_hdf5=glassy_carbon_hdf5,
        glassy_carbon_background_hdf5=glassy_carbon_background_hdf5,
        gc_reference_file=gc_reference_file,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    inputs = validate_inputs(args)

    print("ASWAXS reduction pipeline")
    print(f"PONI file: {inputs.poni_file}")
    print(f"Mask file: {inputs.mask_file}")
    if inputs.background_hdf5:
        print(f"Background HDF5: {inputs.background_hdf5}")
    print(f"Output directory: {inputs.output_dir}")
    gc_calibration = None
    if inputs.glassy_carbon_hdf5:
        gc_calibration = calibrate_with_glassy_carbon(
            glassy_carbon_hdf5=inputs.glassy_carbon_hdf5,
            poni_file=inputs.poni_file,
            mask_file=inputs.mask_file,
            dataset_path=args.dataset_path,
            npt=args.npt,
            unit=args.unit,
            q_range=tuple(args.gc_q_range),
            frame=args.frame,
            glassy_carbon_background_hdf5=inputs.glassy_carbon_background_hdf5,
            background_scale=args.background_scale,
            reference_file=inputs.gc_reference_file,
        )
        gc_output_path = _write_gc_calibration_outputs(gc_calibration, inputs.output_dir, args)
        print(f"Glassy carbon scale factor: {gc_calibration.scale_factor:.8g}")
        print(f"Wrote {gc_output_path}")

    for input_hdf5 in inputs.input_hdf5:
        result = reduce_one_file(
            input_hdf5=input_hdf5,
            poni_file=inputs.poni_file,
            mask_file=inputs.mask_file,
            dataset_path=args.dataset_path,
            npt=args.npt,
            unit=args.unit,
            frame=args.frame,
            background_hdf5=inputs.background_hdf5,
            background_scale=args.background_scale,
        )
        if gc_calibration is not None:
            result.absolute = result.corrected * gc_calibration.scale_factor
        if args.subtract_fluorescence:
            apply_fluorescence_subtraction(
                result,
                q_range=tuple(args.fluorescence_q_range),
                fixed_level=args.fluorescence_level,
            )
        output_path = _write_outputs(result, input_hdf5, inputs.output_dir, args, gc_calibration)
        analysis_path = _analysis_path_for_input(args, inputs.output_dir, input_hdf5, len(inputs.input_hdf5))
        _write_single_file_analysis_h5(
            analysis_path=analysis_path,
            input_hdf5=input_hdf5,
            result=result,
            args=args,
            gc_calibration=gc_calibration,
        )
        print(f"Wrote {output_path}")
        print(f"Wrote analysis HDF5: {analysis_path}")
    return 0


def _analysis_path_for_input(args: argparse.Namespace, output_dir: Path, input_hdf5: Path, input_count: int) -> Path:
    if args.analysis_h5:
        return Path(args.analysis_h5).expanduser().resolve()
    if input_count == 1:
        return output_dir / "analysis.h5"
    return output_dir / f"{input_hdf5.stem}_analysis.h5"


def _write_single_file_analysis_h5(
    analysis_path: Path,
    input_hdf5: Path,
    result: IntegrationResult,
    args: argparse.Namespace,
    gc_calibration: GlassyCarbonCalibration | None,
) -> None:
    """Write the analysis HDF5 companion for the single-file reducer."""
    data_reference_metadata = {
        "data_detector_path": args.dataset_path,
        "source_frame_indices": args.frame if args.frame is not None else "all",
        "source_frame_count": 1,
    }
    create_analysis_h5_from_data(input_hdf5, analysis_path, data_reference_metadata=data_reference_metadata)

    sigma_i = np.full_like(result.corrected, np.nan, dtype=float)
    reduction_metadata = {
        "input_h5_file": str(input_hdf5),
        "input_data_path": args.dataset_path,
        "output_h5_file": str(analysis_path),
        "output_data_path": "/entry/process_01_reduction/data",
        "n_total_frames": 1,
        "n_accepted_frames": 1,
        "n_rejected_frames": 0,
        "notes": "single-file pyFAI reduction; existing text and NPZ outputs are preserved",
    }
    reduction_parameters = {
        "poni_file": str(args.poni),
        "poni_file_hash": file_sha256(args.poni),
        "mask_file": str(args.mask),
        "mask_file_hash": file_sha256(args.mask),
        "q_unit": args.unit,
        "q_min": float(np.nanmin(result.q)),
        "q_max": float(np.nanmax(result.q)),
        "n_q_bins": args.npt,
        "integration_method": "pyFAI.integrate1d",
        "normalization_method": "none",
        "dark_subtraction": False,
        "flatfield_correction": False,
        "solid_angle_correction": "pyFAI_default",
        "polarization_correction": "unknown",
        "error_model": "unknown",
    }
    write_reduction_to_analysis_h5(
        analysis_path,
        input_hdf5,
        result.q,
        result.intensity,
        sigma_i,
        reduction_metadata,
        reduction_parameters,
    )

    if result.background is not None:
        corrected_data = {
            "q": result.q,
            "energy": np.asarray([np.nan]),
            "I_sample_corrected": result.corrected[np.newaxis, :],
            "sigma_sample_corrected": sigma_i[np.newaxis, :],
            "I_gc_corrected": gc_calibration.gc_result.corrected[np.newaxis, :] if gc_calibration else np.asarray([]),
            "sigma_gc_corrected": np.full_like(gc_calibration.gc_result.corrected[np.newaxis, :], np.nan)
            if gc_calibration
            else np.asarray([]),
        }
        subtraction_metadata = {
            "input_h5_file": str(analysis_path),
            "input_data_path": "/entry/process_01_reduction/data",
            "output_h5_file": str(analysis_path),
            "output_data_path": "/entry/process_02_background_subtraction/data",
        }
        subtraction_parameters = {
            "gc_background": "air",
            "sample_background": "empty_cell/solvent",
            "scale_by_I0": False,
            "scale_by_transmission": False,
            "scale_by_exposure_time": False,
            "subtraction_formula": "I_sample_corrected = I_sample - background_scale * I_background",
            "solvent_scale_factor": args.background_scale,
            "empty_cell_scale_factor": args.background_scale,
        }
        subtraction_map = {
            "energy": np.asarray([np.nan]),
            "sample_id": str(input_hdf5),
            "air_id": str(args.background_hdf5) if args.background_hdf5 else "unknown",
            "glassy_carbon_id": str(args.glassy_carbon_hdf5) if args.glassy_carbon_hdf5 else "unknown",
            "empty_cell_id": str(args.background_hdf5) if args.background_hdf5 else "unknown",
            "solvent_id": "unknown",
        }
        write_background_subtraction_to_analysis_h5(
            analysis_path,
            corrected_data,
            subtraction_metadata,
            subtraction_parameters,
            subtraction_map,
        )

    if result.absolute is not None and gc_calibration is not None:
        normalized_data = {
            "q": result.q,
            "energy": np.asarray([np.nan]),
            "I_sample_normalized": result.absolute[np.newaxis, :],
            "sigma_sample_normalized": sigma_i[np.newaxis, :],
            "I_gc_normalized": gc_calibration.gc_result.absolute[np.newaxis, :],
            "sigma_gc_normalized": np.full_like(gc_calibration.gc_result.absolute[np.newaxis, :], np.nan),
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
            "q_range_used": list(gc_calibration.q_range),
            "scale_method": "integrated_area_ratio",
            "absolute_scale": True,
            "uncertainty_propagation": "scale uncertainty not propagated",
        }
        normalization_factors = {
            "energy": np.asarray([np.nan]),
            "scale_factor": np.asarray([gc_calibration.scale_factor]),
            "scale_uncertainty": np.asarray([np.nan]),
            "q_min_used": np.asarray([gc_calibration.q_range[0]]),
            "q_max_used": np.asarray([gc_calibration.q_range[1]]),
            "scale_factor_basis": "glassy_carbon_reference_area / measured_area",
        }
        write_glassy_carbon_normalization_to_analysis_h5(
            analysis_path,
            normalized_data,
            normalization_metadata,
            normalization_parameters,
            normalization_factors,
        )


if __name__ == "__main__":
    raise SystemExit(main())

