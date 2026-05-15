"""Create and validate ASAXS sequence manifests.

The manifest is the bridge between acquisition order and scientific meaning:
each sorted HDF5 file is assigned an energy index, group index, and repeated
frame index before the reduction pipeline starts.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SequenceItem:
    energy_index: int
    group_index: int
    frame_index: int
    sequence_index: int
    path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and map an ASWAXS energy/group/frame HDF5 sequence before reduction."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing the continuous HDF5 sequence.")
    parser.add_argument("--pattern", default="*.h5", help="HDF5 filename pattern. Default: *.h5")
    parser.add_argument("--num-energies", type=int, required=True, help="Number of energies in the sequence.")
    parser.add_argument("--num-groups", type=int, required=True, help="Number of groups per energy.")
    parser.add_argument("--num-frames", type=int, required=True, help="Number of repeated frames per group.")
    parser.add_argument(
        "--skip-files",
        type=int,
        default=0,
        help="Number of leading files to ignore before sequence mapping.",
    )
    parser.add_argument(
        "--skip-sequence-indices",
        nargs="*",
        type=int,
        default=[],
        help=(
            "One-based file positions to remove after sorting and leading skip. "
            "Use this for known beamdown/interrupted measurements that were repeated."
        ),
    )
    parser.add_argument(
        "--allow-extra-files",
        action="store_true",
        help="Allow more files than expected and use only the first expected sequence files after skipping.",
    )
    parser.add_argument(
        "--resume-mode",
        choices=("strict", "first", "last"),
        default="strict",
        help=(
            "How to handle extra files. strict stops on any count mismatch; "
            "first uses the first complete sequence; last uses the last complete sequence, useful after beamdown repeats."
        ),
    )
    parser.add_argument(
        "--manifest",
        default="sequence_manifest.csv",
        help="CSV manifest path to write. Default: sequence_manifest.csv",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not ask for beamdown indices interactively; fail immediately on count mismatch.",
    )
    return parser


def expected_count(num_energies: int, num_groups: int, num_frames: int) -> int:
    for value, label in (
        (num_energies, "num_energies"),
        (num_groups, "num_groups"),
        (num_frames, "num_frames"),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive.")
    return num_energies * num_groups * num_frames


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def collect_files(data_dir: Path, pattern: str, skip_files: int) -> list[Path]:
    if skip_files < 0:
        raise ValueError("skip_files must be zero or positive.")
    files = sorted(data_dir.glob(pattern), key=natural_sort_key)
    if skip_files:
        files = files[skip_files:]
    return [path.resolve() for path in files]


def remove_sequence_indices(files: list[Path], skip_indices: list[int]) -> list[Path]:
    if not skip_indices:
        return files
    total = len(files)
    bad = sorted(set(skip_indices))
    for index in bad:
        if index <= 0 or index > total:
            raise ValueError(f"skip sequence index {index} is outside available file range 1..{total}.")
    skip_set = set(bad)
    return [path for index, path in enumerate(files, start=1) if index not in skip_set]


def validate_sequence_count(
    files: list[Path],
    expected: int,
    allow_extra_files: bool,
    resume_mode: str,
) -> list[Path]:
    actual = len(files)
    if actual == expected:
        return files
    if actual > expected and (allow_extra_files or resume_mode == "first"):
        return files[:expected]
    if actual > expected and resume_mode == "last":
        return files[-expected:]
    raise ValueError(
        f"Sequence file count mismatch: expected {expected} files "
        f"(num_energies * num_groups * num_frames), found {actual}."
    )


def parse_skip_indices(text: str) -> list[int]:
    normalized = text.replace(",", " ").replace(";", " ")
    indices = []
    for token in normalized.split():
        indices.append(int(token))
    return indices


def prompt_for_beamdown_indices(actual: int, expected: int) -> list[int] | None:
    print("")
    print("Sequence file count mismatch.")
    print(f"Expected files: {expected}")
    print(f"Actual files after current skips: {actual}")
    if actual < expected:
        print("There are fewer files than expected. Skipping beamdown files cannot fix this.")
        return None
    extra = actual - expected
    print(f"Extra files: {extra}")
    print("Enter one-based beamdown file numbers to skip, separated by spaces or commas.")
    print("Example: 137 284")
    response = input("Beamdown indices, or q to quit: ").strip()
    if response.lower() in {"q", "quit", "exit", ""}:
        return None
    return parse_skip_indices(response)


def resolve_sequence_files(
    raw_files: list[Path],
    expected: int,
    initial_skip_indices: list[int],
    allow_extra_files: bool,
    resume_mode: str,
    no_prompt: bool,
) -> tuple[list[Path], list[int]]:
    """Apply skip/resume rules and return the exact files used for reduction."""
    skip_indices = sorted(set(initial_skip_indices))
    while True:
        files = remove_sequence_indices(raw_files, skip_indices)
        try:
            sequence_files = validate_sequence_count(files, expected, allow_extra_files, resume_mode)
        except ValueError:
            if no_prompt:
                raise
            prompted = prompt_for_beamdown_indices(len(files), expected)
            if prompted is None:
                raise SystemExit("Sequence validation stopped before reduction.")
            skip_indices = sorted(set(skip_indices + prompted))
            continue
        return sequence_files, skip_indices


def build_sequence_map(files: list[Path], num_energies: int, num_groups: int, num_frames: int) -> list[SequenceItem]:
    """Map sorted files into energy -> group -> frame acquisition order."""
    items: list[SequenceItem] = []
    sequence_index = 0
    for energy_index in range(1, num_energies + 1):
        for group_index in range(1, num_groups + 1):
            for frame_index in range(1, num_frames + 1):
                items.append(
                    SequenceItem(
                        energy_index=energy_index,
                        group_index=group_index,
                        frame_index=frame_index,
                        sequence_index=sequence_index + 1,
                        path=files[sequence_index],
                    )
                )
                sequence_index += 1
    return items


def write_manifest(items: list[SequenceItem], output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence_index", "energy_index", "group_index", "frame_index", "hdf5_path"])
        for item in items:
            writer.writerow([item.sequence_index, item.energy_index, item.group_index, item.frame_index, item.path])
    return output_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

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
    items = build_sequence_map(sequence_files, args.num_energies, args.num_groups, args.num_frames)
    manifest = write_manifest(items, Path(args.manifest))

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
