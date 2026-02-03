#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from nptdms import TdmsFile


TDMS_TIME_PATTERN = re.compile(r"_UTC_(\d{8}_\d{6}\.\d+)\.tdms$", re.IGNORECASE)
DATETIME_FORMATS = (
    "%Y-%m-%d  %H:%M:%S",
    "%Y/%m/%d  %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d  %H:%M:%S.%f",
    "%Y/%m/%d  %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S.%f",
)


@dataclass(frozen=True)
class TdmsIndex:
    path: Path
    start_utc: datetime
    nominal_end_utc: datetime


@dataclass(frozen=True)
class ExtractionTask:
    name: str
    csv_path: Path
    start_local: datetime
    end_local: datetime
    start_utc: datetime
    end_utc: datetime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Use Airtag CSV start/end time to cut name-matched TDMS DAS signals into per-name CSV files."
    )
    p.add_argument("--video-dir", required=True, help="Directory with *.mp4/*.MP4 files (names are used as targets).")
    p.add_argument("--airtag-csv-dir", required=True, help="Directory with Airtag CSV files (same names as videos).")
    p.add_argument("--tdms-dir", required=True, help="Directory with DAS TDMS files.")
    p.add_argument("--output-dir", default="output/name_signals", help="Output directory for extracted CSV files.")
    p.add_argument("--fs", type=float, default=2000.0, help="DAS sample rate in Hz. Default: 2000.")
    p.add_argument(
        "--csv-utc-offset-hours",
        type=float,
        default=8.0,
        help="CSV datetime timezone offset against UTC. Default 8 (Asia/Shanghai).",
    )
    p.add_argument(
        "--name",
        action="append",
        default=None,
        help="Only process this name (repeatable, case-insensitive).",
    )
    p.add_argument("--encoding", default="utf-8-sig", help="Encoding for Airtag CSV files. Default utf-8-sig.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSV.")
    p.add_argument("--dry-run", action="store_true", help="Only print matching and time windows; do not write output.")
    p.add_argument("--use-airtag-only", action="store_true",
                   help="Use Airtag CSV filenames as targets (do not require Video files). Useful for inference without MP4 files.")
    p.add_argument(
        "--skip-channels",
        type=int,
        default=15,
        help="Number of channels to skip from the beginning (default: 15). Set to 0 to keep all channels.",
    )
    return p.parse_args()


def parse_datetime(text: str) -> datetime:
    s = text.strip().replace("T", " ")
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported datetime format: {text!r}")


def read_csv_start_end(csv_path: Path, encoding: str) -> Tuple[datetime, datetime]:
    first_dt: Optional[datetime] = None
    last_dt: Optional[datetime] = None

    with csv_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Empty CSV: {csv_path}")
        if not str(header[0]).strip().lower().startswith("datetime"):
            raise ValueError(f"First column is not datetime in {csv_path}: {header[0]!r}")

        for row in reader:
            if not row:
                continue
            raw = row[0].strip()
            if not raw:
                continue
            dt = parse_datetime(raw)
            if first_dt is None:
                first_dt = dt
            last_dt = dt

    if first_dt is None or last_dt is None:
        raise ValueError(f"No datetime rows found in {csv_path}")
    return first_dt, last_dt


def get_target_names(video_dir: Path, name_filters: Optional[Sequence[str]]) -> List[str]:
    stems = {}
    for p in video_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".mp4":
            stems[p.stem.lower()] = p.stem

    names = sorted(stems.keys())
    if name_filters:
        wanted = {n.lower().strip() for n in name_filters if n.strip()}
        names = [n for n in names if n in wanted]
    return names


def get_target_names_from_airtag(csv_dir: Path, name_filters: Optional[Sequence[str]]) -> List[str]:
    """Return target names derived from CSV filenames in airtag directory."""
    stems = {}
    for p in csv_dir.glob("*.csv"):
        stems[p.stem.lower()] = p.stem

    names = sorted(stems.keys())
    if name_filters:
        wanted = {n.lower().strip() for n in name_filters if n.strip()}
        names = [n for n in names if n in wanted]
    return names


def build_tasks(
    target_names: Sequence[str], csv_dir: Path, encoding: str, offset_hours: float
) -> Tuple[List[ExtractionTask], List[str]]:
    csv_map = {p.stem.lower(): p for p in csv_dir.glob("*.csv")}
    tasks: List[ExtractionTask] = []
    missing: List[str] = []
    offset = timedelta(hours=offset_hours)

    for name in target_names:
        csv_path = csv_map.get(name)
        if csv_path is None:
            missing.append(name)
            continue

        start_local, end_local = read_csv_start_end(csv_path, encoding=encoding)
        start_utc = start_local - offset
        end_utc = end_local - offset
        if end_utc < start_utc:
            raise ValueError(f"End earlier than start in {csv_path}")

        tasks.append(
            ExtractionTask(
                name=name,
                csv_path=csv_path,
                start_local=start_local,
                end_local=end_local,
                start_utc=start_utc,
                end_utc=end_utc,
            )
        )

    return tasks, missing


def parse_tdms_start(path: Path) -> datetime:
    m = TDMS_TIME_PATTERN.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse UTC timestamp from TDMS name: {path.name}")
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S.%f")


def build_tdms_index(tdms_dir: Path, default_duration_sec: float = 60.0) -> List[TdmsIndex]:
    tdms_files = sorted(tdms_dir.glob("*.tdms"), key=lambda p: p.name)
    if not tdms_files:
        raise ValueError(f"No TDMS files found in {tdms_dir}")

    starts = [(p, parse_tdms_start(p)) for p in tdms_files]
    starts.sort(key=lambda x: x[1])

    index: List[TdmsIndex] = []
    for i, (path, start) in enumerate(starts):
        if i + 1 < len(starts):
            nominal_end = starts[i + 1][1]
        else:
            nominal_end = start + timedelta(seconds=default_duration_sec)
        index.append(TdmsIndex(path=path, start_utc=start, nominal_end_utc=nominal_end))
    return index


def choose_overlapping_files(index: Sequence[TdmsIndex], start_utc: datetime, end_utc: datetime) -> List[TdmsIndex]:
    out = []
    for item in index:
        if item.start_utc <= end_utc and item.nominal_end_utc >= start_utc:
            out.append(item)
    return out


def list_channels(tdms_file: TdmsFile):
    channels = []
    for group in tdms_file.groups():
        for channel in group.channels():
            channels.append(channel)
    return channels


def extraction_indices(
    file_start_utc: datetime,
    fs: float,
    n_samples: int,
    seg_start_utc: datetime,
    seg_end_utc: datetime,
) -> Tuple[int, int]:
    dt0 = (seg_start_utc - file_start_utc).total_seconds()
    dt1 = (seg_end_utc - file_start_utc).total_seconds()

    start_idx = int(math.ceil(dt0 * fs - 1e-9))
    end_exclusive = int(math.floor(dt1 * fs + 1e-9)) + 1

    if start_idx < 0:
        start_idx = 0
    if end_exclusive > n_samples:
        end_exclusive = n_samples
    if end_exclusive < start_idx:
        end_exclusive = start_idx

    return start_idx, end_exclusive


def write_array_csv(path: Path, data: np.ndarray, write_header: bool, n_channels: int, append: bool, channel_offset: int = 0) -> None:
    """Write data to CSV with optional channel offset for naming.
    
    Args:
        path: Output file path
        data: Data array to write
        write_header: Whether to write header row
        n_channels: Number of channels in data
        append: Whether to append to existing file
        channel_offset: Starting channel number for header names (default 0)
    """
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        if write_header:
            header = ",".join([f"ch_{i + channel_offset}" for i in range(n_channels)])
            f.write(header + "\n")
        if data.size > 0:
            if np.issubdtype(data.dtype, np.integer):
                np.savetxt(f, data, delimiter=",", fmt="%d")
            else:
                np.savetxt(f, data, delimiter=",", fmt="%.10g")


def extract_one_task(
    task: ExtractionTask,
    tdms_index: Sequence[TdmsIndex],
    output_dir: Path,
    fs: float,
    overwrite: bool,
    dry_run: bool,
    skip_channels: int = 0,
) -> Tuple[int, int, int]:
    """Extract DAS signals for one task.
    
    Args:
        task: Extraction task with time range and output info
        tdms_index: Index of TDMS files
        output_dir: Output directory for CSV
        fs: Sample rate in Hz
        overwrite: Whether to overwrite existing output
        dry_run: Only print plan, don't write
        skip_channels: Number of channels to skip from the beginning (default 0)
    
    Returns:
        Tuple of (used_files, total_rows, output_channels)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{task.name}.csv"
    overlaps = choose_overlapping_files(tdms_index, task.start_utc, task.end_utc)

    print(
        f"[Task] {task.name}: {task.start_local} -> {task.end_local} (UTC {task.start_utc} -> {task.end_utc}), "
        f"matched TDMS files: {len(overlaps)}"
    )

    if not overlaps:
        return (0, 0, 0)

    if out_csv.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output exists, use --overwrite: {out_csv}")

    total_rows = 0
    used_files = 0
    n_channels_ref: Optional[int] = None

    if not dry_run and out_csv.exists() and overwrite:
        out_csv.unlink()

    has_written = False

    for item in overlaps:
        tdms = TdmsFile.read(item.path)
        channels = list_channels(tdms)
        if not channels:
            print(f"  [Warn] No channels in {item.path.name}, skipped")
            continue

        n_channels = len(channels)
        if n_channels_ref is None:
            n_channels_ref = n_channels
        elif n_channels != n_channels_ref:
            raise ValueError(
                f"Channel count mismatch for {task.name}: expected {n_channels_ref}, got {n_channels} in {item.path.name}"
            )

        n_samples = len(channels[0])
        start_idx, end_exclusive = extraction_indices(
            file_start_utc=item.start_utc,
            fs=fs,
            n_samples=n_samples,
            seg_start_utc=task.start_utc,
            seg_end_utc=task.end_utc,
        )
        row_count = end_exclusive - start_idx
        if row_count <= 0:
            continue

        used_files += 1
        total_rows += row_count

        print(
            f"  [Use] {item.path.name}: samples [{start_idx}, {end_exclusive}) => {row_count} rows"
        )

        if dry_run:
            continue

        # Skip the first skip_channels channels
        channels_to_use = channels[skip_channels:] if skip_channels > 0 else channels
        if not channels_to_use:
            print(f"  [Warn] All channels skipped in {item.path.name}")
            continue
        
        cols = [np.asarray(ch[start_idx:end_exclusive]) for ch in channels_to_use]
        data = np.column_stack(cols)
        write_array_csv(
            path=out_csv,
            data=data,
            write_header=not has_written,
            n_channels=len(channels_to_use),
            append=has_written,
            channel_offset=skip_channels,
        )
        has_written = True

    # Report actual output channel count
    output_channels = (n_channels_ref - skip_channels) if n_channels_ref else 0
    if output_channels < 0:
        output_channels = 0

    if dry_run:
        return (used_files, total_rows, output_channels)

    if used_files == 0:
        print(f"  [Warn] No overlapping samples written for {task.name}")
    else:
        print(f"  [Done] {out_csv} | rows={total_rows}, channels={output_channels} (skipped first {skip_channels})")

    return (used_files, total_rows, output_channels)


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir)
    csv_dir = Path(args.airtag_csv_dir)
    tdms_dir = Path(args.tdms_dir)
    output_dir = Path(args.output_dir)

    if not video_dir.exists():
        raise FileNotFoundError(f"video-dir not found: {video_dir}")
    if not csv_dir.exists():
        raise FileNotFoundError(f"airtag-csv-dir not found: {csv_dir}")
    if not tdms_dir.exists():
        raise FileNotFoundError(f"tdms-dir not found: {tdms_dir}")

    if args.use_airtag_only:
        target_names = get_target_names_from_airtag(csv_dir, args.name)
    else:
        target_names = get_target_names(video_dir, args.name)

    if not target_names:
        raise ValueError("No matching names found (check Video files or Airtag CSVs depending on mode).")

    tasks, missing = build_tasks(target_names, csv_dir, args.encoding, args.csv_utc_offset_hours)
    if missing:
        print("[Warn] missing same-name csv:", ", ".join(missing))
    if not tasks:
        raise ValueError("No task can be built from videos + csv files.")

    tdms_index = build_tdms_index(tdms_dir)

    skip_channels = args.skip_channels
    print(f"[Info] targets={len(tasks)}, tdms_files={len(tdms_index)}, fs={args.fs} Hz, skip_channels={skip_channels}")

    total_written = 0
    for task in tasks:
        _, rows, _ = extract_one_task(
            task=task,
            tdms_index=tdms_index,
            output_dir=output_dir,
            fs=args.fs,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            skip_channels=skip_channels,
        )
        total_written += rows

    if args.dry_run:
        print(f"[DryRun] total planned rows: {total_written}")
    else:
        print(f"[Finish] total rows written: {total_written}")


if __name__ == "__main__":
    main()
