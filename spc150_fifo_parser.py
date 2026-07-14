#!/usr/bin/env python3
"""SPC-150 FIFO parser for Becker & Hickl .spc files.

This parser targets the 32-bit FIFO record layout used by SPC-13x/15x-style
streams and keeps marker events as first-class outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


OVERFLOW_COUNT_MASK = np.uint32(0x3FFFFFFF)
INVALID_FLAG = np.uint32(0x80000000)
MACROTIME_OVERFLOW_FLAG = np.uint32(0x40000000)
GAP_FLAG = np.uint32(0x20000000)
MARKER_FLAG = np.uint32(0x10000000)
MACROTIME_WRAP_TICKS = 4096


@dataclass(frozen=True)
class MarkerEvent:
    record_index: int
    word_index: int
    word_hex: str
    aux: int
    detector_mask: int
    macrotime_ticks: int
    nanotime_bin: int
    marker0: bool
    marker1: bool
    marker2: bool
    marker3: bool


@dataclass(frozen=True)
class SetMetadata:
    set_file: str
    module: str | None
    serial_number: str | None
    identification: dict[str, str]
    parameter_groups: dict[str, dict[str, Any]]

    @property
    def spc_params(self) -> dict[str, Any]:
        return self.parameter_groups.get("SP", {})


@dataclass(frozen=True)
class ParseSummary:
    spc_file: str
    set_file: str | None
    total_words: int
    header_word_hex: str
    data_records: int
    overflow_records: int
    overflow_wraps: int
    marker_records: int
    marker_counts_by_bit: dict[str, int]
    marker2_pulse_count: int
    marker2_intervals_ticks: list[int]
    marker2_interval_stats_ticks: dict[str, float] | None
    marker2_interval_stats_seconds: dict[str, float] | None
    sync_rate_hz: float | None
    sync_rate_source: str | None
    set_module: str | None
    set_serial_number: str | None
    set_identification: dict[str, str]
    set_parameter_groups: dict[str, dict[str, Any]]
    set_params: dict[str, Any]


def build_unified_event_dataframe(
    arrays: dict[str, np.ndarray],
    sync_rate_hz: float,
):
    """Return a unified photon/marker event DataFrame.

    Output rows exclude internal overflow-only records.
    """
    if sync_rate_hz <= 0:
        raise ValueError("sync_rate_hz must be > 0 to compute macrotime_s")

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for unified DataFrame export") from exc

    is_marker_all = arrays["is_marker"]
    is_photon_all = arrays.get("is_photon")
    if is_photon_all is None:
        is_photon_all = ~(arrays["is_overflow"] | is_marker_all)
    event_mask = is_photon_all | is_marker_all
    event_index = np.flatnonzero(event_mask).astype(np.int64)
    is_marker = arrays["is_marker"][event_mask]
    detector_raw = arrays["detector"][event_mask].astype(np.int16)
    nanotime = arrays["nanotime_bin"][event_mask].astype(np.int32)
    macrotime_ticks = arrays["macrotime_ticks"][event_mask].astype(np.int64)
    macrotime_s = macrotime_ticks / float(sync_rate_hz)

    event_type = np.where(is_marker, "marker", "photon")
    detector_channel = pd.Series(detector_raw, dtype="Int64")
    detector_channel[is_marker] = pd.NA
    detector_channel_1based = pd.Series(detector_raw + 1, dtype="Int64")
    detector_channel_1based[is_marker] = pd.NA
    nanotime_bin = pd.Series(nanotime, dtype="Int64")
    nanotime_bin[is_marker] = pd.NA
    marker_mask = pd.Series(detector_raw, dtype="Int64")
    marker_mask[~is_marker] = pd.NA

    # Single-bit marker id (0..3) when exactly one marker bit is set, else NA.
    marker_id = pd.Series([pd.NA] * event_index.shape[0], dtype="Int64")
    marker0 = pd.Series([pd.NA] * event_index.shape[0], dtype="boolean")
    marker1 = pd.Series([pd.NA] * event_index.shape[0], dtype="boolean")
    marker2 = pd.Series([pd.NA] * event_index.shape[0], dtype="boolean")
    marker3 = pd.Series([pd.NA] * event_index.shape[0], dtype="boolean")
    marker_rows = np.flatnonzero(is_marker)
    for row in marker_rows.tolist():
        mask = int(detector_raw[row])
        marker0.iloc[row] = bool(mask & 0x1)
        marker1.iloc[row] = bool(mask & 0x2)
        marker2.iloc[row] = bool(mask & 0x4)
        marker3.iloc[row] = bool(mask & 0x8)
        if mask in (1, 2, 4, 8):
            marker_id.iloc[row] = mask.bit_length() - 1

    df = pd.DataFrame(
        {
            "record_index": event_index,
            "word_index": event_index + 1,
            "type": event_type,
            "macrotime_ticks": macrotime_ticks,
            "macrotime_s": macrotime_s,
            "detector_channel": detector_channel,
            "detector_channel_1based": detector_channel_1based,
            "nanotime_bin": nanotime_bin,
            "marker_mask": marker_mask,
            "marker_id": marker_id,
            "marker0": marker0,
            "marker1": marker1,
            "marker2": marker2,
            "marker3": marker3,
            "aux": arrays["aux"][event_mask].astype(np.int16),
        }
    )
    return df


def assign_photons_to_frames(
    events_df,
    marker_id: int = 2,
    frame_start_index: int = 0,
    keep_partial_last_frame: bool = True,
    marker_debounce_ticks: int | None = None,
):
    """Assign photon rows to camera-frame windows delimited by marker events.

    Frame i covers photon events between marker_i and marker_(i+1) in record order.
    Marker records in a burst are treated as one marker pulse. By default, a
    clear bimodal split in the marker-to-marker gaps is detected automatically.
    Supply a non-negative tick threshold to override this behavior; 0 retains
    every marker record.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for frame assignment") from exc

    df = events_df.copy()
    row_count = df.shape[0]
    if row_count == 0:
        return df, pd.DataFrame()
    if marker_debounce_ticks is not None and marker_debounce_ticks < 0:
        raise ValueError("marker_debounce_ticks must be >= 0")

    record_index = df["record_index"].to_numpy(dtype=np.int64)
    event_type = df["type"].to_numpy(dtype=object)
    marker_id_values = pd.to_numeric(df["marker_id"], errors="coerce").to_numpy(dtype=np.float64)
    is_frame_marker = (event_type == "marker") & (marker_id_values == float(marker_id))
    marker_bit_col = f"marker{marker_id}"
    if (not np.any(is_frame_marker)) and (marker_bit_col in df.columns):
        marker_bit_values = df[marker_bit_col].fillna(False).to_numpy(dtype=bool)
        is_frame_marker = (event_type == "marker") & marker_bit_values

    marker_rows = df.loc[is_frame_marker, ["record_index", "macrotime_ticks", "macrotime_s"]].sort_values(
        "record_index"
    )
    if marker_rows.empty:
        raise ValueError(f"No marker events found for marker_id={marker_id}")

    if marker_rows.shape[0] > 1:
        marker_ticks = marker_rows["macrotime_ticks"].to_numpy(dtype=np.int64)
        pulse_start = _marker_pulse_start_mask(marker_ticks, marker_debounce_ticks)
        marker_rows = marker_rows.iloc[np.flatnonzero(pulse_start)].copy()

    frame_table = marker_rows.reset_index(drop=True).copy()
    frame_table["frame_index"] = np.arange(frame_table.shape[0], dtype=np.int64) + int(frame_start_index)
    frame_table["next_record_index"] = frame_table["record_index"].shift(-1)
    frame_table["frame_end_ticks"] = frame_table["macrotime_ticks"].shift(-1)
    frame_table["frame_end_s"] = frame_table["macrotime_s"].shift(-1)
    frame_table = frame_table.rename(
        columns={
            "record_index": "frame_marker_record_index",
            "macrotime_ticks": "frame_start_ticks",
            "macrotime_s": "frame_start_s",
        }
    )

    marker_starts = frame_table["frame_marker_record_index"].to_numpy(dtype=np.int64)
    frame_position = np.searchsorted(marker_starts, record_index, side="right") - 1
    valid = frame_position >= 0
    if not keep_partial_last_frame:
        valid &= frame_position < (marker_starts.size - 1)

    is_photon = event_type == "photon"
    assign_mask = is_photon & valid
    assigned_rows = np.flatnonzero(assign_mask)

    frame_index_values = np.full(row_count, np.nan, dtype=np.float64)
    frame_start_s_values = np.full(row_count, np.nan, dtype=np.float64)
    frame_end_s_values = np.full(row_count, np.nan, dtype=np.float64)
    frame_start_tick_values = np.full(row_count, np.nan, dtype=np.float64)
    frame_end_tick_values = np.full(row_count, np.nan, dtype=np.float64)
    frame_marker_record_values = np.full(row_count, np.nan, dtype=np.float64)
    next_marker_record_values = np.full(row_count, np.nan, dtype=np.float64)

    if assigned_rows.size > 0:
        assigned_pos = frame_position[assigned_rows]
        frame_idx_lookup = frame_table["frame_index"].to_numpy(dtype=np.int64)
        start_s_lookup = frame_table["frame_start_s"].to_numpy(dtype=np.float64)
        end_s_lookup = frame_table["frame_end_s"].to_numpy(dtype=np.float64)
        start_tick_lookup = frame_table["frame_start_ticks"].to_numpy(dtype=np.int64)
        end_tick_lookup = frame_table["frame_end_ticks"].to_numpy(dtype=np.float64)
        start_record_lookup = frame_table["frame_marker_record_index"].to_numpy(dtype=np.int64)
        end_record_lookup = frame_table["next_record_index"].to_numpy(dtype=np.float64)

        frame_index_values[assigned_rows] = frame_idx_lookup[assigned_pos]
        frame_start_s_values[assigned_rows] = start_s_lookup[assigned_pos]
        frame_end_s_values[assigned_rows] = end_s_lookup[assigned_pos]
        frame_start_tick_values[assigned_rows] = start_tick_lookup[assigned_pos]
        frame_end_tick_values[assigned_rows] = end_tick_lookup[assigned_pos]
        frame_marker_record_values[assigned_rows] = start_record_lookup[assigned_pos]
        next_marker_record_values[assigned_rows] = end_record_lookup[assigned_pos]

    df["frame_index"] = pd.Series(frame_index_values, dtype="Int64")
    df["frame_start_s"] = frame_start_s_values
    df["frame_end_s"] = frame_end_s_values
    df["frame_start_ticks"] = pd.Series(frame_start_tick_values, dtype="Int64")
    df["frame_end_ticks"] = pd.Series(frame_end_tick_values, dtype="Int64")
    df["frame_marker_record_index"] = pd.Series(frame_marker_record_values, dtype="Int64")
    df["next_frame_marker_record_index"] = pd.Series(next_marker_record_values, dtype="Int64")

    photon_counts = (
        df.loc[df["type"] == "photon", "frame_index"].dropna().astype("int64").value_counts().to_dict()
    )
    frame_table["photon_count"] = frame_table["frame_index"].map(photon_counts).fillna(0).astype(np.int64)
    if not keep_partial_last_frame and frame_table.shape[0] > 1:
        frame_table = frame_table.iloc[:-1].copy()

    return df, frame_table


def _marker_pulse_start_mask(
    marker_ticks: np.ndarray,
    debounce_ticks: int | None = None,
) -> np.ndarray:
    """Return a mask selecting the first marker record in each pulse burst."""
    marker_ticks = np.asarray(marker_ticks, dtype=np.int64)
    if marker_ticks.size == 0:
        return np.array([], dtype=bool)
    if marker_ticks.size == 1:
        return np.array([True], dtype=bool)

    gaps = np.diff(marker_ticks)
    threshold = debounce_ticks
    if threshold is None:
        positive_gaps = np.unique(gaps[gaps > 0])
        threshold = 0
        if positive_gaps.size > 1:
            ratios = positive_gaps[1:] / positive_gaps[:-1]
            clear_splits = np.flatnonzero(ratios >= 10.0)
            # Use the highest clear split so occasional long gaps inside a
            # pulse remain grouped with the short-gap population.
            if clear_splits.size > 0:
                split = int(clear_splits[-1])
                threshold = int(
                    np.sqrt(float(positive_gaps[split]) * float(positive_gaps[split + 1]))
                )

    return np.concatenate(([True], gaps > int(threshold)))


def _parse_set_value(value_type: str, value: str) -> Any:
    value = value.strip()
    if value_type in {"I", "L", "U", "B"}:
        try:
            return int(value)
        except ValueError:
            return value
    if value_type == "F":
        try:
            return float(value)
        except ValueError:
            return value
    return value


def infer_sync_rate_hz(set_params: dict[str, Any]) -> float | None:
    """Infer sync/macro clock rate from parsed .set parameters.

    For SPC-150 FIFO runs, SP_TAC_R is the configured TAC range (seconds).
    Using 1 / SP_TAC_R gives the macrotime tick rate used for absolute time.
    """
    tac_range_s = set_params.get("SP_TAC_R")
    if isinstance(tac_range_s, (int, float)) and float(tac_range_s) > 0:
        return 1.0 / float(tac_range_s)
    return None


def infer_sync_rate_hz_from_header(header_word: int | None) -> float | None:
    """Read the macrotime clock from a Becker & Hickl .spc header.

    SPC macrotime clocks are stored in the low 24 bits in 0.1 ns units.
    """
    if header_word is None:
        return None
    clock_units_100ps = int(header_word) & 0x00FFFFFF
    if clock_units_100ps <= 0:
        return None
    return 1.0 / (clock_units_100ps * 1e-10)


def resolve_sync_rate_hz(
    sync_rate_hz_override: float | None,
    set_params: dict[str, Any],
    header_word: int | None = None,
) -> tuple[float | None, str | None]:
    if sync_rate_hz_override is not None:
        if sync_rate_hz_override <= 0:
            raise ValueError("sync_rate_hz override must be > 0")
        return float(sync_rate_hz_override), "cli"
    inferred_header = infer_sync_rate_hz_from_header(header_word)
    if inferred_header is not None:
        return inferred_header, "spc-header"
    inferred = infer_sync_rate_hz(set_params)
    if inferred is not None:
        return inferred, "set:SP_TAC_R"
    return None, None


def parse_set_metadata(path: Path) -> SetMetadata:
    """Parse run identity and all typed parameter groups from a SET file."""
    raw = path.read_bytes()
    text = raw.decode("latin1", errors="ignore")

    module_match = re.search(
        r"with module\s+(SPC-\d+)(?:\s+\(Ser\.No\.\s*([^)]+)\))?", text
    )
    module_name = module_match.group(1) if module_match else None
    serial_number = module_match.group(2).strip() if module_match and module_match.group(2) else None

    identification: dict[str, str] = {}
    identification_match = re.search(r"\*IDENTIFICATION(.*?)\*END", text, flags=re.DOTALL)
    if identification_match:
        for line in identification_match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            if key:
                identification[key] = re.sub(r"[\x00-\x1f\x7f]", "", value).strip()

    parameter_groups: dict[str, dict[str, Any]] = {}
    parameter_pattern = r"#([A-Z0-9]+) \[([A-Z0-9_]+),([A-Z]),([^\]]*)\]"
    for group, key, value_type, value in re.findall(parameter_pattern, text):
        parameter_groups.setdefault(group, {})[key] = _parse_set_value(value_type, value)

    return SetMetadata(
        set_file=str(path),
        module=module_name,
        serial_number=serial_number,
        identification=identification,
        parameter_groups=parameter_groups,
    )


def parse_set_file(path: Path | None) -> tuple[str | None, dict[str, Any]]:
    """Backward-compatible view returning the module and all #SP parameters."""
    if path is None:
        return None, {}
    metadata = parse_set_metadata(path)
    return metadata.module, metadata.spc_params


def require_set_metadata(path: Path) -> SetMetadata:
    """Load and validate the required acquisition metadata for an SPC run."""
    if path.suffix.lower() != ".set":
        raise ValueError(f"Expected a .set metadata file, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"SET metadata file not found: {path}")

    metadata = parse_set_metadata(path)
    if metadata.module is None:
        raise ValueError(f"Could not identify an SPC module in SET metadata: {path}")
    if not metadata.spc_params:
        raise ValueError(f"No recognized SPC acquisition parameters found in: {path}")
    return metadata


def parse_spc150_fifo(path: Path) -> tuple[dict[str, np.ndarray], list[MarkerEvent], int]:
    words = np.fromfile(path, dtype="<u4")
    if words.size < 2:
        raise ValueError(f"{path} does not contain enough 32-bit words to parse")

    header_word = int(words[0])
    records = words[1:]

    aux = ((records >> np.uint32(28)) & np.uint32(0xF)).astype(np.uint8)
    b = ((records >> np.uint32(16)) & np.uint32(0xFFF)).astype(np.uint16)
    detector = ((records >> np.uint32(12)) & np.uint32(0xF)).astype(np.uint8)
    macro_low = (records & np.uint32(0xFFF)).astype(np.uint16)

    is_invalid = (records & INVALID_FLAG) != 0
    has_overflow = (records & MACROTIME_OVERFLOW_FLAG) != 0
    has_gap = (records & GAP_FLAG) != 0
    has_marker = (records & MARKER_FLAG) != 0

    # Special overflow-only records have INVALID+MTOV set. MARK+MTOV records
    # and valid photon+MTOV records carry an event and one simultaneous wrap.
    is_marker = is_invalid & has_marker
    is_overflow = is_invalid & has_overflow & ~has_marker
    is_photon = ~is_invalid
    is_invalid_photon = is_invalid & ~has_marker & ~has_overflow

    overflow_steps = np.zeros(records.shape[0], dtype=np.int64)
    overflow_counts = (records & OVERFLOW_COUNT_MASK).astype(np.int64)
    overflow_counts[is_overflow & (overflow_counts == 0)] = 1
    overflow_steps[is_overflow] = overflow_counts[is_overflow] * MACROTIME_WRAP_TICKS
    combined_overflow = has_overflow & ~is_overflow
    overflow_steps[combined_overflow] = MACROTIME_WRAP_TICKS
    overflow_before = np.cumsum(overflow_steps, dtype=np.int64) - overflow_steps

    macrotime_ticks = overflow_before + macro_low.astype(np.int64)
    macrotime_ticks[combined_overflow] += overflow_steps[combined_overflow]
    nanotime_bin = (4095 - b.astype(np.int32)).astype(np.int32)

    marker_indices = np.flatnonzero(is_marker)
    marker_events: list[MarkerEvent] = []
    for idx in marker_indices.tolist():
        mask = int(detector[idx])
        marker_events.append(
            MarkerEvent(
                record_index=idx,
                word_index=idx + 1,
                word_hex=f"0x{int(records[idx]):08X}",
                aux=int(aux[idx]),
                detector_mask=mask,
                macrotime_ticks=int(macrotime_ticks[idx]),
                nanotime_bin=int(nanotime_bin[idx]),
                marker0=bool(mask & 0x1),
                marker1=bool(mask & 0x2),
                marker2=bool(mask & 0x4),
                marker3=bool(mask & 0x8),
            )
        )

    arrays = {
        "records": records,
        "aux": aux,
        "detector": detector,
        "macrotime_ticks": macrotime_ticks,
        "nanotime_bin": nanotime_bin,
        "is_overflow": is_overflow,
        "has_overflow": has_overflow,
        "overflow_wraps": overflow_steps // MACROTIME_WRAP_TICKS,
        "is_marker": is_marker,
        "is_photon": is_photon,
        "is_invalid": is_invalid,
        "is_invalid_photon": is_invalid_photon,
        "has_gap": has_gap,
    }
    return arrays, marker_events, header_word


def _interval_stats(intervals: np.ndarray, sync_rate_hz: float | None) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    if intervals.size == 0:
        return None, None
    stats_ticks = {
        "min": float(intervals.min()),
        "max": float(intervals.max()),
        "mean": float(intervals.mean()),
    }
    if sync_rate_hz is None:
        return stats_ticks, None
    intervals_seconds = intervals / sync_rate_hz
    stats_seconds = {
        "min": float(intervals_seconds.min()),
        "max": float(intervals_seconds.max()),
        "mean": float(intervals_seconds.mean()),
    }
    return stats_ticks, stats_seconds


def build_summary(
    spc_path: Path,
    set_module: str | None,
    set_params: dict[str, Any],
    arrays: dict[str, np.ndarray],
    marker_events: list[MarkerEvent],
    header_word: int,
    sync_rate_hz: float | None,
    sync_rate_source: str | None,
    set_path: Path | None = None,
    set_metadata: SetMetadata | None = None,
) -> ParseSummary:
    detector = arrays["detector"]
    is_marker = arrays["is_marker"]
    is_overflow = arrays["is_overflow"]

    marker_counts_by_bit = {
        "marker0": int(np.sum(is_marker & ((detector & np.uint8(0x1)) != 0))),
        "marker1": int(np.sum(is_marker & ((detector & np.uint8(0x2)) != 0))),
        "marker2": int(np.sum(is_marker & ((detector & np.uint8(0x4)) != 0))),
        "marker3": int(np.sum(is_marker & ((detector & np.uint8(0x8)) != 0))),
    }

    marker2_ticks_raw = np.array(
        [m.macrotime_ticks for m in marker_events if m.marker2], dtype=np.int64
    )
    pulse_starts = _marker_pulse_start_mask(marker2_ticks_raw)
    marker2_ticks = marker2_ticks_raw[pulse_starts]
    marker2_intervals = np.diff(marker2_ticks) if marker2_ticks.size > 1 else np.array([], dtype=np.int64)
    stats_ticks, stats_seconds = _interval_stats(marker2_intervals, sync_rate_hz)

    return ParseSummary(
        spc_file=str(spc_path),
        set_file=str(set_path) if set_path is not None else None,
        total_words=int(arrays["records"].size + 1),
        header_word_hex=f"0x{header_word:08X}",
        data_records=int(arrays["records"].size),
        overflow_records=int(np.sum(is_overflow)),
        overflow_wraps=int(np.sum(arrays["overflow_wraps"])),
        marker_records=int(np.sum(is_marker)),
        marker_counts_by_bit=marker_counts_by_bit,
        marker2_pulse_count=int(marker2_ticks.size),
        marker2_intervals_ticks=marker2_intervals.astype(int).tolist(),
        marker2_interval_stats_ticks=stats_ticks,
        marker2_interval_stats_seconds=stats_seconds,
        sync_rate_hz=sync_rate_hz,
        sync_rate_source=sync_rate_source,
        set_module=set_module,
        set_serial_number=set_metadata.serial_number if set_metadata is not None else None,
        set_identification=set_metadata.identification if set_metadata is not None else {},
        set_parameter_groups=set_metadata.parameter_groups if set_metadata is not None else {},
        set_params=set_params,
    )


def export_markers_csv(path: Path, marker_events: list[MarkerEvent], sync_rate_hz: float | None) -> None:
    fieldnames = [
        "record_index",
        "word_index",
        "word_hex",
        "aux",
        "detector_mask",
        "macrotime_ticks",
        "macrotime_seconds",
        "nanotime_bin",
        "marker0",
        "marker1",
        "marker2",
        "marker3",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in marker_events:
            row = asdict(event)
            row["macrotime_seconds"] = (
                event.macrotime_ticks / sync_rate_hz if sync_rate_hz is not None else ""
            )
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse SPC-150 FIFO .spc files and extract marker events.")
    parser.add_argument("spc_file", type=Path, help="Path to a .spc file")
    parser.add_argument(
        "--set-file",
        type=Path,
        required=True,
        help="Required .set metadata file from the same acquisition run",
    )
    parser.add_argument(
        "--sync-rate-hz",
        type=float,
        default=None,
        help="Optional sync rate override in Hz (default: read from SPC header)",
    )
    parser.add_argument(
        "--export-markers-csv",
        type=Path,
        default=None,
        help="Optional CSV export path for marker events",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    parser.add_argument(
        "--export-events-csv",
        type=Path,
        default=None,
        help="Optional CSV export path for unified photon/marker events",
    )
    parser.add_argument(
        "--export-events-parquet",
        type=Path,
        default=None,
        help="Optional Parquet export path for unified photon/marker events",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    arrays, marker_events, header_word = parse_spc150_fifo(args.spc_file)
    set_metadata = require_set_metadata(args.set_file)
    set_module = set_metadata.module
    set_params = set_metadata.spc_params
    sync_rate_hz, sync_rate_source = resolve_sync_rate_hz(
        args.sync_rate_hz, set_params, header_word=header_word
    )
    summary = build_summary(
        spc_path=args.spc_file,
        set_module=set_module,
        set_params=set_params,
        arrays=arrays,
        marker_events=marker_events,
        header_word=header_word,
        sync_rate_hz=sync_rate_hz,
        sync_rate_source=sync_rate_source,
        set_path=args.set_file,
        set_metadata=set_metadata,
    )

    if args.export_markers_csv is not None:
        export_markers_csv(args.export_markers_csv, marker_events, sync_rate_hz)

    if args.export_events_csv is not None or args.export_events_parquet is not None:
        if sync_rate_hz is None:
            raise ValueError(
                "Could not infer sync rate from set file. "
                "Provide --sync-rate-hz or a --set-file that contains SP_TAC_R."
            )
        events_df = build_unified_event_dataframe(arrays, sync_rate_hz)
        if args.export_events_csv is not None:
            events_df.to_csv(args.export_events_csv, index=False)
        if args.export_events_parquet is not None:
            events_df.to_parquet(args.export_events_parquet, index=False)

    json_kwargs: dict[str, Any] = {"sort_keys": True}
    if args.pretty:
        json_kwargs["indent"] = 2
    print(json.dumps(asdict(summary), **json_kwargs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
