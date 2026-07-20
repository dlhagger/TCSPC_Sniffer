#!/usr/bin/env python3
"""Decode Becker & Hickl SPC-13x/140/15x/830/16x/18x 32-bit FIFO files.

The binary definitions in ``BH/SPCM/SPC_data_file_structure.h`` are the
authoritative specification for this module.  The decoder preserves every
marker edge and every integrity flag.  Camera-frame selection is deliberately
kept as a separate, opt-in analysis step.
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


# Names and values mirror SPC_data_file_structure.h.
INVALID32 = np.uint32(0x80000000)
MTOV32 = np.uint32(0x40000000)
OVRUN32 = np.uint32(0x20000000)
MARK32 = np.uint32(0x10000000)
ROUT32 = np.uint32(0x0000F000)
MT32 = np.uint32(0x00000FFF)
ADC32 = np.uint32(0x0FFF0000)

RB_NO32 = np.uint32(0x78000000)
MT_CLK32 = np.uint32(0x00FFFFFF)
M_FILE32 = np.uint32(0x02000000)
R_FILE32 = np.uint32(0x04000000)
HEADER_RESERVED_FEMTO_FLAG = np.uint32(0x01000000)

OVERFLOW_COUNT_MASK = np.uint32(0x0FFFFFFF)  # CNT[27:0]
MACROTIME_WRAP_TICKS = 1 << 12
MAX_ROUTING_BITS = 4

# Backward-compatible public aliases.
INVALID_FLAG = INVALID32
MACROTIME_OVERFLOW_FLAG = MTOV32
GAP_FLAG = OVRUN32
MARKER_FLAG = MARK32


@dataclass(frozen=True)
class SpcHeader:
    word: int
    word_hex: str
    routing_bits: int
    routing_channels: int
    macrotime_clock_units_100ps: int
    macrotime_period_seconds: float
    macrotime_rate_hz: float
    markers_enabled: bool
    raw_data: bool


@dataclass(frozen=True)
class MarkerEvent:
    record_index: int
    word_index: int
    word_hex: str
    marker_mask: int
    marker_payload_raw: int
    macrotime_ticks: int
    has_overflow: bool
    has_gap: bool
    marker0: bool
    marker1: bool
    marker2: bool
    marker3: bool

    @property
    def detector_mask(self) -> int:
        """Deprecated alias retained for callers of the original parser."""
        return self.marker_mask


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
    header_routing_bits: int
    header_routing_channels: int
    header_markers_enabled: bool
    header_raw_data: bool
    data_records: int
    valid_photon_records: int
    invalid_photon_records: int
    overflow_records: int
    overflow_wraps: int
    marker_records: int
    gap_records: int
    marker_counts_by_bit: dict[str, int]
    marker2_event_count: int
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

    @property
    def marker2_pulse_count(self) -> int:
        """Deprecated name; marker records are edge events, not signal levels."""
        return self.marker2_event_count


def parse_spc_header(header_word: int, *, strict: bool = True) -> SpcHeader:
    """Validate and decode the first 32-bit word of an SPC FIFO file."""
    word = int(header_word) & 0xFFFFFFFF
    errors: list[str] = []
    if not word & int(INVALID32):
        errors.append("bit 31 (header/INVALID marker) is not set")

    routing_bits = (word & int(RB_NO32)) >> 27
    if routing_bits > MAX_ROUTING_BITS:
        errors.append(
            f"routing-bit count is {routing_bits}; this 32-bit format supports at most "
            f"{MAX_ROUTING_BITS}"
        )

    clock_units = word & int(MT_CLK32)
    if clock_units == 0:
        errors.append("macrotime clock in bits 23:0 is zero")

    if word & int(HEADER_RESERVED_FEMTO_FLAG):
        errors.append(
            "bit 24 is set (femtosecond header); this is not an SPC-13x/140/15x/830/16x/18x file"
        )

    if strict and errors:
        raise ValueError(f"Invalid SPC 32-bit FIFO header 0x{word:08X}: " + "; ".join(errors))

    period_s = clock_units * 1e-10 if clock_units else 0.0
    rate_hz = 1.0 / period_s if period_s else 0.0
    return SpcHeader(
        word=word,
        word_hex=f"0x{word:08X}",
        routing_bits=routing_bits,
        routing_channels=1 << routing_bits,
        macrotime_clock_units_100ps=clock_units,
        macrotime_period_seconds=period_s,
        macrotime_rate_hz=rate_hz,
        markers_enabled=bool(word & int(M_FILE32)),
        raw_data=bool(word & int(R_FILE32)),
    )


def _routing_channel_from_inverted_bits(
    routing_raw: np.ndarray, routing_bits: int
) -> np.ndarray:
    """Convert documented inverted R bits to zero-based detector channels."""
    if routing_bits == 0:
        return np.zeros(routing_raw.shape, dtype=np.uint8)
    mask = np.uint8((1 << routing_bits) - 1)
    return ((~routing_raw.astype(np.uint8)) & mask).astype(np.uint8)


def parse_spc150_fifo(
    path: Path,
    *,
    strict: bool = True,
) -> tuple[dict[str, np.ndarray], list[MarkerEvent], int]:
    """Decode an SPC 32-bit FIFO file according to SPC_data_file_structure.h.

    Raw diagnostic files are rejected because their INVALID+MTOV records do not
    have the documented overflow-count semantics.  Invalid photon records and
    overflow-only records have ``macrotime_valid=False`` and a timestamp of -1.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SPC file not found: {path}")
    byte_size = path.stat().st_size
    if byte_size % 4:
        raise ValueError(f"{path} has {byte_size % 4} trailing byte(s); expected 32-bit words")

    words = np.fromfile(path, dtype="<u4")
    if words.size < 2:
        raise ValueError(f"{path} does not contain a header and at least one data record")

    header_word = int(words[0])
    header = parse_spc_header(header_word, strict=strict)
    if header.raw_data:
        raise ValueError(
            f"{path} is a raw diagnostic SPC file (R_FILE32 set); overflow reconstruction "
            "is undefined for raw records"
        )

    records = words[1:]
    flags_nibble = ((records >> np.uint32(28)) & np.uint32(0xF)).astype(np.uint8)
    adc_raw = ((records & ADC32) >> np.uint32(16)).astype(np.uint16)
    routing_raw = ((records & ROUT32) >> np.uint32(12)).astype(np.uint8)
    macrotime_low = (records & MT32).astype(np.uint16)

    is_invalid = (records & INVALID32) != 0
    has_overflow = (records & MTOV32) != 0
    has_gap = (records & OVRUN32) != 0
    has_marker_flag = (records & MARK32) != 0

    is_marker = is_invalid & has_marker_flag
    is_overflow = is_invalid & has_overflow & ~has_marker_flag
    is_photon = ~is_invalid
    is_invalid_photon = is_invalid & ~has_marker_flag & ~has_overflow

    if strict and np.any(has_marker_flag & ~is_invalid):
        first = int(np.flatnonzero(has_marker_flag & ~is_invalid)[0])
        raise ValueError(
            f"Record {first} sets MARK without INVALID; it is not a documented marker record"
        )
    if strict and np.any(is_marker) and not header.markers_enabled:
        raise ValueError("Marker records are present but M_FILE32 is not set in the SPC header")

    overflow_counts = np.zeros(records.shape, dtype=np.int64)
    overflow_counts[is_overflow] = (
        records[is_overflow] & OVERFLOW_COUNT_MASK
    ).astype(np.int64)
    if strict and np.any(is_overflow & (overflow_counts == 0)):
        first = int(np.flatnonzero(is_overflow & (overflow_counts == 0))[0])
        raise ValueError(f"Overflow record {first} contains an invalid zero CNT[27:0]")

    overflow_wraps = overflow_counts.copy()
    event_with_overflow = has_overflow & ~is_overflow
    overflow_wraps[event_with_overflow] = 1
    overflow_steps = overflow_wraps * MACROTIME_WRAP_TICKS
    overflow_before = np.cumsum(overflow_steps, dtype=np.int64) - overflow_steps

    macrotime_ticks = overflow_before + macrotime_low.astype(np.int64)
    macrotime_ticks[event_with_overflow] += MACROTIME_WRAP_TICKS
    macrotime_valid = is_photon | is_marker
    macrotime_ticks[~macrotime_valid] = -1

    nanotime_bin = 4095 - adc_raw.astype(np.int32)
    nanotime_valid = is_photon
    nanotime_bin[~nanotime_valid] = -1

    detector_channel = _routing_channel_from_inverted_bits(
        routing_raw, header.routing_bits
    )

    marker_events: list[MarkerEvent] = []
    for idx in np.flatnonzero(is_marker).tolist():
        marker_mask = int(routing_raw[idx])
        marker_events.append(
            MarkerEvent(
                record_index=idx,
                word_index=idx + 1,
                word_hex=f"0x{int(records[idx]):08X}",
                marker_mask=marker_mask,
                marker_payload_raw=int(adc_raw[idx]),
                macrotime_ticks=int(macrotime_ticks[idx]),
                has_overflow=bool(has_overflow[idx]),
                has_gap=bool(has_gap[idx]),
                marker0=bool(marker_mask & 0x1),
                marker1=bool(marker_mask & 0x2),
                marker2=bool(marker_mask & 0x4),
                marker3=bool(marker_mask & 0x8),
            )
        )

    arrays = {
        "records": records,
        "flags_nibble": flags_nibble,
        "adc_raw": adc_raw,
        "routing_raw": routing_raw,
        "detector": routing_raw,  # compatibility: raw R/M nibble
        "detector_channel": detector_channel,
        "macrotime_low": macrotime_low,
        "macrotime_ticks": macrotime_ticks,
        "macrotime_valid": macrotime_valid,
        "nanotime_bin": nanotime_bin,
        "nanotime_valid": nanotime_valid,
        "overflow_count": overflow_counts,
        "overflow_wraps": overflow_wraps,
        "is_overflow": is_overflow,
        "has_overflow": has_overflow,
        "is_marker": is_marker,
        "is_photon": is_photon,
        "is_invalid": is_invalid,
        "is_invalid_photon": is_invalid_photon,
        "has_gap": has_gap,
        "header_routing_bits": np.full(records.shape, header.routing_bits, dtype=np.uint8),
    }
    return arrays, marker_events, header_word


def build_unified_event_dataframe(
    arrays: dict[str, np.ndarray],
    sync_rate_hz: float,
    *,
    include_invalid: bool = False,
    include_overflows: bool = False,
):
    """Return a typed record table while retaining all integrity flags."""
    if not np.isfinite(sync_rate_hz) or sync_rate_hz <= 0:
        raise ValueError("sync_rate_hz must be finite and > 0")
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for unified DataFrame export") from exc

    event_mask = arrays["is_photon"] | arrays["is_marker"]
    if include_invalid:
        event_mask |= arrays["is_invalid_photon"]
    if include_overflows:
        event_mask |= arrays["is_overflow"]

    record_index = np.flatnonzero(event_mask).astype(np.int64)
    is_photon = arrays["is_photon"][event_mask]
    is_marker = arrays["is_marker"][event_mask]
    is_invalid_photon = arrays["is_invalid_photon"][event_mask]
    is_overflow = arrays["is_overflow"][event_mask]
    row_count = record_index.size

    record_type = np.full(row_count, "unknown", dtype=object)
    record_type[is_photon] = "photon"
    record_type[is_marker] = "marker"
    record_type[is_invalid_photon] = "invalid_photon"
    record_type[is_overflow] = "overflow"

    macrotime_raw = arrays["macrotime_ticks"][event_mask].astype(np.int64)
    macrotime_valid = arrays["macrotime_valid"][event_mask]
    macrotime_ticks = pd.Series(macrotime_raw, dtype="Int64")
    macrotime_ticks[~macrotime_valid] = pd.NA
    macrotime_s = np.full(row_count, np.nan, dtype=np.float64)
    macrotime_s[macrotime_valid] = macrotime_raw[macrotime_valid] / float(sync_rate_hz)

    routing_raw_values = arrays["routing_raw"][event_mask].astype(np.int16)
    routing_bits_raw = pd.Series(routing_raw_values, dtype="Int64")
    routing_bits_raw[~is_photon] = pd.NA
    detector_values = arrays["detector_channel"][event_mask].astype(np.int16)
    detector_channel = pd.Series(detector_values, dtype="Int64")
    detector_channel[~is_photon] = pd.NA
    detector_channel_1based = pd.Series(detector_values + 1, dtype="Int64")
    detector_channel_1based[~is_photon] = pd.NA

    adc_raw = pd.Series(arrays["adc_raw"][event_mask].astype(np.int32), dtype="Int64")
    adc_raw[~is_photon] = pd.NA
    nanotime = pd.Series(arrays["nanotime_bin"][event_mask].astype(np.int32), dtype="Int64")
    nanotime[~is_photon] = pd.NA

    marker_mask = pd.Series(routing_raw_values, dtype="Int64")
    marker_mask[~is_marker] = pd.NA
    marker_payload_raw = pd.Series(
        arrays["adc_raw"][event_mask].astype(np.int32), dtype="Int64"
    )
    marker_payload_raw[~is_marker] = pd.NA
    marker_id = pd.Series([pd.NA] * row_count, dtype="Int64")
    marker_columns = {
        bit: pd.Series([pd.NA] * row_count, dtype="boolean") for bit in range(4)
    }
    for row in np.flatnonzero(is_marker).tolist():
        mask = int(routing_raw_values[row])
        for bit in range(4):
            marker_columns[bit].iloc[row] = bool(mask & (1 << bit))
        if mask in (1, 2, 4, 8):
            marker_id.iloc[row] = mask.bit_length() - 1

    return pd.DataFrame(
        {
            "record_index": record_index,
            "word_index": record_index + 1,
            "word_hex": [f"0x{int(v):08X}" for v in arrays["records"][event_mask]],
            "type": record_type,
            "macrotime_ticks": macrotime_ticks,
            "macrotime_s": macrotime_s,
            "routing_bits_raw": routing_bits_raw,
            "detector_channel": detector_channel,
            "detector_channel_1based": detector_channel_1based,
            "adc_raw": adc_raw,
            "nanotime_bin": nanotime,
            "marker_mask": marker_mask,
            "marker_payload_raw": marker_payload_raw,
            "marker_id": marker_id,
            "marker0": marker_columns[0],
            "marker1": marker_columns[1],
            "marker2": marker_columns[2],
            "marker3": marker_columns[3],
            "is_invalid": arrays["is_invalid"][event_mask],
            "has_overflow": arrays["has_overflow"][event_mask],
            "has_gap": arrays["has_gap"][event_mask],
            "overflow_count": arrays["overflow_count"][event_mask],
            "flags_nibble": arrays["flags_nibble"][event_mask],
        }
    )


def infer_marker_debounce_ticks(marker_ticks: np.ndarray) -> int:
    """Suggest an application-level debounce threshold from a bimodal gap set.

    This is not part of SPC decoding.  A return value of zero means that no
    defensible >=10x separation was found and no edges should be suppressed.
    """
    marker_ticks = np.asarray(marker_ticks, dtype=np.int64)
    if marker_ticks.size < 2:
        return 0
    gaps = np.diff(marker_ticks)
    if np.any(gaps < 0):
        raise ValueError("marker_ticks must be monotonically nondecreasing")
    positive_gaps = np.unique(gaps[gaps > 0])
    if positive_gaps.size < 2:
        return 0
    ratios = positive_gaps[1:] / positive_gaps[:-1]
    clear_splits = np.flatnonzero(ratios >= 10.0)
    if clear_splits.size == 0:
        return 0
    split = int(clear_splits[-1])
    return int(np.sqrt(float(positive_gaps[split]) * float(positive_gaps[split + 1])))


def _marker_pulse_start_mask(
    marker_ticks: np.ndarray, debounce_ticks: int | None = None
) -> np.ndarray:
    """Select edges; by default every documented marker event is retained."""
    marker_ticks = np.asarray(marker_ticks, dtype=np.int64)
    if marker_ticks.size == 0:
        return np.array([], dtype=bool)
    if debounce_ticks is None or debounce_ticks == 0:
        return np.ones(marker_ticks.size, dtype=bool)
    if debounce_ticks < 0:
        raise ValueError("debounce_ticks must be >= 0")
    gaps = np.diff(marker_ticks)
    if np.any(gaps < 0):
        raise ValueError("marker_ticks must be monotonically nondecreasing")
    return np.concatenate(([True], gaps > int(debounce_ticks)))


def assign_photons_to_frames(
    events_df,
    marker_id: int = 2,
    frame_start_index: int = 0,
    keep_partial_last_frame: bool = True,
    marker_debounce_ticks: int | None = None,
):
    """Assign photons between selected marker edges to frame windows.

    Every requested marker edge is used by default.  ``marker_debounce_ticks``
    is an explicit application-level filter: after an accepted edge, subsequent
    edges separated by no more than the threshold are suppressed.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for frame assignment") from exc
    if marker_id not in range(4):
        raise ValueError("marker_id must be in 0..3")
    if marker_debounce_ticks is not None and marker_debounce_ticks < 0:
        raise ValueError("marker_debounce_ticks must be >= 0")

    df = events_df.copy().reset_index(drop=True)
    if df.empty:
        return df, pd.DataFrame()
    required = {"record_index", "type", "macrotime_ticks", "macrotime_s", f"marker{marker_id}"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"events_df is missing required column(s): {', '.join(missing)}")

    record_index = df["record_index"].to_numpy(dtype=np.int64)
    if np.any(np.diff(record_index) < 0):
        raise ValueError("events_df must be sorted by record_index")
    event_type = df["type"].to_numpy(dtype=object)
    marker_bit = df[f"marker{marker_id}"].fillna(False).to_numpy(dtype=bool)
    is_frame_marker = (event_type == "marker") & marker_bit
    marker_rows = df.loc[
        is_frame_marker, ["record_index", "macrotime_ticks", "macrotime_s"]
    ].copy()
    if marker_rows.empty:
        raise ValueError(f"No marker events found with marker bit {marker_id} set")

    marker_ticks = marker_rows["macrotime_ticks"].to_numpy(dtype=np.int64)
    selected = _marker_pulse_start_mask(marker_ticks, marker_debounce_ticks)
    marker_rows = marker_rows.iloc[np.flatnonzero(selected)].copy()

    frame_table = marker_rows.reset_index(drop=True).rename(
        columns={
            "record_index": "frame_marker_record_index",
            "macrotime_ticks": "frame_start_ticks",
            "macrotime_s": "frame_start_s",
        }
    )
    frame_table["frame_index"] = (
        np.arange(frame_table.shape[0], dtype=np.int64) + int(frame_start_index)
    )
    frame_table["next_record_index"] = frame_table["frame_marker_record_index"].shift(-1)
    frame_table["frame_end_ticks"] = frame_table["frame_start_ticks"].shift(-1)
    frame_table["frame_end_s"] = frame_table["frame_start_s"].shift(-1)

    marker_starts = frame_table["frame_marker_record_index"].to_numpy(dtype=np.int64)
    frame_position = np.searchsorted(marker_starts, record_index, side="right") - 1
    valid = frame_position >= 0
    if not keep_partial_last_frame:
        valid &= frame_position < marker_starts.size - 1
    assigned_rows = np.flatnonzero((event_type == "photon") & valid)

    lookup_specs = {
        "frame_index": ("frame_index", "Int64"),
        "frame_start_s": ("frame_start_s", None),
        "frame_end_s": ("frame_end_s", None),
        "frame_start_ticks": ("frame_start_ticks", "Int64"),
        "frame_end_ticks": ("frame_end_ticks", "Int64"),
        "frame_marker_record_index": ("frame_marker_record_index", "Int64"),
        "next_frame_marker_record_index": ("next_record_index", "Int64"),
    }
    for output_col, (frame_col, nullable_dtype) in lookup_specs.items():
        values = np.full(df.shape[0], np.nan, dtype=np.float64)
        if assigned_rows.size:
            lookup = frame_table[frame_col].to_numpy(dtype=np.float64)
            values[assigned_rows] = lookup[frame_position[assigned_rows]]
        df[output_col] = (
            pd.Series(values, dtype=nullable_dtype) if nullable_dtype else values
        )

    photon_counts = (
        df.loc[df["type"] == "photon", "frame_index"]
        .dropna()
        .astype("int64")
        .value_counts()
        .to_dict()
    )
    frame_table["photon_count"] = (
        frame_table["frame_index"].map(photon_counts).fillna(0).astype(np.int64)
    )
    if not keep_partial_last_frame:
        frame_table = frame_table.iloc[:-1].copy()
    return df, frame_table


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


def parse_set_metadata(path: Path) -> SetMetadata:
    """Parse run identity and all typed parameter groups from a SET file."""
    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("latin1", errors="ignore")
    module_match = re.search(
        r"with module\s+(SPC-\d+)(?:\s+\(Ser\.No\.\s*([^)]+)\))?", text
    )
    module_name = module_match.group(1) if module_match else None
    serial_number = (
        module_match.group(2).strip()
        if module_match and module_match.group(2)
        else None
    )

    identification: dict[str, str] = {}
    identification_match = re.search(r"\*IDENTIFICATION(.*?)\*END", text, re.DOTALL)
    if identification_match:
        for line in identification_match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            if key:
                identification[key] = re.sub(r"[\x00-\x1f\x7f]", "", value).strip()

    groups: dict[str, dict[str, Any]] = {}
    pattern = r"#([A-Z0-9]+)\s+\[([A-Z0-9_]+),([A-Z]),([^\]]*)\]"
    for group, key, value_type, value in re.findall(pattern, text):
        groups.setdefault(group, {})[key] = _parse_set_value(value_type, value)
    return SetMetadata(str(path), module_name, serial_number, identification, groups)


def parse_set_file(path: Path | None) -> tuple[str | None, dict[str, Any]]:
    if path is None:
        return None, {}
    metadata = parse_set_metadata(path)
    return metadata.module, metadata.spc_params


def require_set_metadata(path: Path) -> SetMetadata:
    path = Path(path)
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


def validate_acquisition_pair(spc_path: Path, metadata: SetMetadata) -> None:
    """Reject a SET file that explicitly names a different SPC acquisition."""
    embedded = metadata.spc_params.get("SP_SPE_FN")
    if not isinstance(embedded, str) or not embedded.strip():
        return
    embedded_name = re.split(r"[\\/]", embedded.strip())[-1]
    if embedded_name.casefold() != Path(spc_path).name.casefold():
        raise ValueError(
            f"SET file references {embedded_name!r}, not {Path(spc_path).name!r}"
        )


def infer_sync_rate_hz(set_params: dict[str, Any]) -> float | None:
    """Return None: SET parameters alone do not define the FIFO timebase.

    In particular, SP_TAC_R is the microtime/TAC range and must not be used as
    the macrotime clock.
    """
    del set_params
    return None


def infer_sync_rate_hz_from_header(header_word: int | None) -> float | None:
    if header_word is None:
        return None
    header = parse_spc_header(header_word, strict=False)
    return header.macrotime_rate_hz or None


def resolve_sync_rate_hz(
    sync_rate_hz_override: float | None,
    set_params: dict[str, Any],
    header_word: int | None = None,
) -> tuple[float | None, str | None]:
    del set_params
    if sync_rate_hz_override is not None:
        if not np.isfinite(sync_rate_hz_override) or sync_rate_hz_override <= 0:
            raise ValueError("sync_rate_hz override must be finite and > 0")
        return float(sync_rate_hz_override), "cli"
    rate = infer_sync_rate_hz_from_header(header_word)
    return (rate, "spc-header") if rate is not None else (None, None)


def _interval_stats(
    intervals: np.ndarray, sync_rate_hz: float | None
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    if intervals.size == 0:
        return None, None
    ticks = {
        "min": float(intervals.min()),
        "max": float(intervals.max()),
        "mean": float(intervals.mean()),
    }
    if sync_rate_hz is None:
        return ticks, None
    seconds = intervals / sync_rate_hz
    return ticks, {
        "min": float(seconds.min()),
        "max": float(seconds.max()),
        "mean": float(seconds.mean()),
    }


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
    header = parse_spc_header(header_word)
    marker_counts = {
        f"marker{bit}": sum(bool(event.marker_mask & (1 << bit)) for event in marker_events)
        for bit in range(4)
    }
    marker2_ticks = np.array(
        [event.macrotime_ticks for event in marker_events if event.marker2], dtype=np.int64
    )
    intervals = np.diff(marker2_ticks) if marker2_ticks.size > 1 else np.array([], dtype=np.int64)
    stats_ticks, stats_seconds = _interval_stats(intervals, sync_rate_hz)
    return ParseSummary(
        spc_file=str(spc_path),
        set_file=str(set_path) if set_path is not None else None,
        total_words=int(arrays["records"].size + 1),
        header_word_hex=header.word_hex,
        header_routing_bits=header.routing_bits,
        header_routing_channels=header.routing_channels,
        header_markers_enabled=header.markers_enabled,
        header_raw_data=header.raw_data,
        data_records=int(arrays["records"].size),
        valid_photon_records=int(np.sum(arrays["is_photon"])),
        invalid_photon_records=int(np.sum(arrays["is_invalid_photon"])),
        overflow_records=int(np.sum(arrays["is_overflow"])),
        overflow_wraps=int(np.sum(arrays["overflow_wraps"])),
        marker_records=int(np.sum(arrays["is_marker"])),
        gap_records=int(np.sum(arrays["has_gap"])),
        marker_counts_by_bit=marker_counts,
        marker2_event_count=int(marker2_ticks.size),
        marker2_intervals_ticks=intervals.astype(int).tolist(),
        marker2_interval_stats_ticks=stats_ticks,
        marker2_interval_stats_seconds=stats_seconds,
        sync_rate_hz=sync_rate_hz,
        sync_rate_source=sync_rate_source,
        set_module=set_module,
        set_serial_number=set_metadata.serial_number if set_metadata else None,
        set_identification=set_metadata.identification if set_metadata else {},
        set_parameter_groups=set_metadata.parameter_groups if set_metadata else {},
        set_params=set_params,
    )


def export_markers_csv(
    path: Path, marker_events: list[MarkerEvent], sync_rate_hz: float | None
) -> None:
    fieldnames = [
        "record_index", "word_index", "word_hex", "marker_mask", "marker_payload_raw",
        "macrotime_ticks", "macrotime_seconds", "has_overflow", "has_gap",
        "marker0", "marker1", "marker2", "marker3",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in marker_events:
            row = asdict(event)
            row["macrotime_seconds"] = (
                event.macrotime_ticks / sync_rate_hz if sync_rate_hz else ""
            )
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode documented Becker & Hickl 32-bit SPC FIFO records."
    )
    parser.add_argument("spc_file", type=Path)
    parser.add_argument("--set-file", type=Path, required=True)
    parser.add_argument("--sync-rate-hz", type=float)
    parser.add_argument("--export-markers-csv", type=Path)
    parser.add_argument("--export-events-csv", type=Path)
    parser.add_argument("--export-events-parquet", type=Path)
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--include-overflows", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    arrays, marker_events, header_word = parse_spc150_fifo(args.spc_file)
    metadata = require_set_metadata(args.set_file)
    validate_acquisition_pair(args.spc_file, metadata)
    sync_rate_hz, sync_rate_source = resolve_sync_rate_hz(
        args.sync_rate_hz, metadata.spc_params, header_word
    )
    if sync_rate_hz is None:
        parser.error("SPC header has no usable clock; provide --sync-rate-hz")

    summary = build_summary(
        args.spc_file,
        metadata.module,
        metadata.spc_params,
        arrays,
        marker_events,
        header_word,
        sync_rate_hz,
        sync_rate_source,
        args.set_file,
        metadata,
    )
    if args.export_markers_csv:
        export_markers_csv(args.export_markers_csv, marker_events, sync_rate_hz)
    if args.export_events_csv or args.export_events_parquet:
        events = build_unified_event_dataframe(
            arrays,
            sync_rate_hz,
            include_invalid=args.include_invalid,
            include_overflows=args.include_overflows,
        )
        if args.export_events_csv:
            events.to_csv(args.export_events_csv, index=False)
        if args.export_events_parquet:
            try:
                events.to_parquet(args.export_events_parquet, index=False)
            except ImportError as exc:
                raise ImportError(
                    "Parquet export requires pyarrow or fastparquet"
                ) from exc

    json_kwargs: dict[str, Any] = {"sort_keys": True}
    if args.pretty:
        json_kwargs["indent"] = 2
    print(json.dumps(asdict(summary), **json_kwargs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
