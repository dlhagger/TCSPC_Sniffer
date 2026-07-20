# TCSPC Sniffer

`spc150_fifo_parser.py` decodes the Becker & Hickl 32-bit FIFO format used by
SPC-13x, SPC-140, SPC-15x, SPC-830, SPC-16x, and SPC-18x modules.

## Source of truth

The binary layout is implemented directly from the vendor-supplied
`BH/SPCM/SPC_data_file_structure.h`, section **FIFO Data Files (SPC-13x,
SPC-140, SPC-15x, SPC-830, SPC-16x, SPC-18x)**. The parser uses the constants
and meanings documented there:

- `INVALID32`, `MTOV32`, `OVRUN32`, `ROUT32`, `MT32`, and `ADC32`
- the 28-bit standalone-overflow count
- `RB_NO32`, `MT_CLK32`, `M_FILE32`, and `R_FILE32` in the first word
- Marker 0–3 records, including records with multiple marker bits

Raw diagnostic files (`R_FILE32`) are rejected because their invalid/overflow
entries explicitly do not have the normal documented semantics.

## Important marker behavior

Every marker record is preserved as an event. Marker inputs record edges, not
the sustained state of a TTL signal. The core decoder never merges marker
records automatically.

Camera-frame assignment is a separate analysis operation. By default every
requested marker edge starts a frame:

```python
events = build_unified_event_dataframe(arrays, sync_rate_hz)
framed_events, frames = assign_photons_to_frames(events, marker_id=2)
```

If acquisition-specific deglitching is required, provide an explicit threshold:

```python
marker2_ticks = events.loc[events["marker2"].fillna(False), "macrotime_ticks"]
suggested = infer_marker_debounce_ticks(marker2_ticks.to_numpy())
framed_events, frames = assign_photons_to_frames(
    events,
    marker_id=2,
    marker_debounce_ticks=suggested,
)
```

The suggestion is only a gap-distribution heuristic and is not part of the SPC
file format. It should be validated against the camera trigger configuration.

## Basic notebook workflow

`spc150_fifo_parser.ipynb` provides a minimal starting workflow:

1. Select paired SPC and SET paths and run strict integrity checks.
2. Build one pandas DataFrame containing valid photons and every raw marker
   edge, with macrotime, calibrated photon microtime, and absolute event time.
3. Build camera-frame windows and view all detector-channel count traces, the
   counts in one selected frame, and that frame's TCSPC microtime distribution.

The notebook uses `camera_frame` as the eventual join key for frame-indexed
behavioral annotations. Verify trigger counts against the saved video before
assuming a one-to-one join, particularly when the camera can drop frames.

Set the notebook's optional `VIDEO_PATH` to a SpinView MP4 to perform that
validation with FFprobe. It counts decoded frames, requires an exact match with
the selected TCSPC triggers, and adds both `video_frame_0based` and
`video_frame_1based` columns. The TCSPC trigger times remain authoritative;
MP4 timing metadata is diagnostic only. FFprobe must be installed separately
as part of [FFmpeg](https://ffmpeg.org/download.html).

## Command line

```shell
python spc150_fifo_parser.py run.spc --set-file run.set --pretty
```

The macrotime period comes from the low 24 bits of the SPC header in 0.1 ns
units. `SP_TAC_R` is the microtime/TAC range and is never used as a macrotime
fallback. `--sync-rate-hz` is available only as an explicit override.

CSV event exports retain `has_gap`, `has_overflow`, the raw flag nibble, raw
inverted routing bits, and the decoded detector channel. Invalid photon and
overflow-only records can be included with `--include-invalid` and
`--include-overflows`. Marker ADC bits are retained as `marker_payload_raw`;
they are unused for SPC-150 but represent marker intensity on SPC-160.

## Tests

```shell
python -m unittest discover -s tests -v
```

The synthetic tests cover header validation, all record classes, simultaneous
event/overflow flags, multi-bit markers, GAP propagation, 28-bit overflow
counts, routing inversion, explicit marker deglitching, and acquisition-pair
validation.
