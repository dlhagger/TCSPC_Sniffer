from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spc150_fifo_parser import (
    MACROTIME_WRAP_TICKS,
    MarkerEvent,
    SetMetadata,
    _marker_pulse_start_mask,
    assign_photons_to_frames,
    build_summary,
    build_unified_event_dataframe,
    infer_marker_debounce_ticks,
    infer_sync_rate_hz,
    parse_spc150_fifo,
    parse_spc_header,
    resolve_sync_rate_hz,
    validate_acquisition_pair,
)


HEADER = 0x80000000 | (4 << 27) | 0x02000000 | 500


def photon(*, mt: int, adc: int = 1000, routing_raw: int = 0xE, mtov: bool = False, gap: bool = False) -> int:
    return (
        (0x40000000 if mtov else 0)
        | (0x20000000 if gap else 0)
        | ((adc & 0xFFF) << 16)
        | ((routing_raw & 0xF) << 12)
        | (mt & 0xFFF)
    )


def marker(*, mt: int, mask: int, mtov: bool = False, gap: bool = False) -> int:
    return (
        0x90000000
        | (0x40000000 if mtov else 0)
        | (0x20000000 if gap else 0)
        | ((mask & 0xF) << 12)
        | (mt & 0xFFF)
    )


def overflow(count: int, *, gap: bool = False) -> int:
    return 0xC0000000 | (0x20000000 if gap else 0) | (count & 0x0FFFFFFF)


class TemporarySpc:
    def __init__(self, words: list[int], trailing: bytes = b"") -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "synthetic.spc"
        self.path.write_bytes(struct.pack(f"<{len(words)}I", *words) + trailing)

    def cleanup(self) -> None:
        self.directory.cleanup()


class HeaderTests(unittest.TestCase):
    def test_decodes_documented_header_fields(self) -> None:
        header = parse_spc_header(HEADER)
        self.assertEqual(header.routing_bits, 4)
        self.assertEqual(header.routing_channels, 16)
        self.assertTrue(header.markers_enabled)
        self.assertFalse(header.raw_data)
        self.assertEqual(header.macrotime_clock_units_100ps, 500)
        self.assertEqual(header.macrotime_rate_hz, 20_000_000.0)

    def test_rejects_invalid_and_foreign_headers(self) -> None:
        with self.assertRaisesRegex(ValueError, "bit 31"):
            parse_spc_header(500)
        with self.assertRaisesRegex(ValueError, "femtosecond"):
            parse_spc_header(HEADER | 0x01000000)
        with self.assertRaisesRegex(ValueError, "at most 4"):
            parse_spc_header(0x80000000 | (5 << 27) | 500)

    def test_tac_range_is_never_used_as_macrotime_clock(self) -> None:
        self.assertIsNone(infer_sync_rate_hz({"SP_TAC_R": 50e-9}))
        rate, source = resolve_sync_rate_hz(None, {"SP_TAC_R": 1e-6}, HEADER)
        self.assertEqual(rate, 20_000_000.0)
        self.assertEqual(source, "spc-header")


class RecordDecodingTests(unittest.TestCase):
    def parse(self, records: list[int]):
        temp = TemporarySpc([HEADER, *records])
        self.addCleanup(temp.cleanup)
        return parse_spc150_fifo(temp.path)

    def test_decodes_all_documented_record_classes(self) -> None:
        records = [
            photon(mt=10),
            photon(mt=20, mtov=True),
            overflow(3),
            marker(mt=30, mask=0b0101),
            marker(mt=40, mask=0b0100, mtov=True, gap=True),
            0x80000000 | (2000 << 16) | (0xA << 12) | 50,
        ]
        arrays, markers, _ = self.parse(records)

        np.testing.assert_array_equal(
            arrays["is_photon"], [True, True, False, False, False, False]
        )
        np.testing.assert_array_equal(
            arrays["is_overflow"], [False, False, True, False, False, False]
        )
        np.testing.assert_array_equal(
            arrays["is_marker"], [False, False, False, True, True, False]
        )
        np.testing.assert_array_equal(
            arrays["is_invalid_photon"], [False, False, False, False, False, True]
        )
        np.testing.assert_array_equal(arrays["overflow_wraps"], [0, 1, 3, 0, 1, 0])
        np.testing.assert_array_equal(
            arrays["macrotime_ticks"], [10, 4096 + 20, -1, 4 * 4096 + 30, 5 * 4096 + 40, -1]
        )
        self.assertEqual(int(arrays["detector_channel"][0]), 1)
        self.assertEqual(int(arrays["nanotime_bin"][0]), 3095)
        self.assertEqual(int(arrays["nanotime_bin"][3]), -1)

        self.assertEqual(len(markers), 2)
        self.assertEqual(markers[0].marker_mask, 0b0101)
        self.assertEqual(markers[0].marker_payload_raw, 0)
        self.assertTrue(markers[0].marker0)
        self.assertTrue(markers[0].marker2)
        self.assertTrue(markers[1].has_overflow)
        self.assertTrue(markers[1].has_gap)

    def test_overflow_count_is_exactly_28_bits_even_with_gap(self) -> None:
        arrays, _, _ = self.parse([overflow(2, gap=True), photon(mt=7)])
        self.assertEqual(int(arrays["overflow_count"][0]), 2)
        self.assertEqual(int(arrays["overflow_wraps"][0]), 2)
        self.assertEqual(int(arrays["macrotime_ticks"][1]), 2 * MACROTIME_WRAP_TICKS + 7)
        self.assertTrue(bool(arrays["has_gap"][0]))

    def test_rejects_zero_count_overflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero CNT"):
            self.parse([overflow(0)])

    def test_rejects_raw_diagnostic_file(self) -> None:
        temp = TemporarySpc([HEADER | 0x04000000, photon(mt=1)])
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "raw diagnostic"):
            parse_spc150_fifo(temp.path)

    def test_rejects_marker_when_header_does_not_declare_markers(self) -> None:
        temp = TemporarySpc([HEADER & ~0x02000000, marker(mt=1, mask=4)])
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "M_FILE32"):
            parse_spc150_fifo(temp.path)

    def test_rejects_trailing_partial_word(self) -> None:
        temp = TemporarySpc([HEADER, photon(mt=1)], trailing=b"x")
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "trailing byte"):
            parse_spc150_fifo(temp.path)


class DataFrameAndFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporarySpc(
            [
                HEADER,
                marker(mt=100, mask=0b0101),
                photon(mt=110),
                marker(mt=120, mask=0b0100, gap=True),
                photon(mt=130),
                0x80000001,
                overflow(2),
            ]
        )
        self.addCleanup(self.temp.cleanup)
        self.arrays, self.markers, self.header_word = parse_spc150_fifo(self.temp.path)

    def test_dataframe_preserves_flags_and_nulls_non_applicable_fields(self) -> None:
        df = build_unified_event_dataframe(
            self.arrays, 20_000_000, include_invalid=True, include_overflows=True
        )
        self.assertEqual(
            df["type"].tolist(),
            ["marker", "photon", "marker", "photon", "invalid_photon", "overflow"],
        )
        self.assertEqual(int(df.loc[0, "marker_mask"]), 5)
        self.assertEqual(int(df.loc[0, "marker_payload_raw"]), 0)
        self.assertTrue(bool(df.loc[0, "marker0"]))
        self.assertTrue(bool(df.loc[0, "marker2"]))
        self.assertTrue(bool(df.loc[2, "has_gap"]))
        self.assertTrue(bool(df["nanotime_bin"].isna().iloc[0]))
        self.assertTrue(bool(df["macrotime_ticks"].isna().iloc[4]))
        self.assertIn("flags_nibble", df.columns)
        self.assertNotIn("aux", df.columns)

    def test_frame_assignment_uses_marker_bits_and_preserves_all_edges_by_default(self) -> None:
        df = build_unified_event_dataframe(self.arrays, 20_000_000)
        framed, frames = assign_photons_to_frames(df, marker_id=2)
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames["frame_marker_record_index"].tolist(), [0, 2])
        self.assertEqual(frames["photon_count"].tolist(), [1, 1])
        self.assertEqual(framed.loc[framed["type"] == "photon", "frame_index"].tolist(), [0, 1])

    def test_explicit_debounce_is_application_level(self) -> None:
        df = build_unified_event_dataframe(self.arrays, 20_000_000)
        _, frames = assign_photons_to_frames(df, marker_id=2, marker_debounce_ticks=25)
        self.assertEqual(len(frames), 1)

    def test_summary_counts_raw_marker_events_without_debouncing(self) -> None:
        summary = build_summary(
            self.temp.path,
            "SPC-150",
            {},
            self.arrays,
            self.markers,
            self.header_word,
            20_000_000,
            "spc-header",
        )
        self.assertEqual(summary.marker2_event_count, 2)
        self.assertEqual(summary.marker2_pulse_count, 2)
        self.assertEqual(summary.gap_records, 1)


class MarkerSelectionTests(unittest.TestCase):
    def test_no_debounce_keeps_every_edge(self) -> None:
        np.testing.assert_array_equal(
            _marker_pulse_start_mask(np.array([10, 11, 100])), [True, True, True]
        )

    def test_suggested_threshold_requires_clear_separation(self) -> None:
        self.assertEqual(infer_marker_debounce_ticks(np.array([0, 1, 2, 102, 103])), 10)
        self.assertEqual(infer_marker_debounce_ticks(np.array([0, 5, 11, 18])), 0)


class MetadataTests(unittest.TestCase):
    def metadata(self, embedded_name: str) -> SetMetadata:
        return SetMetadata(
            "run.set",
            "SPC-150",
            None,
            {},
            {"SP": {"SP_SPE_FN": rf"C:\\data\\{embedded_name}"}},
        )

    def test_pair_validation_accepts_matching_basename(self) -> None:
        validate_acquisition_pair(Path("/other/Test.SPC"), self.metadata("test.spc"))

    def test_pair_validation_rejects_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "not 'test.spc'"):
            validate_acquisition_pair(Path("test.spc"), self.metadata("other.spc"))


class BundledSampleRegressionTests(unittest.TestCase):
    def test_camera_sample_matches_known_record_counts_and_explicit_edge_selection(self) -> None:
        project = Path(__file__).resolve().parents[1]
        arrays, markers, header_word = parse_spc150_fifo(project / "test_camera_30fps.spc")
        self.assertEqual(header_word, 0xA20001F4)
        self.assertEqual(arrays["records"].size, 343_051)
        self.assertEqual(int(np.sum(arrays["is_photon"])), 14_460)
        self.assertEqual(int(np.sum(arrays["is_invalid_photon"])), 180_829)
        self.assertEqual(int(np.sum(arrays["is_overflow"])), 138_966)
        self.assertEqual(int(np.sum(arrays["overflow_wraps"])), 297_832)
        self.assertEqual(len(markers), 8_796)
        self.assertEqual(int(np.sum(arrays["has_gap"])), 0)

        marker2_ticks = np.array(
            [event.macrotime_ticks for event in markers if event.marker2], dtype=np.int64
        )
        threshold = infer_marker_debounce_ticks(marker2_ticks)
        self.assertEqual(threshold, 142_161)
        self.assertEqual(
            int(np.sum(_marker_pulse_start_mask(marker2_ticks, threshold))), 1_830
        )


if __name__ == "__main__":
    unittest.main()
