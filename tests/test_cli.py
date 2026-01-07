"""Tests for CLI helpers."""

from __future__ import annotations

import json

import numpy as np
import tifffile

from capella_reader import cli


def _write_test_tiff(path, metadata_dict):
    test_data = np.random.randn(10, 10).astype(np.complex64)
    gcp_values = np.array(
        [
            0.0,
            0.0,
            0.0,
            10.0,
            20.0,
            30.0,
            5.0,
            10.0,
            0.0,
            40.0,
            50.0,
            60.0,
        ],
        dtype=np.float64,
    )
    extratags = [(33922, "d", len(gcp_values), gcp_values, False)]

    with tifffile.TiffWriter(path) as tif:
        tif.write(
            test_data,
            description=json.dumps(metadata_dict),
            extratags=extratags,
        )


def test_cli_dump_suppresses_long_entries(tmp_path, sample_metadata_dict, capsys):
    """Test dump suppresses long entries by default."""
    metadata_path = tmp_path / "test.json"
    metadata_path.write_text(json.dumps(sample_metadata_dict))

    exit_code = cli.main(["dump", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "attitude samples; use --full" in captured.out
    assert "state vectors; use --full" in captured.out
    assert "transmit_antenna" in captured.out
    assert "receive_antenna" in captured.out
    assert "suppressed; use --full" in captured.out


def test_cli_dump_full_includes_entries(tmp_path, sample_metadata_dict, capsys):
    """Test dump includes long entries with --full."""
    metadata_path = tmp_path / "test.json"
    metadata_path.write_text(json.dumps(sample_metadata_dict))

    exit_code = cli.main(["dump", str(metadata_path), "--full"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "attitude samples; use --full" not in captured.out
    assert "state vectors; use --full" not in captured.out
    assert "suppressed; use --full" not in captured.out


def test_cli_bounds_text_output(tmp_path, sample_metadata_dict, capsys):
    """Test bounds emits plain text output."""
    tiff_path = tmp_path / "test_gcps.tif"
    _write_test_tiff(tiff_path, sample_metadata_dict)

    exit_code = cli.main(["bounds", str(tiff_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "10.0 20.0 40.0 50.0"


def test_cli_bounds_json_output(tmp_path, sample_metadata_dict, capsys):
    """Test bounds emits JSON output."""
    tiff_path = tmp_path / "test_gcps.tif"
    _write_test_tiff(tiff_path, sample_metadata_dict)

    exit_code = cli.main(["bounds", str(tiff_path), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "[10.0, 20.0, 40.0, 50.0]"
