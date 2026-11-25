"""Tests for the SNAP adapter."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile

from capella_reader import cli
from capella_reader.adapters import snap
from capella_reader.slc import CapellaImageGeometryError, CapellaSLC

DATA_DIR = Path(__file__).parent / "data"
STRIPMAP_JSON = (
    DATA_DIR / "CAPELLA_C17_SM_SLC_HH_20251103180619_20251103180628_extended.json"
)
STRIPMAP_REAL_TIME_JSON = (
    DATA_DIR / "CAPELLA_C11_SM_SLC_VV_20251031191104_20251031191109_extended.json"
)
PFA_JSON = (
    DATA_DIR / "CAPELLA_C13_SP_SLC_HH_20250826023518_20250826023527_extended.json"
)
SMALL_SHAPE = (300, 420)


def _as_spotlight(slc: CapellaSLC) -> CapellaSLC:
    """Give a slant-plane SLC the spotlight mode.

    The test data has no slant-plane spotlight SLC. The stripmap metadata has
    the reference antenna and target positions that the spotlight phase uses.
    """
    collect = slc.collect.model_copy(update={"mode": "spotlight"})
    return slc.model_copy(
        update={"meta": slc.meta.model_copy(update={"collect": collect})}
    )


def _write_small_tiff(
    json_file: Path, out_dir: Path, *, tiled: bool = True
) -> tuple[Path, np.ndarray]:
    """Write a small SLC GeoTIFF that has the metadata of `json_file`."""
    meta = json.loads(json_file.read_text())
    rows, cols = SMALL_SHAPE
    meta["collect"]["image"]["rows"] = rows
    meta["collect"]["image"]["columns"] = cols
    rng = np.random.default_rng(0)
    data = (
        rng.integers(-500, 500, SMALL_SHAPE) + 1j * rng.integers(-500, 500, SMALL_SHAPE)
    ).astype(np.complex64)
    path = out_dir / json_file.name.replace("_extended.json", ".tif")
    layout: dict[str, Any] = {"tile": (128, 128)} if tiled else {"rowsperstrip": 64}
    tifffile.imwrite(
        path,
        data,
        compression="zlib",
        description=json.dumps(meta),
        metadata=None,
        **layout,
    )
    return path, data


def _radar_coordinates(slc: CapellaSLC, target: np.ndarray) -> tuple[float, float]:
    """Find the zero-Doppler (row, col) of an ECEF point."""
    orbit = snap._Orbit(slc)
    t = orbit.seconds(slc.center_time.as_numpy())
    for _ in range(10):
        position, velocity = orbit(t)
        t += float(np.dot(target - position, velocity) / np.dot(velocity, velocity))
    position, _ = orbit(t)
    row = (t - orbit.seconds(slc.first_line_time.as_numpy())) / slc.delta_line_time
    col = (
        np.linalg.norm(target - position) - slc.range_to_first_sample
    ) / slc.delta_range_sample
    return float(row), float(col)


@pytest.mark.parametrize("json_file", [STRIPMAP_JSON, STRIPMAP_REAL_TIME_JSON])
def test_geolocate_reproduces_center_target(json_file):
    slc = CapellaSLC.from_file(json_file)
    target = slc.collect.image.center_pixel.target_position.as_array()
    height = float(snap._ecef_to_llh(target)[2])
    row, col = _radar_coordinates(slc, target)

    located = snap.geolocate(slc, row, col, height=height)

    np.testing.assert_allclose(located, target, atol=1e-3)


def test_geolocate_broadcasts():
    slc = CapellaSLC.from_file(STRIPMAP_JSON)
    rows = np.array([0.0, 1000.0])[:, None]
    cols = np.array([0.0, 500.0, 1000.0])[None, :]

    located = snap.geolocate(slc, rows, cols)

    assert located.shape == (2, 3, 3)
    heights = snap._ecef_to_llh(located)[2]
    np.testing.assert_allclose(heights, snap._reference_height(slc), atol=1e-3)


def test_reference_height_is_the_focusing_surface():
    slc = CapellaSLC.from_file(STRIPMAP_JSON)
    assert snap._reference_height(slc) == pytest.approx(-10.8501615524292)


def test_abstracted_metadata():
    slc = CapellaSLC.from_file(STRIPMAP_JSON)

    meta = snap.abstracted_metadata(slc)

    assert meta["PRODUCT"] == "CAPELLA_C17_SM_SLC_HH_20251103180619_20251103180628"
    assert meta["ACQUISITION_MODE"] == "stripmap"
    assert meta["antenna_pointing"] == "right"
    # SNAP needs the image line rate here, not the radar PRF
    assert meta["pulse_repetition_frequency"] == pytest.approx(1 / slc.delta_line_time)
    assert meta["range_sampling_rate"] == pytest.approx(
        snap.C_LIGHT / (2 * slc.delta_range_sample) / 1e6
    )
    assert meta["range_bandwidth"] == pytest.approx(
        slc.collect.image.processed_range_bandwidth / 1e6
    )
    assert meta["radar_frequency"] == pytest.approx(slc.center_frequency / 1e6)
    assert meta["num_output_lines"] == slc.shape[0]
    assert meta["num_samples_per_line"] == slc.shape[1]
    # SNAP takes the range direction from the incidence angles
    assert meta["incidence_near"] < meta["incidence_far"]
    center_incidence = slc.collect.image.center_pixel.incidence_angle
    assert meta["incidence_near"] < center_incidence < meta["incidence_far"]


def test_format_utc_rounds_to_microseconds():
    assert (
        snap._format_utc(np.datetime64("2025-09-21T04:33:57.525136517", "ns"))
        == "21-SEP-2025 04:33:57.525137"
    )
    assert (
        snap._format_utc(np.datetime64("2025-01-02T03:04:05", "ns"))
        == "02-JAN-2025 03:04:05.000000"
    )


def test_spotlight_phase_does_not_depend_on_height():
    slc = _as_spotlight(CapellaSLC.from_file(STRIPMAP_JSON))
    center_row = slc.shape[0] / 2
    # A span of 2.5 km along track on each side of the scene center
    half_span = 2500 / slc.collect.image.pixel_spacing_row
    rows = center_row + np.linspace(-half_span, half_span, 5)[:, None]
    cols = np.linspace(0, slc.shape[1] - 1, 5)[None, :]
    height = snap._reference_height(slc)

    low = snap.spotlight_phase(slc, rows, cols, height=height)
    high = snap.spotlight_phase(slc, rows, cols, height=height + 100)

    assert np.abs(high - low).max() < 0.1


def test_spotlight_phase_is_an_azimuth_chirp():
    slc = _as_spotlight(CapellaSLC.from_file(STRIPMAP_JSON))
    antenna = slc.collect.image.reference_antenna_position.as_array()
    target = slc.collect.image.center_pixel.target_position.as_array()
    antenna_row, _ = _radar_coordinates(slc, antenna + (target - antenna) * 1e-9)
    rows = antenna_row + np.array([-2000.0, 0.0, 2000.0])
    col = slc.shape[1] / 2

    range_term = snap._spotlight_range(slc, rows, np.full(3, col), None)

    # |A - T| - R0 is zero where the antenna is at closest approach. At the
    # along-track distance x of the antenna it is about x^2 / (2 R0), times
    # the ratio of the ground speed to the orbit speed (0.9 for this orbit).
    assert abs(range_term[1]) < 1e-3
    slant_range = slc.range_to_first_sample + col * slc.delta_range_sample
    orbit = snap._Orbit(slc)
    _, velocity = orbit(orbit.seconds(slc.center_time.as_numpy()))
    x = 2000 * slc.delta_line_time * np.linalg.norm(velocity)
    ratio = range_term[[0, 2]] / (x**2 / (2 * slant_range))
    assert np.all((ratio > 0.85) & (ratio < 1.0))
    assert ratio[0] == pytest.approx(ratio[1], rel=1e-3)


def test_spotlight_phase_model_matches_exact_phase():
    slc = _as_spotlight(CapellaSLC.from_file(STRIPMAP_JSON))
    rows = np.arange(1000, 1004)
    cols = np.arange(0, slc.shape[1], 501)

    model = snap._SpotlightPhaseModel(slc)(1000, 1004)[:, ::501]
    exact = snap.spotlight_phase(slc, rows[:, None], cols[None, :])

    np.testing.assert_allclose(model, exact, atol=0.01)


def test_spotlight_phase_needs_reference_positions():
    slc = _as_spotlight(CapellaSLC.from_file(STRIPMAP_JSON))
    image = slc.collect.image.model_copy(update={"reference_antenna_position": None})
    collect = slc.collect.model_copy(update={"image": image})
    slc = slc.model_copy(
        update={"meta": slc.meta.model_copy(update={"collect": collect})}
    )

    with pytest.raises(ValueError, match="reference_antenna_position"):
        snap.spotlight_phase(slc, 0, 0)


def test_pfa_geometry_is_not_supported():
    slc = CapellaSLC.from_file(PFA_JSON)
    with pytest.raises(CapellaImageGeometryError):
        snap.abstracted_metadata(slc)


def test_write_dimap_needs_the_geotiff(tmp_path):
    slc = CapellaSLC.from_file(STRIPMAP_JSON)
    with pytest.raises(ValueError, match="GeoTIFF"):
        snap.write_dimap(slc, tmp_path / "out.dim")


def _read_band(header: Path, name: str) -> np.ndarray:
    data_dir = header.with_suffix(".data")
    return np.fromfile(data_dir / f"{name}.img", dtype=">f4").reshape(SMALL_SHAPE)


def test_write_dimap_stripmap(tmp_path):
    tif, data = _write_small_tiff(STRIPMAP_JSON, tmp_path)
    slc = CapellaSLC.from_file(tif)

    header = snap.write_dimap(slc, tmp_path / "product")

    assert header == tmp_path / "product.dim"
    written = _read_band(header, "i_HH") + 1j * _read_band(header, "q_HH")
    np.testing.assert_allclose(
        written, data * slc.collect.image.scale_factor, rtol=1e-6
    )

    root = ET.parse(header).getroot()
    assert root.findtext("Raster_Dimensions/NCOLS") == str(SMALL_SHAPE[1])
    assert root.findtext("Raster_Dimensions/NROWS") == str(SMALL_SHAPE[0])
    bands = [e.text for e in root.iter("BAND_NAME")]
    assert bands == ["i_HH", "q_HH", "Intensity_HH"]
    grids = [e.text for e in root.iter("TIE_POINT_GRID_NAME")]
    assert grids == ["latitude", "longitude", "incident_angle", "slant_range_time"]

    abstracted = next(
        e for e in root.iter("MDElem") if e.get("name") == "Abstracted_Metadata"
    )
    attributes = {e.get("name"): e.text for e in abstracted.findall("MDATTR")}
    assert attributes["MISSION"] == "Capella"
    assert attributes["SAMPLE_TYPE"] == "COMPLEX"
    assert float(attributes["line_time_interval"]) == slc.delta_line_time
    assert float(attributes["slant_range_to_first_pixel"]) == slc.range_to_first_sample
    # Attributes without a Capella value get the SNAP "no metadata" value
    assert attributes["REL_ORBIT"] == "99999"
    elements = {e.get("name"): e for e in abstracted.findall("MDElem")}
    n_vectors = len(slc.collect.state.state_vectors)
    assert len(elements["Orbit_State_Vectors"]) == n_vectors
    # SNAP stops with an error if there are no Doppler centroid coefficients
    assert len(elements["Doppler_Centroid_Coefficients"]) == 1

    incidence = np.fromfile(
        header.with_suffix(".data") / "tie_point_grids" / "incident_angle.img",
        dtype=">f4",
    ).reshape(21, 21)
    assert np.all(np.diff(incidence, axis=1) > 0)


def test_write_dimap_spotlight_applies_the_phase(tmp_path):
    tif, data = _write_small_tiff(STRIPMAP_JSON, tmp_path)
    slc = _as_spotlight(CapellaSLC.from_file(tif))
    rows = np.arange(SMALL_SHAPE[0])[:, None]
    cols = np.arange(SMALL_SHAPE[1])[None, :]
    expected = (
        data
        * slc.collect.image.scale_factor
        * np.exp(1j * snap.spotlight_phase(slc, rows, cols))
    )

    converted = snap.write_dimap(slc, tmp_path / "converted")
    unchanged = snap.write_dimap(slc, tmp_path / "unchanged", zero_doppler_phase=False)

    written = _read_band(converted, "i_HH") + 1j * _read_band(converted, "q_HH")
    np.testing.assert_allclose(written, expected, atol=2e-3 * np.abs(expected).max())
    written = _read_band(unchanged, "i_HH") + 1j * _read_band(unchanged, "q_HH")
    np.testing.assert_allclose(
        written, data * slc.collect.image.scale_factor, rtol=1e-6
    )
    assert "zero_doppler_phase_applied" in converted.read_text()


def test_write_dimap_reads_striped_tiff(tmp_path):
    tif, data = _write_small_tiff(STRIPMAP_JSON, tmp_path, tiled=False)
    slc = CapellaSLC.from_file(tif)

    header = snap.write_dimap(slc, tmp_path / "product", block_rows=100)

    written = _read_band(header, "i_HH") + 1j * _read_band(header, "q_HH")
    np.testing.assert_allclose(
        written, data * slc.collect.image.scale_factor, rtol=1e-6
    )


def test_cli_to_dimap(tmp_path, capsys):
    tif, _ = _write_small_tiff(STRIPMAP_JSON, tmp_path)
    output = tmp_path / "from_cli.dim"

    assert cli.main(["to-dimap", str(tif), str(output)]) == 0

    assert capsys.readouterr().out.strip() == str(output)
    assert output.exists()
    assert (tmp_path / "from_cli.data" / "i_HH.img").exists()
