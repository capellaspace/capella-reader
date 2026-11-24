"""Command-line tools for Capella SLC metadata inspection and conversion."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from capella_reader.adapters import snap
from capella_reader.slc import CapellaSLC


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capella-reader",
        description="Inspect Capella SLC metadata and derived properties.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump_parser = subparsers.add_parser(
        "dump",
        help="Pretty-print metadata using rich.",
    )
    dump_parser.add_argument("path", help="Path to a Capella SLC GeoTIFF or JSON file.")
    dump_parser.add_argument(
        "--full",
        action="store_true",
        help="Include long attitude and orbit entries.",
    )

    bounds_parser = subparsers.add_parser(
        "bounds",
        help="Print lon/lat approximate bounds from GCPs.",
    )
    bounds_parser.add_argument(
        "path", help="Path to a Capella SLC GeoTIFF or JSON file."
    )
    bounds_parser.add_argument(
        "--json",
        action="store_true",
        help="Print bounds as a JSON array.",
    )

    dimap_parser = subparsers.add_parser(
        "to-dimap",
        help="Convert an SLC GeoTIFF to a BEAM-DIMAP product for ESA SNAP.",
    )
    dimap_parser.add_argument("path", help="Path to a Capella SLC GeoTIFF file.")
    dimap_parser.add_argument(
        "output", help="Path of the output .dim header (data go to <name>.data)."
    )
    dimap_parser.add_argument(
        "--keep-spotlight-phase",
        action="store_true",
        help=(
            "Do not convert the phase of a spotlight SLC to the zero-Doppler "
            "convention. SNAP interferograms need the conversion."
        ),
    )

    return parser


def _suppress_long_entries(meta_dict: dict[str, Any]) -> dict[str, Any]:
    collect = meta_dict.get("collect")
    if not isinstance(collect, dict):
        return meta_dict

    pointing = collect.get("pointing")
    if isinstance(pointing, list):
        collect["pointing"] = f"<{len(pointing)} attitude samples; use --full>"

    state = collect.get("state")
    if isinstance(state, dict):
        state_vectors = state.get("state_vectors")
        if isinstance(state_vectors, list):
            state["state_vectors"] = f"<{len(state_vectors)} state vectors; use --full>"

    if "transmit_antenna" in collect:
        collect["transmit_antenna"] = "<suppressed; use --full>"
    if "receive_antenna" in collect:
        collect["receive_antenna"] = "<suppressed; use --full>"

    return meta_dict


def _dump(path: str, full: bool) -> int:
    slc = CapellaSLC.from_file(path)
    data = slc.meta.model_dump()
    if not full:
        data = _suppress_long_entries(data)
    print(json.dumps(data, indent=2, default=str))
    return 0


def _bounds(path: str, as_json: bool) -> int:
    slc = CapellaSLC.from_file(path)
    bounds = slc.bounds
    if as_json:
        print(json.dumps(bounds, indent=2))
    else:
        print(f"{bounds[0]} {bounds[1]} {bounds[2]} {bounds[3]}")
    return 0


def _to_dimap(path: str, output: str, keep_spotlight_phase: bool) -> int:
    slc = CapellaSLC.from_file(path)
    header = snap.write_dimap(slc, output, zero_doppler_phase=not keep_spotlight_phase)
    print(header)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Capella reader CLI.

    Parameters
    ----------
    argv
        Optional argument sequence for testing. Uses `sys.argv` when omitted.

    Returns
    -------
    int
        Process exit code.

    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "dump":
        return _dump(args.path, args.full)
    if args.command == "to-dimap":
        return _to_dimap(args.path, args.output, args.keep_spotlight_phase)
    return _bounds(args.path, args.json)
