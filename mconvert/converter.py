"""Facade over the vendored structure-conversion engine.

This is the ONLY module allowed to import mconvert.engine. main_window.py and
convert_worker.py talk to the engine exclusively through the functions and
types defined here. That keeps a single seam to update if the engine's
internals change (a new mkits sync), if its exception types change, or if a
different backend is ever swapped in.
"""

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from mconvert import formats
from mconvert.brillouin import brillouin_zone_edges, points_outside_zone
from mconvert.engine import database
from mconvert.engine.functions import MkitsError
from mconvert.engine.structure import seekpath as _EngineSeekpath
from mconvert.engine.structure import struct as _EngineStruct


class ConversionError(Exception):
    """A known, user-facing error (unsupported format, malformed input file,
    failed write, ...). Safe to show to the user as plain text.

    Any other exception that escapes this module is an unexpected bug in the
    engine and should be surfaced with its traceback instead of swallowed.
    """


@dataclass(frozen=True)
class StructureSummary:
    """GUI-facing summary of a loaded structure. Never exposes engine internals
    (numpy arrays, the struct object itself) so the GUI has nothing to depend
    on beyond this module."""
    detected_format: str
    total_atoms: int
    chemical_formula: str
    lattice_lengths: Tuple[float, float, float]   # a, b, c in angstrom
    lattice_angles: Tuple[float, float, float]    # alpha, beta, gamma in degrees


class LoadedStructure:
    """Opaque handle returned by load_structure().

    Pass it back into preview()/convert_to_file(); it is not meant to be
    inspected from GUI code beyond reading .summary.
    """

    def __init__(self, engine_struct, summary: StructureSummary, path: str):
        self._engine_struct = engine_struct
        self.summary = summary
        self.path = path


def _chemical_formula(engine_struct) -> str:
    """Build a simple 'Symbol+count' formula, atoms ordered by atomic number
    (mirrors the ordering the engine itself uses when writing POSCAR/CIF)."""
    atomic_indices = [int(z) for z in engine_struct.position[1:, 0]]
    atomic_types = sorted(set(atomic_indices))
    parts = []
    for z in atomic_types:
        count = atomic_indices.count(z)
        symbol = database.atom_data[z][1]
        parts.append(symbol if count == 1 else "%s%d" % (symbol, count))
    return "".join(parts)


def _check_output_format(target_format: str) -> None:
    if target_format not in formats.OUTPUT_FORMATS:
        raise ConversionError(
            "Unsupported output format: %r (supported: %s)"
            % (target_format, ", ".join(formats.OUTPUT_FORMATS))
        )


def load_structure(path: str) -> LoadedStructure:
    """Parse a structure file, auto-detecting its format.

    Raises ConversionError with a user-facing message if the file cannot be
    read or parsed.
    """
    try:
        engine_struct = _EngineStruct(path)
    except MkitsError as err:
        raise ConversionError(str(err)) from err

    summary = StructureSummary(
        detected_format=engine_struct.calculator,
        total_atoms=engine_struct.total_atom,
        chemical_formula=_chemical_formula(engine_struct),
        lattice_lengths=tuple(float(v) for v in engine_struct.lattice6[:3]),
        lattice_angles=tuple(float(v) for v in engine_struct.lattice6[3:]),
    )
    return LoadedStructure(engine_struct, summary, path)


def preview(loaded: LoadedStructure, target_format: str, frac: bool = True, dyn: bool = False) -> str:
    """Return the text mconvert would write for target_format, without touching disk."""
    _check_output_format(target_format)
    try:
        _, lines = loaded._engine_struct.write_struct(
            calculator=target_format, dyn=dyn, frac=frac, write2file=False,
        )
    except MkitsError as err:
        raise ConversionError(str(err)) from err
    return "".join(lines)


def convert_to_file(loaded: LoadedStructure, target_format: str, output_path: str,
                     frac: bool = True, dyn: bool = False) -> None:
    """Write the converted structure to output_path (a full file path)."""
    _check_output_format(target_format)
    fpath, fname = os.path.split(output_path)
    try:
        loaded._engine_struct.write_struct(
            fpath=fpath or ".", fname=fname, calculator=target_format,
            dyn=dyn, frac=frac, write2file=True,
        )
    except MkitsError as err:
        raise ConversionError(str(err)) from err
    except OSError as err:
        raise ConversionError("Failed to write output file: %s" % err) from err


# ---------------------------------------------------------------------- #
# 3D scene data for the real/reciprocal-space viewer
#
# Plain numpy arrays and frozen dataclasses only: the viewer never touches
# the engine's struct object or its position array layout.
# ---------------------------------------------------------------------- #
_CELL_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)
_CELL_EDGE_INDICES = tuple(
    (a, b) for a in range(8) for b in range(a + 1, 8)
    if np.count_nonzero(_CELL_CORNERS[a] != _CELL_CORNERS[b]) == 1
)

_KPOINT_DISPLAY_LABELS = {"GAMMA": "\u0393", "SIGMA": "\u03a3", "SIGMA1": "\u03a31"}


@dataclass(frozen=True)
class AtomInfo:
    index: int                      # 1-based position in the structure
    symbol: str
    name: str
    atomic_number: int
    atomic_mass: float              # amu, nan if unknown
    frac: Tuple[float, float, float]
    cart: Tuple[float, float, float]   # angstrom


@dataclass(frozen=True)
class RealSpaceScene:
    atoms: Tuple[AtomInfo, ...]
    lattice: np.ndarray             # (3, 3), rows a1, a2, a3 in angstrom
    cell_edges: np.ndarray          # (12, 2, 3) unit-cell edges, Cartesian


@dataclass(frozen=True)
class KPointInfo:
    label: str                      # display label, eg the Greek capital gamma
    key: str                        # raw label from the k-path table, eg "GAMMA"
    frac: Tuple[float, float, float]   # coordinates w.r.t. b1, b2, b3
    cart: Tuple[float, float, float]   # 1/angstrom, including the 2*pi factor


@dataclass(frozen=True)
class ReciprocalScene:
    rec_vectors: np.ndarray         # (3, 3), rows b1, b2, b3 in 1/angstrom, with 2*pi
    zone_edges: np.ndarray          # (N, 2, 3) first Brillouin zone edges, Cartesian
    kpoints: Tuple[KPointInfo, ...]
    kpath_segments: Tuple[Tuple[str, ...], ...]   # sequences of KPointInfo.key
    lattice_type: str
    is_2d: bool
    points_outside_zone: bool       # True when the structure is not a primitive cell: its
                                    # high-symmetry points are those of the primitive cell
                                    # and lie outside this (smaller) cell's zone


def get_real_space_scene(loaded: LoadedStructure) -> RealSpaceScene:
    """Atoms and the unit cell of a loaded structure, ready to draw."""
    engine_struct = loaded._engine_struct
    lattice = np.array(engine_struct.lattice9, dtype=float)

    atoms = []
    for number, row in enumerate(engine_struct.position[1:], start=1):
        z = int(row[0])
        _, symbol, name, mass = database.atom_data[z][:4]
        atoms.append(AtomInfo(
            index=number, symbol=symbol, name=name, atomic_number=z,
            atomic_mass=float("nan") if mass is None else float(mass),
            frac=tuple(float(v) for v in row[1:4]),
            cart=tuple(float(v) for v in row[4:7]),
        ))

    corners = _CELL_CORNERS @ lattice
    cell_edges = np.array([[corners[a], corners[b]] for a, b in _CELL_EDGE_INDICES])
    return RealSpaceScene(atoms=tuple(atoms), lattice=lattice, cell_edges=cell_edges)


def get_reciprocal_scene(loaded: LoadedStructure) -> ReciprocalScene:
    """
    Reciprocal basis, first Brillouin zone and high-symmetry points.

    3D high-symmetry points follow the Setyawan-Curtarolo convention and are
    expressed in this structure's own reciprocal basis (see the engine's
    seekpath docstring); for a non-primitive cell they are the primitive
    cell's points seen through the smaller zone, flagged by
    ReciprocalScene.points_outside_zone. 2D paths are still a first
    approximation.
    """
    try:
        path = _EngineSeekpath(loaded.path)
    except MkitsError as err:
        raise ConversionError(str(err)) from err

    rec_vectors = np.array(path.get_rec_basis(with_2pi=True), dtype=float)
    kpoints = []
    for key, frac in path.kpath_points.items():
        frac = np.array(frac, dtype=float)
        cart = frac @ rec_vectors
        kpoints.append(KPointInfo(
            label=_KPOINT_DISPLAY_LABELS.get(key, key), key=key,
            frac=tuple(float(v) for v in frac), cart=tuple(float(v) for v in cart),
        ))

    is_2d = bool(path.is2d)
    # The engine classifies the lattice within spglib's symprec, so a slightly
    # distorted cell (a relaxed structure, a rounded POSCAR) still gets the
    # ideal-lattice points, which can poke a hair outside its zone. Genuine
    # folding from a non-primitive cell overshoots by tens of percent of |b|,
    # so a 2 % margin separates the two.
    outside = points_outside_zone([k.cart for k in kpoints], rec_vectors, is_2d=is_2d, tol=0.02)
    return ReciprocalScene(
        rec_vectors=rec_vectors,
        zone_edges=brillouin_zone_edges(rec_vectors, is_2d=is_2d),
        kpoints=tuple(kpoints),
        kpath_segments=tuple(tuple(segment) for segment in path.kpath_segments),
        lattice_type=path.lattice_type,
        is_2d=is_2d,
        points_outside_zone=bool(np.any(outside)),
    )
