"""Capability table describing the structure formats mconvert can read/write.

The engine (mconvert.engine.structure.struct) defines which formats actually
exist and what "calculator" key each one uses; this module only describes how
the GUI should present them (label, file-dialog filter, whether the
fractional/Cartesian and selective-dynamics options apply, and a sensible
default output filename). Adding a new format touches one row here plus
whatever engine change makes the format work, instead of several parallel
dicts that can drift out of sync.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class FormatSpec:
    key: str                        # calculator key passed to struct()/write_struct()
    label: str                      # shown in combo boxes
    extensions: Tuple[str, ...]     # filename patterns, for file-dialog filters
    readable: bool                  # struct() can auto-detect and parse this format
    writable: bool                  # write_struct(calculator=key) can produce this format
    supports_frac_cart: bool        # the "fractional / Cartesian" option applies
    supports_dyn: bool              # the "selective dynamics" option applies
    default_filename: Optional[str] # suggested filename for a single converted file
    batch_extension: Optional[str]  # extension used when naming many converted files
                                     # (stem + batch_extension); differs from
                                     # default_filename for formats like poscar whose
                                     # single-file convention is a fixed bare name


# POSCAR/CONTCAR and LAMMPS data files conventionally carry no extension at
# all; the engine identifies every format from file content, not from the
# name, so extensions here are only a dialog convenience, never a hard filter.
FORMATS = (
    FormatSpec("poscar", "POSCAR (VASP)", ("POSCAR*", "CONTCAR*", "*.vasp"),
               True, True, True, True, "POSCAR", ".vasp"),
    FormatSpec("cif", "CIF", ("*.cif",),
               True, True, False, False, "structure.cif", ".cif"),
    FormatSpec("xsd", "XSD (Materials Studio)", ("*.xsd",),
               True, False, False, False, None, None),
    FormatSpec("xyz", "XYZ", ("*.xyz",),
               True, True, False, False, "structure.xyz", ".xyz"),
    FormatSpec("qein", "Quantum ESPRESSO input (qein)", ("*.in", "*.pwi"),
               True, True, True, True, "pwscf.in", ".in"),
    FormatSpec("qeout", "Quantum ESPRESSO output (qeout)", ("*.out", "*.pwo"),
               True, False, False, False, None, None),
    FormatSpec("lammps", "LAMMPS data", ("*.lmp", "*.data"),
               True, True, False, False, "structure.lmp", ".lmp"),
    FormatSpec("wien2k", "WIEN2k struct", ("*.struct",),
               True, True, False, False, "case.struct", ".struct"),
)

_BY_KEY = {spec.key: spec for spec in FORMATS}

INPUT_FORMATS = tuple(spec.key for spec in FORMATS if spec.readable)
OUTPUT_FORMATS = tuple(spec.key for spec in FORMATS if spec.writable)


def get(key: str) -> FormatSpec:
    """Look up a FormatSpec by its engine calculator key."""
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError("Unknown structure format key: %r" % key) from None


def supports_frac_cart(key: str) -> bool:
    """Whether the fractional/Cartesian coordinate choice applies to this format."""
    return get(key).supports_frac_cart


def supports_dyn(key: str) -> bool:
    """Whether the selective-dynamics option applies to this format."""
    return get(key).supports_dyn


def default_filename(key: str) -> str:
    """The suggested output filename for a writable format."""
    spec = get(key)
    if spec.default_filename is None:
        raise ValueError("Format %r has no default output filename (not writable)" % key)
    return spec.default_filename


def batch_extension(key: str) -> str:
    """The extension to append to a preserved filename stem in batch mode."""
    spec = get(key)
    if spec.batch_extension is None:
        raise ValueError("Format %r has no batch extension (not writable)" % key)
    return spec.batch_extension


def label(key: str) -> str:
    """The combo-box display label for a format."""
    return get(key).label


def dialog_filter(key: str) -> str:
    """A Qt file-dialog filter string for one format, e.g. 'CIF (*.cif)'."""
    spec = get(key)
    return "%s (%s)" % (spec.label, " ".join(spec.extensions))


def all_input_dialog_filter() -> str:
    """
    The filter string for the "choose input file" dialog.

    "All files" is listed first on purpose: formats are identified by content,
    not extension, and several of them (POSCAR, LAMMPS data) commonly have no
    extension at all, so a strict extension filter would hide exactly the
    files a user needs to pick.
    """
    per_format = ";;".join(dialog_filter(spec.key) for spec in FORMATS if spec.readable)
    return "All files (*);;%s" % per_format
