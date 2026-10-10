# -*- coding: utf-8 -*-
"""Atomic structure parsing and manipulation for common DFT input/output formats."""

import re
import xml.etree.ElementTree as ET

import numpy as np
import spglib as spg

from . import database
from . import functions


def _spglib_field(dataset, key):
    """
    Read a field from a spglib symmetry dataset regardless of whether the
    installed spglib version returns an object with attribute access
    (>=2.5) or a plain dict (<2.5, still supported via a deprecated
    __getitem__ shim on newer versions).
    """
    if dataset is None:
        return None
    try:
        return getattr(dataset, key)
    except AttributeError:
        return dataset[key]


def _qeout_cell_block(lines, header_idx, latticeunit):
    """
    Parse one QE "CELL_PARAMETERS ..." block at lines[header_idx] into a
    (3,3) angstrom lattice matrix, honoring its own unit tag: "angstrom"
    (as-is), "bohr" (* uc_bohr2ang), or no tag at all (pw.x's own default
    for this card, dimensionless multiples of `latticeunit`, the alat in
    bohr). Shared by struct's own qeout parser (last block only) and
    mkits.structure.trajectory's QE reader (every block).
    """
    _tag = lines[header_idx]
    _raw = np.array(
        [[float(_v) for _v in lines[header_idx + 1 + _row].split()[:3]] for _row in range(3)]
    )
    if "angstrom" in _tag:
        return _raw
    if "bohr" in _tag:
        return _raw * database.uc_bohr2ang
    return _raw * latticeunit * database.uc_bohr2ang


def _qeout_positions_block(lines, header_idx, nat):
    """
    Parse one QE "ATOMIC_POSITIONS ..." block at lines[header_idx] into
    (symbols, coords, is_crystal) -- `coords` is the raw (nat,3) array,
    already converted to angstrom if the block's tag says "bohr" (its
    fractional-vs-Cartesian meaning is otherwise left to the caller, via
    `is_crystal`: True for "crystal" -- fractional -- False for a
    Cartesian block, "bohr" or "angstrom" (QE's own default when no tag is
    given)). Shared by struct's own qeout parser (last block only) and
    mkits.structure.trajectory's QE reader (every block).
    """
    _is_crystal = "crystal" in lines[header_idx]
    _is_bohr = "bohr" in lines[header_idx]
    _symbols = []
    _coords = []
    for _line in lines[header_idx + 1: header_idx + 1 + nat]:
        _parts = _line.split()
        _symbols.append(_parts[0])
        _coords.append([float(_v) for _v in _parts[1:4]])
    _coords = np.array(_coords)
    if _is_bohr:
        _coords = _coords * database.uc_bohr2ang
    return _symbols, _coords, _is_crystal


def _read_xdatcar_frames(lines):
    """
    Bulk-read every configuration from a fixed-cell VASP XDATCAR (comment,
    scale, 3 lattice vectors, species line, species-count line, then one
    "Direct configuration=<n>" block of fractional coordinates per step).
    Shared by struct's own "xdatcar" format (keeps only the last
    configuration, matching __parse_qeout's "final geometry" convention)
    and mkits.structure.trajectory (keeps all of them).

    Return
    ------
    (lattice, species, positions_frac): lattice is (3,3) angstrom --
        assumed constant across the trajectory (a variable-cell run's
        XDATCAR repeats a lattice block before every configuration
        instead of writing it once up front, which is not handled here);
        species is a list of element symbols, one per ion (species-block
        order); positions_frac is (nstep, natom, 3).
    """
    _scale = float(lines[1].split()[0])
    _lattice = np.array([[float(_v) for _v in lines[_i].split()] for _i in range(2, 5)]) * _scale
    _symbols = lines[5].split()
    _counts = [int(_v) for _v in lines[6].split()]
    _natom = sum(_counts)
    _species = [_s for _s, _n in zip(_symbols, _counts) for _ in range(_n)]

    _block_lineno = [_i for _i, _l in enumerate(lines) if _l.strip().startswith("Direct configuration")]
    if not _block_lineno:
        raise functions.MkitsError(
            "No 'Direct configuration=' blocks found -- expected a fixed-cell AIMD/relaxation XDATCAR."
        )

    _positions = np.zeros((len(_block_lineno), _natom, 3))
    for _t, _start in enumerate(_block_lineno):
        _block = np.loadtxt(lines[_start + 1: _start + 1 + _natom])
        _positions[_t] = _block[:, :3]

    return _lattice, _species, _positions


class struct(object):
    """
    Structure object.

    Supported formats
    ------------------
    poscar, cif, xsd, xyz, qein, qeout, xdatcar, lammps, wien2k
    (legacy VASP 4.x POSCARs without a species line are not supported)

    Attributes
    ----------
    inp
        The name of the input file, or a dictionary describing the structure.
    calculator: str
        Detected input format.
    title: str
        Structure title/comment line.
    total_atom: int
        Total number of atoms.
    lattice9: 3x3 numpy array
        Cartesian lattice vectors [[ax,ay,az], [bx,by,bz], [cx,cy,cz]].
    lattice6: 6x1 numpy array
        Direct lattice parameters [a, b, c, alpha, beta, gamma].
    position: (n+1)x11 numpy array
        [atomic_index, frac_x, _y, _z, cart_x, _y, _z, dyn_x, _y, _z, magmom]
        Row 0 is an internal buffer row; real atoms start at row 1.

    Functions
    ---------
    sort_atoms, add_dyn, set_magmom, scale_lattice9, write_struct,
    get_rec_basis, add_atom, replace_atom, supercell, transit_axis,
    get_spacegroup, get_symmetry, get_primitive_cell
    """

    def __init__(self, inp):
        """
        :param self: Initialize the class
        :param inp: The input of structures: a file path, a structure dict,
                    or the literal string "none" for an empty structure.
        """
        self.inp = inp
        self.calculator = "none"
        self.title = "none"
        self.lattice9 = np.array([[]])
        self.lattice6 = np.array([])
        self.position = np.zeros([1, 11])
        self.total_atom = 0

        if isinstance(inp, dict):
            self.calculator = inp["calculator"]
        elif isinstance(inp, str) and inp != "none" and "\n" in inp:
            raise functions.MkitsError("Parsing a structure directly from a string is not supported yet.")
        elif inp == "none":
            pass
        else:
            try:
                with open(inp, "r") as f:
                    _struct_lines = f.readlines()
            except OSError as e:
                raise functions.MkitsError("Cannot open the structure file %s: %s" % (inp, e))
            self.__parse_calculator(_struct_lines)
            self.__parse_structure(_struct_lines)
            self.sort_atoms()

    # ================================================================== #
    # format detection
    # ================================================================== #
    def __parse_calculator(self, lines):
        """
        Detect the structure format from the raw file content.

        Return
        ------
        A string: poscar, cif, xsd, xyz, qein, qeout, xdatcar, lammps, wien2k.
        """
        if "open-source Quantum ESPRESSO" in lines[3]:
            self.calculator = "qeout"
        elif "&SYSTEM" in "".join(lines) or "&system" in "".join(lines):
            self.calculator = "qein"
        elif "_cell_length_a" in "".join(lines):
            self.calculator = "cif"
        elif len(lines) > 1 and "LATTICE" in lines[1] and "NONEQUIV.ATOMS" in lines[1]:
            self.calculator = "wien2k"
        elif self.__is_lammps_data(lines):
            self.calculator = "lammps"
        elif any(_l.strip().startswith("Direct configuration") for _l in lines[:12]):
            # checked before the plain "poscar" branch below -- an XDATCAR's
            # header (comment/scale/lattice/species/counts) is byte-for-byte
            # identical to a POSCAR's own, only the "Direct configuration=<n>"
            # coordinate-block header distinguishes the two
            self.calculator = "xdatcar"
        elif len(lines[2].split()) == 3:
            self.calculator = "poscar"
        elif len(lines) >= 3 and lines[0].strip().isdigit() and len(lines[2].split()) >= 4:
            self.calculator = "xyz"
        elif "<?xml" in "".join(lines[:20]) or "AtomisticTreeRoot" in "".join(lines[:20]) \
                or "IdentityMapping" in "".join(lines):
            self.calculator = "xsd"
        else:
            raise functions.MkitsError(
                "Unsupported structure format, supported: poscar, cif, xsd, xyz, qein, qeout, "
                "xdatcar, lammps, wien2k."
            )

    @staticmethod
    def __is_lammps_data(lines):
        """Heuristically detect a LAMMPS data file by its header keywords."""
        _head = " ".join(lines[:30])
        return bool(re.search(r"\d+\s+atoms", _head)) and \
            bool(re.search(r"\d+\s+atom types", _head)) and \
            "xlo xhi" in _head

    # ================================================================== #
    # structure parsing
    # ================================================================== #
    def __parse_structure(self, lines):
        """Dispatch to the format-specific parser and populate self.position."""
        if self.calculator == "poscar":
            self.__parse_poscar(lines)
        elif self.calculator == "cif":
            self.__parse_cif(lines)
        elif self.calculator == "xsd":
            self.__parse_xsd(lines)
        elif self.calculator == "xyz":
            self.__parse_xyz(lines)
        elif self.calculator == "qein":
            self.__parse_qein(lines)
        elif self.calculator == "qeout":
            self.__parse_qeout(lines)
        elif self.calculator == "xdatcar":
            self.__parse_xdatcar(lines)
        elif self.calculator == "lammps":
            self.__parse_lammps(lines)
        elif self.calculator == "wien2k":
            self.__parse_wien2k(lines)

    def __default_magmom(self, atomic_indices):
        """Look up the default initial magnetic moment for a list of atomic numbers."""
        return np.array([[database.atom_data[int(_z)][4]] for _z in atomic_indices])

    # ------------------------------------------------------------------ #
    # POSCAR
    # ------------------------------------------------------------------ #
    def __parse_poscar(self, lines):
        """Parse a VASP POSCAR file (species-line format only)."""
        self.title = lines[0][:-1]
        _ratio = float(lines[1][:-1])
        _lattice_x = [float(v) for v in lines[2].split()]
        _lattice_y = [float(v) for v in lines[3].split()]
        _lattice_z = [float(v) for v in lines[4].split()]
        self.lattice9 = np.array([_lattice_x, _lattice_y, _lattice_z]) * _ratio
        self.lattice6 = functions.lattice_conversion(self.lattice9)

        # read elements from the 6th/7th line; legacy POSCARs (no species
        # line, VASP 4.x style) are not supported
        try:
            _atom_type = lines[5].split()
            _atom_num = np.array([int(v) for v in lines[6].split()])
            _atoms = functions.listcross(_atom_type, _atom_num)
            self.total_atom = int(np.sum(_atom_num))
        except (ValueError, IndexError):
            raise functions.MkitsError("POSCAR without a species line (VASP 4.x style) is not supported.")

        _atom_index = np.array([database.symbol_map[i] for i in _atoms]).reshape(-1, 1)

        if functions.rmspace(lines[7]).upper()[0] == "S" and len(lines[9].split()) >= 6:
            for _i in range(self.total_atom):
                lines[9 + _i] = lines[9 + _i].upper().replace("F", "0").replace("T", "1")
            _position = np.array([[float(v) for v in line.split()[:3]] for line in lines[9: 9 + self.total_atom]])
            _dyn = np.array([[float(v) for v in line.split()[3:6]] for line in lines[9: 9 + self.total_atom]])
            _coord_tag = lines[8]
        elif functions.rmspace(lines[7]).upper()[0] in ("D", "C"):
            _position = np.array([[float(v) for v in line.split()[:3]] for line in lines[8: 8 + self.total_atom]])
            _dyn = np.ones([self.total_atom, 3])
            _coord_tag = lines[7]
        else:
            raise functions.MkitsError("Cannot determine the selective-dynamics flag of the POSCAR file.")

        _magmom = self.__default_magmom(_atom_index.flatten())

        if functions.rmspace(_coord_tag).upper()[0] == "D":
            _cart = functions.frac2cart(self.lattice9, _position)
            _block = np.hstack((_atom_index, _position, _cart, _dyn, _magmom))
        elif functions.rmspace(_coord_tag).upper()[0] == "C":
            _frac = functions.cart2frac(self.lattice6, _position)
            _block = np.hstack((_atom_index, _frac, _position, _dyn, _magmom))
        else:
            raise functions.MkitsError("Cannot determine whether POSCAR coordinates are fractional or Cartesian.")

        self.position = np.vstack((self.position, _block))

    # ------------------------------------------------------------------ #
    # CIF
    # ------------------------------------------------------------------ #
    @staticmethod
    def __find_cif_loop(lines, required_headers):
        """
        Find a CIF loop_ block containing at least one of the required headers.

        Return
        ------
        (headers, first_data_line_index), or (None, None) if not found.
        """
        for i, line in enumerate(lines):
            if "loop_" in line:
                _headers = []
                j = i + 1
                while j < len(lines):
                    _stripped = lines[j].strip()
                    if _stripped.startswith("_"):
                        _headers.append(_stripped)
                        j += 1
                    elif _stripped.startswith("#") or _stripped == "":
                        j += 1
                    else:
                        break
                if any(h in _headers for h in required_headers):
                    return _headers, j
        return None, None

    @staticmethod
    def __read_cif_loop_rows(lines, start):
        """Yield the raw data rows of a CIF loop_ block starting at `start`."""
        for line in lines[start:]:
            _stripped = line.strip()
            if not _stripped:
                continue
            if _stripped.startswith("#") or _stripped.startswith("loop_") or _stripped.startswith("_"):
                break
            yield _stripped

    @staticmethod
    def __symop_str_to_matrix(op_str):
        """
        Convert a CIF-style symmetry operation string, eg "-x+1/2,y,z", into
        a (3x3 rotation, 3 translation) pair, without using eval().
        """
        _op_str = op_str.lower().replace("'", "").replace('"', "").strip()
        _parts = _op_str.split(",")
        _rot = np.zeros((3, 3))
        _trans = np.zeros(3)
        if len(_parts) != 3:
            return np.eye(3), np.zeros(3)

        _axis_index = {"x": 0, "y": 1, "z": 2}
        for _row, _expr in enumerate(_parts):
            _expr = _expr.replace(" ", "")
            for _term in re.findall(r"[+-]?[^+-]+", _expr):
                _sign = -1.0 if _term.startswith("-") else 1.0
                _term = _term.lstrip("+-")
                if not _term:
                    continue
                if _term[0] in _axis_index:
                    _rot[_row, _axis_index[_term[0]]] = _sign
                elif "/" in _term:
                    _num, _den = _term.split("/")
                    _trans[_row] += _sign * float(_num) / float(_den)
                else:
                    _trans[_row] += _sign * float(_term)
        return _rot, _trans

    @staticmethod
    def __symops_from_spacegroup_number(sg_number):
        """Look up symmetry operations for a space-group number via spglib's database."""
        for _hall_number in range(1, 531):
            _sg_type = spg.get_spacegroup_type(_hall_number)
            if _sg_type is not None and _spglib_field(_sg_type, "number") == sg_number:
                _dataset = spg.get_symmetry_from_database(_hall_number)
                return list(_spglib_field(_dataset, "rotations")), list(_spglib_field(_dataset, "translations"))
        return [np.eye(3)], [np.zeros(3)]

    @staticmethod
    def __expand_by_symmetry(frac_coords, symbols, rotations, translations, tol=1e-4):
        """
        Apply a set of symmetry operations to representative (asymmetric-unit)
        fractional coordinates and remove duplicate images, building the full
        list of atoms in the unit cell. Shared by the CIF and WIEN2k parsers.

        Parameters
        ----------
        frac_coords: (n, 3) array-like
        symbols: list of str, length n
        rotations: list of (3, 3) array-like
        translations: list of (3,) array-like

        Return
        ------
        (expanded_symbols, expanded_frac_coords)
        """
        _expanded_symbols = []
        _expanded_coords = []
        for _sym, _frac in zip(symbols, frac_coords):
            _site_coords = []
            for _rot, _trans in zip(rotations, translations):
                _new_pos = (np.array(_rot, dtype=float) @ np.array(_frac, dtype=float)
                            + np.array(_trans, dtype=float)) % 1.0
                _is_duplicate = False
                for _existing in _site_coords:
                    _diff = np.abs(_new_pos - _existing)
                    _diff = np.minimum(_diff, 1.0 - _diff)
                    if np.sum(_diff ** 2) < tol:
                        _is_duplicate = True
                        break
                if not _is_duplicate:
                    _site_coords.append(_new_pos)
            for _c in _site_coords:
                _expanded_symbols.append(_sym)
                _expanded_coords.append(_c)
        return _expanded_symbols, np.array(_expanded_coords)

    def __parse_cif(self, lines):
        """Parse a CIF file, expanding the asymmetric unit into the full atom list."""
        _params = {}
        _sg_number = None
        for line in lines:
            _line = line.strip()
            if _line.startswith("_cell_"):
                _parts = _line.split()
                if len(_parts) >= 2:
                    _params[_parts[0]] = float(re.sub(r"\(.+\)", "", _parts[1]))
            elif _line.lower().startswith("_symmetry_int_tables_number") or \
                    _line.lower().startswith("_space_group_it_number"):
                _parts = _line.split()
                if len(_parts) >= 2:
                    _sg_number = int(re.sub(r"\D", "", _parts[1]))

        _a = _params.get("_cell_length_a")
        _b = _params.get("_cell_length_b")
        _c = _params.get("_cell_length_c")
        _alpha = _params.get("_cell_angle_alpha", 90.0)
        _beta = _params.get("_cell_angle_beta", 90.0)
        _gamma = _params.get("_cell_angle_gamma", 90.0)
        if None in (_a, _b, _c):
            raise functions.MkitsError("CIF file is missing one of the cell length tags.")

        self.lattice6 = np.array([_a, _b, _c, _alpha, _beta, _gamma])
        self.lattice9 = functions.lattice_conversion(self.lattice6)

        # symmetry operations: explicit loop, else space-group number via
        # spglib, else P1
        _sym_headers, _sym_start = self.__find_cif_loop(
            lines, ["_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"]
        )
        if _sym_start is not None:
            _idx_op = _sym_headers.index(
                "_space_group_symop_operation_xyz" if "_space_group_symop_operation_xyz" in _sym_headers
                else "_symmetry_equiv_pos_as_xyz"
            )
            _rotations, _translations = [], []
            for _row in self.__read_cif_loop_rows(lines, _sym_start):
                if "'" in _row:
                    _op_str = _row.split("'")[1]
                else:
                    _parts = _row.split()
                    if len(_parts) <= _idx_op:
                        continue
                    _op_str = _parts[_idx_op]
                _rot, _trans = self.__symop_str_to_matrix(_op_str)
                _rotations.append(_rot)
                _translations.append(_trans)
        elif _sg_number and _sg_number > 1:
            _rotations, _translations = self.__symops_from_spacegroup_number(_sg_number)
        else:
            _rotations, _translations = [np.eye(3)], [np.zeros(3)]

        # asymmetric-unit atomic positions
        _loop_headers, _loop_start = self.__find_cif_loop(lines, ["_atom_site_fract_x"])
        if _loop_start is None:
            raise functions.MkitsError("CIF file does not contain an atomic coordinates loop.")

        _idx_x = _loop_headers.index("_atom_site_fract_x")
        _idx_y = _loop_headers.index("_atom_site_fract_y")
        _idx_z = _loop_headers.index("_atom_site_fract_z")
        if "_atom_site_type_symbol" in _loop_headers:
            _idx_sym = _loop_headers.index("_atom_site_type_symbol")
        elif "_atom_site_label" in _loop_headers:
            _idx_sym = _loop_headers.index("_atom_site_label")
        else:
            raise functions.MkitsError("CIF file does not contain an atom-symbol column.")

        _rep_symbols, _rep_coords = [], []
        for _row in self.__read_cif_loop_rows(lines, _loop_start):
            _parts = _row.split()
            if len(_parts) < len(_loop_headers):
                continue
            _sym = re.sub(r"[^a-zA-Z]", "", _parts[_idx_sym])
            try:
                _fx = float(re.sub(r"\(.+\)", "", _parts[_idx_x]))
                _fy = float(re.sub(r"\(.+\)", "", _parts[_idx_y]))
                _fz = float(re.sub(r"\(.+\)", "", _parts[_idx_z]))
            except ValueError:
                continue
            _rep_symbols.append(_sym)
            _rep_coords.append([_fx, _fy, _fz])

        _symbols, _frac_coords = self.__expand_by_symmetry(_rep_coords, _rep_symbols, _rotations, _translations)

        self.total_atom = len(_symbols)
        _atom_index = np.array([database.symbol_map[s] for s in _symbols]).reshape(-1, 1)
        _cart_coords = functions.frac2cart(self.lattice9, _frac_coords)
        _dyn = np.ones((self.total_atom, 3))
        _magmom = self.__default_magmom(_atom_index.flatten())

        _block = np.hstack((_atom_index, _frac_coords, _cart_coords, _dyn, _magmom))
        self.position = np.vstack((self.position, _block))

    # ------------------------------------------------------------------ #
    # XSD (Materials Studio)
    # ------------------------------------------------------------------ #
    def __parse_xsd(self, lines):
        """Parse a Materials Studio .xsd file (crystal or isolated molecule)."""
        try:
            _root = ET.fromstring("".join(lines))

            _space_group = None
            for _elem in _root.iter():
                if "SpaceGroup" in _elem.tag:
                    _space_group = _elem
                    break

            if _space_group is not None:
                try:
                    _vec_a = [float(v) for v in _space_group.attrib["AVector"].split(",")]
                    _vec_b = [float(v) for v in _space_group.attrib["BVector"].split(",")]
                    _vec_c = [float(v) for v in _space_group.attrib["CVector"].split(",")]
                    self.lattice9 = np.array([_vec_a, _vec_b, _vec_c])
                    self.lattice6 = functions.lattice_conversion(self.lattice9)
                except KeyError:
                    raise functions.MkitsError("XSD file has a SpaceGroup tag but is missing its lattice vectors.")
            else:
                # isolated molecule: no periodicity, use a large dummy box
                self.lattice9 = np.eye(3) * 100.0
                self.lattice6 = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])

            _atom_indices = []
            _cart_coords = []
            _found_atoms = False
            for _elem in _root.iter():
                if "Atom3d" in _elem.tag:
                    _found_atoms = True
                    _attrib = _elem.attrib

                    _sym = _attrib.get("Components") or _attrib.get("UserSymbol")
                    if not _sym:
                        _sym = re.sub(r"[^a-zA-Z]", "", _attrib.get("Name", ""))
                    if _sym and "," in _sym:
                        _sym = _sym.split(",")[0]

                    _atom_indices.append(database.symbol_map.get(_sym, 0))
                    if "XYZ" in _attrib:
                        _cart_coords.append([float(v) for v in _attrib["XYZ"].split(",")])
                    else:
                        _cart_coords.append([0.0, 0.0, 0.0])

            if not _found_atoms:
                raise functions.MkitsError("XSD file does not contain any Atom3d tags.")

            self.total_atom = len(_atom_indices)
            _atom_indices = np.array(_atom_indices).reshape(self.total_atom, 1)
            _cart_coords = np.array(_cart_coords)

            if _space_group is not None:
                _frac_coords = functions.cart2frac(self.lattice9, _cart_coords)
            else:
                _frac_coords = np.zeros((self.total_atom, 3))

            _dyn = np.ones((self.total_atom, 3))
            _magmom = self.__default_magmom(_atom_indices.flatten())

            _block = np.hstack((_atom_indices, _frac_coords, _cart_coords, _dyn, _magmom))
            self.position = np.vstack((self.position, _block))

        except functions.MkitsError:
            raise
        except Exception as e:
            raise functions.MkitsError("Failed to parse XSD file: %s" % e)

    # ------------------------------------------------------------------ #
    # XYZ
    # ------------------------------------------------------------------ #
    def __parse_xyz(self, lines):
        """Parse a plain XYZ file (no periodicity, Cartesian coordinates only)."""
        try:
            self.total_atom = int(lines[0].strip())
            self.title = lines[1].strip()

            # XYZ carries no lattice information; use a large dummy box
            self.lattice9 = np.eye(3) * 100.0
            self.lattice6 = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])

            _atom_indices = []
            _cart_coords = []
            for line in lines[2: 2 + self.total_atom]:
                _parts = line.split()
                if len(_parts) < 4:
                    continue
                _sym_raw = _parts[0]
                if _sym_raw.isdigit():
                    _idx = int(_sym_raw)
                else:
                    _sym = re.sub(r"[^a-zA-Z]", "", _sym_raw)
                    _idx = database.symbol_map.get(_sym, 0)
                _atom_indices.append(_idx)
                _cart_coords.append([float(_parts[1]), float(_parts[2]), float(_parts[3])])

            _n = len(_atom_indices)
            _atom_indices = np.array(_atom_indices).reshape(_n, 1)
            _cart_coords = np.array(_cart_coords)
            _frac_coords = np.zeros((_n, 3))
            _dyn = np.ones((_n, 3))
            _magmom = self.__default_magmom(_atom_indices.flatten())

            _block = np.hstack((_atom_indices, _frac_coords, _cart_coords, _dyn, _magmom))
            self.position = np.vstack((self.position, _block))

        except functions.MkitsError:
            raise
        except Exception as e:
            raise functions.MkitsError("Failed to parse XYZ file: %s" % e)

    # ------------------------------------------------------------------ #
    # Quantum ESPRESSO input/output
    # ------------------------------------------------------------------ #
    def __parse_qein_block(self, lines):
        """Split a QE input file into its &BLOCK ... / namelists and cards."""
        _qeinblock = {}
        _blockname = []
        _start_idx = []
        _end_idx = []
        for i, line in enumerate(lines):
            _stripped = functions.rmspace(line)
            if _stripped and _stripped[0] == "&":
                _blockname.append(_stripped[1:].rstrip("\n"))
                _start_idx.append(i)
        for i in _start_idx:
            for j in range(i, len(lines)):
                if functions.rmspace(lines[j]).startswith("/"):
                    _end_idx.append(j)
                    break
        for _name, _s, _e in zip(_blockname, _start_idx, _end_idx):
            _qeinblock[_name] = lines[_s + 1: _e]

        if "SYSTEM" not in _qeinblock:
            raise functions.MkitsError("QE input file does not contain an &SYSTEM namelist.")

        _system_block = functions.parse_inputfile(
            lines=_qeinblock["SYSTEM"], comment_sym="!", assign_sym="=", seprate_sym=","
        )
        try:
            _tot_atom = int(_system_block["nat"])
        except KeyError:
            raise functions.MkitsError("The QE input file does not define 'nat' in &SYSTEM.")

        for i, line in enumerate(lines):
            _upper = line.upper()
            if "ATOMIC_POSITIONS" in _upper:
                _qeinblock["ATOMIC_POSITIONS"] = lines[i: i + _tot_atom + 1]
            elif "K_POINTS" in _upper:
                _qeinblock["K_POINTS"] = lines[i: i + 2]
            elif "CELL_PARAMETERS" in _upper:
                _qeinblock["CELL_PARAMETERS"] = lines[i: i + 4]

        return _qeinblock

    def __parse_qein(self, lines):
        """Parse a Quantum ESPRESSO pw.x input file."""
        _block = self.__parse_qein_block(lines)

        _lattice_lines = _block["CELL_PARAMETERS"]
        self.lattice9 = np.array([[float(v) for v in line.split()] for line in _lattice_lines[1:]])
        if "bohr" in _lattice_lines[0]:
            self.lattice9 *= database.uc_bohr2ang
        self.lattice6 = functions.lattice_conversion(self.lattice9)

        _position_lines = _block["ATOMIC_POSITIONS"]
        self.total_atom = len(_position_lines) - 1

        _symbols = [line.split()[0] for line in _position_lines[1:]]
        _atom_index = np.array([database.symbol_map[s] for s in _symbols]).reshape(-1, 1)

        _ncol = len(_position_lines[1].split())
        if _ncol == 4:
            _coords = np.array([[float(v) for v in line.split()[1:4]] for line in _position_lines[1:]])
            _dyn = np.ones((self.total_atom, 3))
        elif _ncol == 7:
            _coords = np.array([[float(v) for v in line.split()[1:4]] for line in _position_lines[1:]])
            _dyn = np.array([[float(v) for v in line.split()[4:7]] for line in _position_lines[1:]])
        else:
            raise functions.MkitsError("QE ATOMIC_POSITIONS block must have 4 or 7 columns per line.")

        _tag = _position_lines[0]
        if "crystal" in _tag:
            _frac = _coords
            _cart = functions.frac2cart(self.lattice9, _frac)
        elif "bohr" in _tag:
            _cart = _coords * database.uc_bohr2ang
            _frac = functions.cart2frac(self.lattice6, _cart)
        else:
            # angstrom is QE's default unit when no keyword is given
            _cart = _coords
            _frac = functions.cart2frac(self.lattice6, _cart)

        _magmom = self.__default_magmom(_atom_index.flatten())
        _block2 = np.hstack((_atom_index, _frac, _cart, _dyn, _magmom))
        self.position = np.vstack((self.position, _block2))

    def __parse_qeout(self, lines):
        """Parse the final geometry (lattice + atomic positions) from a QE output file."""
        _latticeunit = 1.0
        _nat = 0
        _crystal_axes_idx = 0
        _cartesian_axes_idx = 0
        _atomic_position_idx = []
        _cell_parameter_idx = []

        for i, line in enumerate(lines):
            if "lattice parameter" in line:
                _latticeunit = float(line.split()[-2])
            elif "number of atoms/cell" in line:
                _nat = int(line.split()[-1])
            elif "crystal axes" in line:
                _crystal_axes_idx = i
            elif "Cartesian axes" in line:
                _cartesian_axes_idx = i
            elif "ATOMIC_POSITIONS" in line:
                _atomic_position_idx.append(i)
            elif "CELL_PARAMETERS" in line:
                _cell_parameter_idx.append(i)

        self.total_atom = _nat

        # lattice: prefer the last relaxed CELL_PARAMETERS block, otherwise
        # fall back to the initial crystal axes block (units of alat, bohr)
        if _cell_parameter_idx:
            self.lattice9 = _qeout_cell_block(lines, _cell_parameter_idx[-1], _latticeunit)
        else:
            self.lattice9 = np.array(
                [[float(v) for v in lines[_crystal_axes_idx + 1 + row].split()[3:6]] for row in range(3)]
            ) * _latticeunit * database.uc_bohr2ang
        self.lattice6 = functions.lattice_conversion(self.lattice9)

        # atomic positions: prefer the last relaxed ATOMIC_POSITIONS block,
        # otherwise the initial Cartesian axes block (units of alat, bohr)
        _symbols = []
        _coords = []
        if _atomic_position_idx:
            _symbols, _coords, _is_crystal = _qeout_positions_block(lines, _atomic_position_idx[-1], _nat)
            if _is_crystal:
                _frac = _coords
                _cart = functions.frac2cart(self.lattice9, _frac)
            else:
                _cart = _coords
                _frac = functions.cart2frac(self.lattice6, _cart)
        else:
            for line in lines[_cartesian_axes_idx + 3: _cartesian_axes_idx + 3 + _nat]:
                _parts = line.split()
                _symbols.append(_parts[1])
                _coords.append([float(v) for v in _parts[6:9]])
            _cart = np.array(_coords) * _latticeunit * database.uc_bohr2ang
            _frac = functions.cart2frac(self.lattice6, _cart)

        _atom_index = np.array([database.symbol_map[s] for s in _symbols]).reshape(-1, 1)
        _dyn = np.ones((_nat, 3))
        _magmom = self.__default_magmom(_atom_index.flatten())

        _block = np.hstack((_atom_index, _frac, _cart, _dyn, _magmom))
        self.position = np.vstack((self.position, _block))

    def __parse_xdatcar(self, lines):
        """
        Parse a VASP XDATCAR AIMD/relaxation trajectory file, keeping only
        the last configuration -- the same "final geometry" convention
        __parse_qeout uses for a QE output file. Fixed-cell only, see
        _read_xdatcar_frames's own docstring.
        """
        _lattice, _species, _positions = _read_xdatcar_frames(lines)
        self.lattice9 = _lattice
        self.lattice6 = functions.lattice_conversion(self.lattice9)
        self.total_atom = len(_species)

        _atom_index = np.array([database.symbol_map[_s] for _s in _species]).reshape(-1, 1)
        _frac = _positions[-1]
        _cart = functions.frac2cart(self.lattice9, _frac)
        _dyn = np.ones((self.total_atom, 3))
        _magmom = self.__default_magmom(_atom_index.flatten())

        _block = np.hstack((_atom_index, _frac, _cart, _dyn, _magmom))
        self.position = np.vstack((self.position, _block))

    # ------------------------------------------------------------------ #
    # LAMMPS
    # ------------------------------------------------------------------ #
    def __parse_lammps(self, lines):
        """
        Parse a LAMMPS data file (atomic/charge atom style, orthogonal or
        triclinic box). Element symbols are recovered from the Masses
        section by matching each type's mass to the closest entry in
        mkits.database.atom_data.
        """
        self.title = lines[0].strip()

        _xlo = _xhi = _ylo = _yhi = _zlo = _zhi = 0.0
        _xy = _xz = _yz = 0.0
        _masses = {}
        _atoms_start = None
        _n_atoms = 0

        for i, line in enumerate(lines):
            _stripped = line.split("#")[0].strip()
            if re.match(r"^\d+\s+atoms$", _stripped):
                _n_atoms = int(_stripped.split()[0])
            elif "xlo" in _stripped and "xhi" in _stripped:
                _xlo, _xhi = (float(v) for v in _stripped.split()[:2])
            elif "ylo" in _stripped and "yhi" in _stripped:
                _ylo, _yhi = (float(v) for v in _stripped.split()[:2])
            elif "zlo" in _stripped and "zhi" in _stripped:
                _zlo, _zhi = (float(v) for v in _stripped.split()[:2])
            elif "xy" in _stripped and "xz" in _stripped and "yz" in _stripped:
                _xy, _xz, _yz = (float(v) for v in _stripped.split()[:3])
            elif _stripped == "Masses":
                j = i + 2
                while j < len(lines) and lines[j].strip():
                    _parts = lines[j].split("#")[0].split()
                    _masses[int(_parts[0])] = float(_parts[1])
                    j += 1
            elif _stripped == "Atoms":
                _atoms_start = i + 2

        if _atoms_start is None:
            raise functions.MkitsError("LAMMPS data file does not contain an Atoms section.")

        self.lattice9 = np.array([
            [_xhi - _xlo, 0.0, 0.0],
            [_xy, _yhi - _ylo, 0.0],
            [_xz, _yz, _zhi - _zlo]
        ])
        self.lattice6 = functions.lattice_conversion(self.lattice9)

        _type_symbol = {}
        for _atype, _mass in _masses.items():
            _best_z = min(range(1, 119), key=lambda z: abs((database.atom_data[z][3] or 1e9) - _mass))
            _type_symbol[_atype] = _best_z

        _atom_indices = []
        _cart_coords = []
        for line in lines[_atoms_start: _atoms_start + _n_atoms]:
            _parts = line.split("#")[0].split()
            if len(_parts) < 5:
                continue
            _atype = int(_parts[1])
            # supports "atomic" (id type x y z) and "charge" (id type q x y z) styles
            _xyz = _parts[-3:]
            _atom_indices.append(_type_symbol.get(_atype, 0))
            _cart_coords.append([float(v) for v in _xyz])

        self.total_atom = len(_atom_indices)
        _atom_indices = np.array(_atom_indices).reshape(-1, 1)
        _cart_coords = np.array(_cart_coords)
        _frac_coords = functions.cart2frac(self.lattice9, _cart_coords)
        _dyn = np.ones((self.total_atom, 3))
        _magmom = self.__default_magmom(_atom_indices.flatten())

        _block = np.hstack((_atom_indices, _frac_coords, _cart_coords, _dyn, _magmom))
        self.position = np.vstack((self.position, _block))

    # ------------------------------------------------------------------ #
    # WIEN2k
    # ------------------------------------------------------------------ #
    def __parse_wien2k(self, lines):
        """
        Parse a WIEN2k case.struct file. If an atom's declared multiplicity
        (MULT) exceeds the number of equivalent positions explicitly listed
        in the file, the missing ones are completed using the symmetry
        operations block at the end of the file.
        """
        _is_bohr = "ang" not in lines[2].lower()
        _a, _b, _c, _alpha, _beta, _gamma = (float(v) for v in lines[3].split()[:6])
        if _is_bohr:
            _a *= database.uc_bohr2ang
            _b *= database.uc_bohr2ang
            _c *= database.uc_bohr2ang
        self.lattice6 = np.array([_a, _b, _c, _alpha, _beta, _gamma])
        self.lattice9 = functions.lattice_conversion(self.lattice6)

        # trailing symmetry-operation block, used as a fallback to complete
        # atom groups whose equivalent positions are not all listed explicitly
        _sym_start = None
        _n_symops = 0
        for i, line in enumerate(lines):
            if "NUMBER OF SYMMETRY OPERATIONS" in line:
                _sym_start = i
                _n_symops = int(line.split()[0])
                break

        if _sym_start is not None:
            _rotations, _translations = [], []
            i = _sym_start + 1
            for _ in range(_n_symops):
                _rows, _trans = [], []
                for _r in range(3):
                    _parts = lines[i].split()
                    _rows.append([float(_parts[0]), float(_parts[1]), float(_parts[2])])
                    _trans.append(float(_parts[3]))
                    i += 1
                _rotations.append(np.array(_rows))
                _translations.append(np.array(_trans))
                i += 1  # skip the operation-index line
        else:
            _rotations, _translations = [np.eye(3)], [np.zeros(3)]

        _symbols = []
        _frac_coords = []
        i = 4
        while i < len(lines) and lines[i].startswith("ATOM"):
            _coord = [float(v) for v in re.findall(r"[XYZ]=\s*(-?\d+\.\d+)", lines[i])]
            _mult = int(re.findall(r"MULT=\s*(\d+)", lines[i + 1])[0])
            i += 2

            _rep_frac = [_coord]
            for _ in range(_mult - 1):
                _rep_frac.append([float(v) for v in re.findall(r"[XYZ]=\s*(-?\d+\.\d+)", lines[i])])
                i += 1

            _sym = re.sub(r"[^A-Za-z]", "", lines[i].split()[0])
            i += 1
            i += 3  # skip the 3-line local rotation matrix block

            if len(_rep_frac) < _mult:
                _exp_syms, _exp_coords = self.__expand_by_symmetry([_rep_frac[0]], [_sym], _rotations, _translations)
                _symbols += _exp_syms
                _frac_coords += _exp_coords.tolist()
            else:
                _symbols += [_sym] * _mult
                _frac_coords += _rep_frac

        self.total_atom = len(_symbols)
        _atom_index = np.array([database.symbol_map[s] for s in _symbols]).reshape(-1, 1)
        _frac_coords = np.array(_frac_coords)
        _cart_coords = functions.frac2cart(self.lattice9, _frac_coords)
        _dyn = np.ones((self.total_atom, 3))
        _magmom = self.__default_magmom(_atom_index.flatten())

        _block = np.hstack((_atom_index, _frac_coords, _cart_coords, _dyn, _magmom))
        self.position = np.vstack((self.position, _block))

    # ================================================================== #
    # in-place structure edits
    # ================================================================== #
    def sort_atoms(self):
        """Sort the atomic coordinates by atomic index. Updates self.position."""
        self.position = self.position[np.argsort(self.position[:, 0]), :]

    def add_dyn(self,
                xmin=-1e8, xmax=1e8,
                ymin=-1e8, ymax=1e8,
                zmin=-1e8, zmax=1e8,
                fix="none",
                move="none",
                frac=True):
        """
        Set atoms enclosed by the given box to fixed (dyn=0) and the rest
        to free (dyn=1); `fix`/`move` additionally override by element
        symbol (fix -> 0, move -> 1). Updates the dynrange columns of
        self.position.
        """
        _coord = self.position[1:, 1:4] if frac else self.position[1:, 4:7]

        for _i in range(len(_coord)):
            if xmax > _coord[_i, 0] > xmin and ymax > _coord[_i, 1] > ymin and zmax > _coord[_i, 2] > zmin:
                self.position[1 + _i, 7:10] = 0.0
            else:
                self.position[1 + _i, 7:10] = 1.0

        if fix != "none" and fix in database.symbol_map:
            _z = database.symbol_map[fix]
            for _i in range(1, self.total_atom + 1):
                if self.position[_i, 0] == _z:
                    self.position[_i, 7:10] = 0.0
        elif move != "none" and move in database.symbol_map:
            _z = database.symbol_map[move]
            for _i in range(1, self.total_atom + 1):
                if self.position[_i, 0] == _z:
                    self.position[_i, 7:10] = 1.0

    def set_magmom(self,
                    magmom=0.0,
                    xmin=-1e8, xmax=1e8,
                    ymin=-1e8, ymax=1e8,
                    zmin=-1e8, zmax=1e8,
                    element="none",
                    frac=True):
        """
        Set the initial magnetic moment of the atoms enclosed by the given
        box, or of every atom of a given element. Updates the magmom column
        of self.position.
        """
        if element != "none" and element in database.symbol_map:
            _z = database.symbol_map[element]
            for _i in range(1, self.total_atom + 1):
                if self.position[_i, 0] == _z:
                    self.position[_i, 10] = magmom
            return

        _coord = self.position[1:, 1:4] if frac else self.position[1:, 4:7]
        for _i in range(len(_coord)):
            if xmax > _coord[_i, 0] > xmin and ymax > _coord[_i, 1] > ymin and zmax > _coord[_i, 2] > zmin:
                self.position[1 + _i, 10] = magmom

    def scale_lattice9(
            self,
            scale=np.array([
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ])
    ):
        """Scale the Cartesian lattice matrix element-wise by `scale`."""
        self.lattice9 = self.lattice9 * scale

    def apply_strain(self, exx=0.0, eyy=0.0, ezz=0.0, exy=0.0, exz=0.0, eyz=0.0):
        """
        Deform the lattice by the given strain tensor components
        (dimensionless, eg 0.01 = 1% strain), keeping fractional
        coordinates fixed -- the clamped-ion strain convention used for
        deformation-potential and elastic-constant calculations (a
        subsequent fixed-cell ionic relaxation, if wanted, is layered on
        top by the caller, not by this method). Shear components are the
        plain tensor strain, not the engineering 2*strain convention.

        :param exx, eyy, ezz: normal strain along a, b, c
        :param exy, exz, eyz: shear strain components
        """
        _eps = np.array([
            [exx, exy, exz],
            [exy, eyy, eyz],
            [exz, eyz, ezz],
        ])
        self.lattice9 = self.lattice9 @ (np.eye(3) + _eps)
        self.lattice6 = functions.lattice_conversion(self.lattice9)
        self.position[1:, 4:7] = functions.frac2cart(self.lattice9, self.position[1:, 1:4])

    # ================================================================== #
    # writers
    # ================================================================== #
    def write_struct(self,
                      fpath: str = "./",
                      fname: str = "pwscf.in",
                      calculator: str = "qein",
                      dyn: bool = False,
                      frac: bool = True,
                      write2file: bool = True):
        """
        Write the structure to a file.

        Parameters
        ----------
        calculator: str
            Output format: poscar, qein, cif, xyz, lammps, wien2k.
        dyn: bool
            Whether to include selective-dynamics / constraint information
            (poscar, qein only).
        frac: bool
            Write fractional (True) or Cartesian (False) coordinates
            (poscar, qein only; cif is always fractional, xyz/lammps/wien2k
            always use their own native convention).
        write2file: bool
            If False, return the generated lines instead of writing them.
        """
        if calculator == "none":
            calculator = self.calculator

        _qesystemblock = {}
        if calculator == "poscar":
            _lines = self.__write_poscar_lines(dyn, frac)
        elif calculator == "qein":
            _qesystemblock, _lines = self.__write_qein_lines(dyn, frac)
        elif calculator == "cif":
            _lines = self.__write_cif_lines()
        elif calculator == "xyz":
            _lines = self.__write_xyz_lines()
        elif calculator == "lammps":
            _lines = self.__write_lammps_lines()
        elif calculator == "wien2k":
            _lines = self.__write_wien2k_lines()
        else:
            raise functions.MkitsError("Unsupported output format: %s" % calculator)

        if write2file:
            with open(fpath + "/" + fname, "w", newline="\n") as f:
                f.writelines(_lines)
        else:
            return _qesystemblock, _lines

    def __write_poscar_lines(self, dyn, frac):
        """Build the lines of a VASP POSCAR file."""
        _coord = self.position[1:, 1:4] if frac else self.position[1:, 4:7]
        _coord = functions.convert_array2strlist(_coord, fmt="{:20.10f}")

        _atomic_index = [int(_) for _ in self.position[1:, 0]]
        _atomic_type = np.sort(np.array(list(set(_atomic_index))))
        _atomic_num = [_atomic_index.count(_) for _ in _atomic_type]
        _atomic_symbols = [database.atom_data[_][1] for _ in _atomic_type]

        _lines = ["%s\n" % self.title, "1.0\n"]
        _lines += functions.convert_high2writeablelist(functions.convert_array2strlist(self.lattice9))
        _lines.append("   ".join(_atomic_symbols) + "\n")
        _lines += functions.convert_high2writeablelist(functions.convert_array2strlist(_atomic_num, fmt="{:<6d}"))

        if dyn:
            _lines.append("Selective dynamics\n")
            _dyn_flags = self.position[1:, 7:10]
            _dyn_flags = np.where(_dyn_flags > 0.5, "T", "F")
            _dyn_flags = functions.convert_array2strlist(_dyn_flags, fmt="{:>5s}")
            _lines.append("Direct\n" if frac else "Cartesian\n")
            _lines += functions.convert_high2writeablelist(functions.hstack_append_list(_coord, _dyn_flags))
        else:
            _lines.append("Direct\n" if frac else "Cartesian\n")
            _lines += functions.convert_high2writeablelist(_coord)

        return _lines

    def __write_qein_lines(self, dyn, frac):
        """Build the lines of a Quantum ESPRESSO pw.x input file."""
        _atomic_index = [int(_) for _ in self.position[1:, 0]]
        _atomic_type = np.sort(np.array(list(set(_atomic_index))))
        _atomic_symbols = [database.atom_data[_][1] for _ in _atomic_index]

        _coord = self.position[1:, 1:4] if frac else self.position[1:, 4:7]
        _coord = functions.convert_array2strlist(_coord, fmt="{:20.10f}")

        _qesystemblock = {"ibrav": 0, "nat": self.total_atom, "ntyp": len(_atomic_type)}

        _lines = ["&SYSTEM\n"]
        for _key in ("ibrav", "nat", "ntyp"):
            _lines.append("%s=%s\n" % (_key, _qesystemblock[_key]))
        _lines.append("/\n\n")

        _symbol_col = functions.convert_array2strlist(
            np.array(_atomic_symbols).reshape(self.total_atom, -1), fmt="{:>6s}"
        )

        _lines.append("ATOMIC_POSITIONS crystal\n" if frac else "ATOMIC_POSITIONS angstrom\n")
        if dyn:
            _dyn_flags = self.position[1:, 7:10]
            _dyn_flags = np.where(_dyn_flags > 0.5, "1.0", "0.0")
            _dyn_flags = functions.convert_array2strlist(_dyn_flags, fmt="{:>5s}")
            _rows = functions.hstack_append_list(_symbol_col, _coord)
            _rows = functions.hstack_append_list(_rows, _dyn_flags)
            _lines += functions.convert_high2writeablelist(_rows)
        else:
            _lines += functions.convert_high2writeablelist(functions.hstack_append_list(_symbol_col, _coord))

        _lines.append("CELL_PARAMETERS angstrom\n")
        _lines += functions.convert_high2writeablelist(functions.convert_array2strlist(self.lattice9))

        return _qesystemblock, _lines

    def __write_cif_lines(self):
        """Build the lines of a CIF file (P1, fractional coordinates)."""
        _safe_title = self.title.replace(" ", "_") if self.title != "none" else "generated_structure"
        _lines = [
            "data_%s\n" % _safe_title,
            "_audit_creation_method   'Generated by mkits.structure'\n\n",
            "_symmetry_space_group_name_H-M    'P 1'\n",
            "_symmetry_Int_Tables_number       1\n",
            "_symmetry_cell_setting            triclinic\n\n",
        ]

        if len(self.lattice6) != 6:
            self.lattice6 = functions.lattice_conversion(self.lattice9)
        _a, _b, _c, _alpha, _beta, _gamma = self.lattice6

        _lines.append("_cell_length_a        %12.6f\n" % _a)
        _lines.append("_cell_length_b        %12.6f\n" % _b)
        _lines.append("_cell_length_c        %12.6f\n" % _c)
        _lines.append("_cell_angle_alpha     %12.6f\n" % _alpha)
        _lines.append("_cell_angle_beta      %12.6f\n" % _beta)
        _lines.append("_cell_angle_gamma     %12.6f\n\n" % _gamma)

        _lines += [
            "loop_\n",
            " _atom_site_label\n",
            " _atom_site_type_symbol\n",
            " _atom_site_fract_x\n",
            " _atom_site_fract_y\n",
            " _atom_site_fract_z\n",
        ]

        _atomic_index = [int(_) for _ in self.position[1:, 0]]
        _atomic_symbols = [database.atom_data[_][1] for _ in _atomic_index]
        _frac_coords = self.position[1:, 1:4]

        _element_counter = {}
        for i, _sym in enumerate(_atomic_symbols):
            _element_counter[_sym] = _element_counter.get(_sym, 0) + 1
            _label = "%s%d" % (_sym, _element_counter[_sym])
            _fx, _fy, _fz = _frac_coords[i]
            _lines.append(" %-8s %-4s %12.8f %12.8f %12.8f\n" % (_label, _sym, _fx, _fy, _fz))

        return _lines

    def __write_xyz_lines(self):
        """Build the lines of a plain XYZ file (Cartesian coordinates)."""
        _lines = ["%d\n" % self.total_atom, "%s\n" % str(self.title).strip()]

        _indices = self.position[1:, 0]
        _cart_coords = self.position[1:, 4:7]
        for i in range(self.total_atom):
            _idx = int(_indices[i])
            try:
                _sym = database.atom_data[_idx][1]
            except IndexError:
                _sym = "X"
            _x, _y, _z = _cart_coords[i]
            _lines.append("%-4s %12.6f %12.6f %12.6f\n" % (_sym, _x, _y, _z))

        return _lines

    def __write_lammps_lines(self):
        """Build the lines of a LAMMPS data file (atomic style, triclinic box)."""
        _atomic_index = [int(_) for _ in self.position[1:, 0]]
        _atomic_type = sorted(set(_atomic_index))
        _type_of = {z: i + 1 for i, z in enumerate(_atomic_type)}

        _cart = self.position[1:, 4:7].astype(float)
        _cart = _cart - _cart.min(axis=0)

        _a, _b, _c = self.lattice9
        _xlo, _ylo, _zlo = 0.0, 0.0, 0.0
        _xhi, _xy, _yhi, _xz, _yz, _zhi = _a[0], _b[0], _b[1], _c[0], _c[1], _c[2]

        _lines = [
            "%s\n\n" % self.title,
            "%d atoms\n" % self.total_atom,
            "%d atom types\n\n" % len(_atomic_type),
            "%.10f %.10f xlo xhi\n" % (_xlo, _xhi),
            "%.10f %.10f ylo yhi\n" % (_ylo, _yhi),
            "%.10f %.10f zlo zhi\n" % (_zlo, _zhi),
            "%.10f %.10f %.10f xy xz yz\n\n" % (_xy, _xz, _yz),
            "Masses\n\n",
        ]
        for z in _atomic_type:
            _lines.append("%d %.6f\n" % (_type_of[z], database.atom_data[z][3] or 0.0))

        _lines.append("\nAtoms\n\n")
        for i, z in enumerate(_atomic_index):
            _x, _y, _w = _cart[i]
            _lines.append("%d %d %.10f %.10f %.10f\n" % (i + 1, _type_of[z], _x, _y, _w))

        return _lines

    def __write_wien2k_lines(self):
        """
        Build the lines of a WIEN2k case.struct file. Symmetry-equivalent
        atoms and the symmetry-operations block are obtained from spglib;
        local rotation matrices are written as identity (see
        mkits.database.local_rot_matrix).
        """
        _dataset = self.get_symmetry(symprec=1e-3)
        _equiv = np.array(_dataset["equivalent_atoms"])
        _rotations = _dataset["rotations"]
        _translations = _dataset["translations"]
        _sg_number = _dataset["number"]

        _atomic_index = self.position[1:, 0].astype(int)
        _frac = self.position[1:, 1:4]

        _unique_reps = sorted(set(_equiv.tolist()))
        _lines = [
            "mkits generated structure\n",
            "P   LATTICE,NONEQUIV.ATOMS: %3d %3d\n" % (len(_unique_reps), _sg_number),
            "MODE OF CALC=RELA unit=bohr\n",
        ]

        _a, _b, _c, _alpha, _beta, _gamma = self.lattice6
        _lines.append("%10.6f%10.6f%10.6f%10.6f%10.6f%10.6f\n" % (
            _a / database.uc_bohr2ang, _b / database.uc_bohr2ang, _c / database.uc_bohr2ang,
            _alpha, _beta, _gamma
        ))

        for _rep in _unique_reps:
            _members = np.where(_equiv == _rep)[0]
            _z = _atomic_index[_rep]
            _sym = database.atom_data[_z][1]
            _x, _y, _w = _frac[_rep]
            _lines.append("ATOM %3d: X=%10.8f Y=%10.8f Z=%10.8f\n" % (-(_rep + 1), _x, _y, _w))
            _lines.append("          MULT= %2d          ISPLIT= 8\n" % len(_members))
            for _m in _members[1:]:
                _x, _y, _w = _frac[_m]
                _lines.append("    %d: X=%10.8f Y=%10.8f Z=%10.8f\n" % (_rep + 1, _x, _y, _w))
            _lines.append("%-10s NPT=  781  R0=0.00010000 RMT=   2.0000   Z: %6.2f\n" % (_sym, float(_z)))
            _lines.append(database.local_rot_matrix)

        _lines.append("%4d      NUMBER OF SYMMETRY OPERATIONS\n" % len(_rotations))
        for _op_idx, (_rot, _trans) in enumerate(zip(_rotations, _translations)):
            for _row in range(3):
                # columns are space-separated (rather than WIEN2k's native
                # fixed-width, no-space I2 columns) so mkits' own whitespace-
                # based reader can parse the file it just wrote
                _lines.append("%3d %3d %3d %11.8f\n" % (_rot[_row, 0], _rot[_row, 1], _rot[_row, 2], _trans[_row]))
            _lines.append("      %d\n" % (_op_idx + 1))

        return _lines

    # ================================================================== #
    # geometry utilities
    # ================================================================== #
    def get_rec_basis(self, with_2pi=False):
        """Return the reciprocal lattice basis (rows are reciprocal vectors)."""
        if with_2pi:
            return np.linalg.inv(self.lattice9).T * 2 * np.pi
        else:
            return np.linalg.inv(self.lattice9).T

    def is_2d(self, vacuum_ratio=2.0, vacuum_min=12.0):
        """
        Heuristically flag the structure as 2D when the c-axis is both
        longer than an absolute threshold and much longer than a and b
        (the conventional slab-model convention: vacuum stacked along c).
        Shared by mkits.structure.seekpath and by the VASP/QE/... k-mesh
        generators, so every caller agrees on the same definition of "2D".

        :param vacuum_ratio: c is treated as a vacuum axis if c > vacuum_ratio * max(a, b)
        :param vacuum_min: minimum absolute length (angstrom) of the vacuum axis
        """
        _a, _b, _c = self.lattice6[:3]
        return _c > vacuum_min and _c > vacuum_ratio * max(_a, _b)

    def kmesh(self, kspacing=0.3, oddeven="none", kfix=None):
        """
        Compute a Gamma-centered Monkhorst-Pack mesh (n1, n2, n3) from the
        true reciprocal lattice, honoring 2D structures (the vacuum axis,
        assumed to be c, is forced to 1 point). Code-agnostic: VASP, QE and
        other front-ends format the returned tuple into their own KPOINTS
        card.

        :param kspacing: target spacing between k-points (1/angstrom, 2*pi convention)
        :param oddeven: "odd", "even" or "none": force the mesh to an odd/even
            number of points along each non-fixed direction
        :param kfix: optional list of 3 int; -1 means "not fixed", otherwise
            overrides the computed mesh count for that direction
        """
        if oddeven == "odd":
            _oddeven = 1
        elif oddeven == "even":
            _oddeven = 0
        else:
            _oddeven = -1

        _recip_lengths = np.linalg.norm(self.get_rec_basis(with_2pi=True), axis=1)
        _n = [
            int(max(1, functions.round_even_odd(_b / kspacing, _oddeven)))
            for _b in _recip_lengths
        ]

        if self.is_2d():
            _n[2] = 1

        if kfix is not None:
            for _i in range(3):
                if int(kfix[_i]) != -1:
                    _n[_i] = int(kfix[_i])

        return tuple(_n)

    def local_kbox(self, k0_frac, delta=0.01, npoints=5):
        """
        Build an explicit grid of fractional k-points centered on
        `k0_frac`, spaced `delta` apart (fractional reciprocal-lattice
        coordinates) along each reciprocal lattice direction -- used to
        sample a small neighborhood around a band extremum for a full
        effective-mass-tensor fit (mkits.mobility.effective_mass_tensor),
        unlike kmesh() (BZ coverage) or seekpath (a high-symmetry path).
        Honors is_2d() the same way kmesh() does: the vacuum axis (c*) is
        forced to 1 point, since a 2D structure's effective mass is a 2x2
        in-plane tensor, not a 3x3 one.

        :param k0_frac: (3,) fractional coordinates of the extremum
        :param delta: spacing between adjacent grid points, fractional
            reciprocal-lattice coordinates -- convert via get_rec_basis()
            for the absolute 1/angstrom scale this corresponds to
        :param npoints: number of points per sampled dimension; use an odd
            number so k0_frac itself is included exactly at the grid center

        Return
        ------
        (n, 3) array of fractional k-points, n = npoints^3 (3D) or
        npoints^2 (2D, is_2d()==True)
        """
        _offsets = (np.arange(npoints) - npoints // 2) * delta
        _dims = [_offsets, _offsets, np.array([0.0]) if self.is_2d() else _offsets]
        _grid = np.array(np.meshgrid(*_dims, indexing="ij")).reshape(3, -1).T
        return np.array(k0_frac, dtype=float) + _grid

    def add_atom(self, atomic_symbo, position, is_frac=True, magmom=None):
        """
        Append a single atom to the structure.

        Parameters
        ----------
        atomic_symbo: str
            Element symbol.
        position: (3,) array-like
        is_frac: bool
        magmom: float or None
            Initial magnetic moment; defaults to the element's database value.
        """
        _z = database.symbol_map[atomic_symbo]
        if is_frac:
            _frac_pos = np.array(position, dtype=float)
            _cart_pos = functions.frac2cart(self.lattice9, _frac_pos)
        else:
            _cart_pos = np.array(position, dtype=float)
            _frac_pos = functions.cart2frac(self.lattice9, _cart_pos)

        if magmom is None:
            magmom = database.atom_data[_z][4]

        self.total_atom += 1
        _new_line = np.array([
            _z,
            _frac_pos[0], _frac_pos[1], _frac_pos[2],
            _cart_pos[0], _cart_pos[1], _cart_pos[2],
            1, 1, 1,
            magmom
        ])
        self.position = np.vstack((self.position, _new_line))
        self.sort_atoms()

    def replace_atom(
            self,
            ranges="xmin=-1,xmax=-1,ymin=-1,ymax=-1,zmin=-1,zmax=-1",
            atomic_symbo="Ti",
            is_frac=True
    ):
        """
        Replace the element of every atom enclosed by the given box.

        :param ranges: box boundaries, "xmin=..,xmax=..,ymin=..,ymax=..,zmin=..,zmax=.."
        :param atomic_symbo: the new element symbol
        :param is_frac: whether `ranges` is expressed in fractional coordinates
        """
        if not is_frac:
            raise functions.MkitsError("replace_atom currently only supports fractional ranges.")

        _replace_range = {
            "xmin": -1e8, "ymin": -1e8, "zmin": -1e8,
            "xmax": 1e8, "ymax": 1e8, "zmax": 1e8,
        }
        _given = functions.parser_inputpara(ranges)
        for _key in _given:
            _replace_range[_key] = float(_given[_key])

        for _i in range(self.total_atom):
            if _replace_range["xmin"] < self.position[_i + 1, 1] < _replace_range["xmax"] and \
                    _replace_range["ymin"] < self.position[_i + 1, 2] < _replace_range["ymax"] and \
                    _replace_range["zmin"] < self.position[_i + 1, 3] < _replace_range["zmax"]:
                self.position[_i + 1, 0] = database.symbol_map[atomic_symbo]

        self.sort_atoms()

    def supercell(self, super_matrix=[2, 2, 1]):
        """
        Build a supercell by repeating the cell along each lattice vector.

        :param super_matrix: repetition counts along a, b, c.
        """
        _na, _nb, _nc = super_matrix
        _base = self.position[1:].astype(float)
        _n_base = _base.shape[0]

        _shifts = np.array([
            [i, j, k]
            for i in range(_na) for j in range(_nb) for k in range(_nc)
        ])
        _n_shift = _shifts.shape[0]

        _expanded = np.repeat(_base, _n_shift, axis=0)
        _tiled_shifts = np.tile(_shifts, (_n_base, 1))
        _expanded[:, 1:4] = (_expanded[:, 1:4] + _tiled_shifts) / np.array([_na, _nb, _nc])

        self.lattice6 = np.hstack((self.lattice6[:3] * np.array([_na, _nb, _nc]), self.lattice6[3:]))
        self.lattice9 = functions.lattice_conversion(self.lattice6)
        _expanded[:, 4:7] = functions.frac2cart(self.lattice9, _expanded[:, 1:4])

        self.position = np.vstack((np.zeros((1, _expanded.shape[1])), _expanded))
        self.total_atom = _expanded.shape[0]
        self.sort_atoms()

    def transit_axis(
            self,
            new_axis=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
    ):
        """
        Change the lattice to `new_axis` and update fractional coordinates
        accordingly. Atoms falling outside the new cell are dropped.

        :param new_axis: 3x3 numpy array in angstrom
        """
        _cart_pos = self.position[1:, 4:7].astype(float)

        _new_lattice9 = np.array(new_axis, dtype=float)
        _new_frac_pos = functions.cart2frac(_new_lattice9, _cart_pos)

        # fractional tolerance corresponding to 0.01 angstrom
        _tol_angstrom = 0.01
        _inv_new_lattice = np.linalg.inv(_new_lattice9)
        _recip_lengths = np.linalg.norm(_inv_new_lattice, axis=0)
        _frac_tols = _tol_angstrom * _recip_lengths

        _inside_mask = np.all(
            (_new_frac_pos >= -_frac_tols) & (_new_frac_pos <= 1.0 + _frac_tols),
            axis=1
        )
        _new_frac_pos[_inside_mask] = np.clip(_new_frac_pos[_inside_mask], 0.0, 1.0)

        _buffer_row = self.position[0:1, :]
        _kept_atoms_rows = self.position[1:][_inside_mask].copy()
        _kept_atoms_rows[:, 1:4] = _new_frac_pos[_inside_mask]

        self.position = np.vstack((_buffer_row, _kept_atoms_rows))
        self.lattice9 = _new_lattice9
        self.lattice6 = functions.lattice_conversion(self.lattice9)
        self.total_atom = len(_kept_atoms_rows)

        self.sort_atoms()

    # ================================================================== #
    # symmetry (spglib)
    # ================================================================== #
    def __to_spglib_cell(self):
        """Build the (lattice, positions, numbers) tuple required by spglib."""
        _frac_pos = self.position[1:, 1:4].astype(float)
        _numbers = self.position[1:, 0].astype(int).tolist()
        return (self.lattice9.tolist(), _frac_pos.tolist(), _numbers)

    def get_spacegroup(self, symprec=1e-3):
        """Return the space-group symbol and number detected by spglib, eg 'Fm-3m (225)'."""
        return spg.get_spacegroup(self.__to_spglib_cell(), symprec=symprec)

    def get_symmetry(self, symprec=1e-3):
        """
        Return the symmetry dataset detected by spglib as a plain dict with
        the keys: number, international, hall, rotations, translations,
        equivalent_atoms, wyckoffs, std_lattice, transformation_matrix.
        `std_lattice` (one lattice vector per row) is spglib's idealized
        standard conventional cell; `transformation_matrix` P relates it to
        this structure's own cell: (std axes) = (own axes) @ inv(P), all as
        columns.
        """
        _dataset = spg.get_symmetry_dataset(self.__to_spglib_cell(), symprec=symprec)
        _keys = ["number", "international", "hall", "rotations", "translations",
                 "equivalent_atoms", "wyckoffs", "std_lattice", "transformation_matrix"]
        return {_k: _spglib_field(_dataset, _k) for _k in _keys}

    def get_primitive_cell(self, symprec=1e-3):
        """Return a new struct instance reduced to the primitive cell found by spglib."""
        _result = spg.find_primitive(self.__to_spglib_cell(), symprec=symprec)
        if _result is None or _result[0] is None:
            raise functions.MkitsError("spglib failed to find a primitive cell for this structure.")
        _lattice, _positions, _numbers = _result

        _prim = struct("none")
        _prim.calculator = self.calculator
        _prim.title = self.title
        _prim.lattice9 = np.array(_lattice)
        _prim.lattice6 = functions.lattice_conversion(_prim.lattice9)

        _numbers = np.array(_numbers).reshape(-1, 1)
        _frac = np.array(_positions)
        _cart = functions.frac2cart(_prim.lattice9, _frac)
        _dyn = np.ones((len(_numbers), 3))
        _magmom = self.__default_magmom(_numbers.flatten())

        _prim.total_atom = len(_numbers)
        _prim.position = np.vstack((
            np.zeros((1, 11)),
            np.hstack((_numbers, _frac, _cart, _dyn, _magmom))
        ))
        _prim.sort_atoms()
        return _prim


class volumetric(struct):
    """
    Volumetric data object (eg VASP CHGCAR-like files: a structure followed
    by a scalar field on a regular grid).

    :param inp: the name of the input structure/volumetric file
    """

    def __init__(self, inp):
        super().__init__(inp)
        self.grid = np.array([0, 0, 0])
        self.voldata = np.array([])
        self.__parse_volumetric()

    def __parse_volumetric(self):
        """Parse the grid data following the structure block."""
        if self.calculator == "poscar":
            self.grid = np.loadtxt(
                self.inp,
                skiprows=self.total_atom + 9,
                max_rows=1
            )
            _total_grid = int(self.grid[0] * self.grid[1] * self.grid[2])
            self.voldata = np.loadtxt(
                self.inp,
                skiprows=self.total_atom + 10,
                max_rows=_total_grid // 5
            ).flatten()
            if _total_grid % 5 != 0:
                self.voldata = np.hstack((
                    self.voldata,
                    np.loadtxt(
                        self.inp,
                        skiprows=self.total_atom + 9 + _total_grid // 5,
                        max_rows=1
                    )
                ))
            self.voldata = self.voldata.reshape((
                int(self.grid[2]),
                int(self.grid[1]),
                int(self.grid[0])
            ))
        else:
            pass

    def write_volumetric(
        self,
        fpath="./",
        fname="pwscf.in",
        calculator="qein",
        dyn=False,
        frac=True
    ):
        """Write the structure followed by the volumetric grid data."""
        _, _lines = super().write_struct(
            fpath, fname, calculator, dyn, frac, write2file=False
        )
        _total_grid = int(self.grid[0] * self.grid[1] * self.grid[2])

        if calculator == "poscar":
            _lines.append("\n")
            with open(fpath + "/" + fname, "w", newline="\n") as f:
                f.writelines(_lines)
                np.savetxt(f, np.array([self.grid]), fmt="%5d")
                _voldata = self.voldata.flatten()[:_total_grid // 5 * 5]
                np.savetxt(f, _voldata.reshape((-1, 5)), fmt="%10.11e")
                if _total_grid % 5 != 0:
                    np.savetxt(f, self.voldata.flatten()[_total_grid // 5 * 5:], fmt="%10.11e")

    def set_value(
        self,
        value,
        xmin=0, xmax=1,
        ymin=0, ymax=1,
        zmin=0, zmax=1,
        frac=True
    ):
        """Set the grid values inside the given fractional box to `value`."""
        if frac:
            xmin = int(xmin * self.grid[0])
            xmax = int(xmax * self.grid[0])
            ymin = int(ymin * self.grid[1])
            ymax = int(ymax * self.grid[1])
            zmin = int(zmin * self.grid[2])
            zmax = int(zmax * self.grid[2])
        self.voldata[zmin:zmax, ymin:ymax, xmin:xmax] = value


# ------------------------------------------------------------------ #
# Setyawan-Curtarolo (SC) standard primitive cells, used by seekpath
# ------------------------------------------------------------------ #
# Conventional -> primitive matrices of the SC standard cells: the columns
# are the primitive lattice vectors expressed in the conventional (a, b, c)
# basis. "C_ORTHO" and "C_MONO" differ on purpose: SC defines the
# base-centered orthorhombic primitive vectors as (a/2, -b/2, 0), (a/2, b/2,
# 0), (0, 0, c) but the base-centered monoclinic ones as (a/2, b/2, 0),
# (-a/2, b/2, 0), (0, c cos(alpha), c sin(alpha)).
_SC_PRIM_MATRIX = {
    "P": np.eye(3),
    "F": 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float),
    "I": 0.5 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float),
    "C_ORTHO": np.array([[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]),
    "C_MONO": np.array([[0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]),
    "R": np.array([[2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0],
                   [1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0],
                   [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]),
}

# space groups with a rhombohedral (R-centered) lattice -- the other
# trigonal groups (143-167) have a hexagonal lattice
_SC_RHOMBOHEDRAL_GROUPS = (146, 148, 155, 160, 161, 166, 167)

# tolerance of every "equal to" / "which side of" decision below (relative
# for lengths and parameter combinations, absolute for angle cosines); it
# only ever decides between neighboring SC variants
_SC_TOL = 1e-4


def _sc_signed_permutation(perm, signs=(1.0, 1.0, 1.0)):
    """
    3x3 matrix Q with (new axes) = (old axes) @ Q (axes as columns), where
    new axis j is old axis `perm[j]` times `signs[j]`.
    """
    _q = np.zeros((3, 3))
    for _j in range(3):
        _q[perm[_j], _j] = signs[_j]
    return _q


def _sc_cos_angle(vec1, vec2):
    """Cosine of the angle between two vectors."""
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def _sc_primitive_vectors(name, params):
    """
    The lattice vectors (rows) of the SC standard primitive cell of lattice
    variant `name` (Setyawan & Curtarolo, Comput. Mater. Sci. 49, 299
    (2010), Table 2), from its conventional-cell `params`.
    """
    _a = params.get("a")
    _b = params.get("b")
    _c = params.get("c")
    if name == "CUB":
        return np.diag([_a, _a, _a])
    if name == "FCC":
        return _a / 2.0 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    if name == "BCC":
        return _a / 2.0 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float)
    if name == "TET":
        return np.diag([_a, _a, _c])
    if name in ("BCT1", "BCT2"):
        return np.array([[-_a, _a, _c], [_a, -_a, _c], [_a, _a, -_c]]) / 2.0
    if name == "ORC":
        return np.diag([_a, _b, _c])
    if name in ("ORCF1", "ORCF2", "ORCF3"):
        return np.array([[0.0, _b, _c], [_a, 0.0, _c], [_a, _b, 0.0]]) / 2.0
    if name == "ORCI":
        return np.array([[-_a, _b, _c], [_a, -_b, _c], [_a, _b, -_c]]) / 2.0
    if name == "ORCC":
        return np.array([[_a / 2.0, -_b / 2.0, 0.0], [_a / 2.0, _b / 2.0, 0.0], [0.0, 0.0, _c]])
    if name == "HEX":
        return np.array([[_a / 2.0, -_a * np.sqrt(3.0) / 2.0, 0.0],
                         [_a / 2.0, _a * np.sqrt(3.0) / 2.0, 0.0],
                         [0.0, 0.0, _c]])
    if name in ("RHL1", "RHL2"):
        _alpha = np.radians(params["alpha"])
        _cos_half = np.cos(_alpha / 2.0)
        _ratio = np.cos(_alpha) / _cos_half
        return _a * np.array([[_cos_half, -np.sin(_alpha / 2.0), 0.0],
                              [_cos_half, np.sin(_alpha / 2.0), 0.0],
                              [_ratio, 0.0, np.sqrt(1.0 - _ratio ** 2)]])
    _alpha = np.radians(params["alpha"])
    _third = [0.0, _c * np.cos(_alpha), _c * np.sin(_alpha)]
    if name == "MCL":
        return np.array([[_a, 0.0, 0.0], [0.0, _b, 0.0], _third])
    if name.startswith("MCLC"):
        return np.array([[_a / 2.0, _b / 2.0, 0.0], [-_a / 2.0, _b / 2.0, 0.0], _third])
    raise functions.MkitsError("No SC standard primitive cell defined for lattice variant '%s'." % name)


def _sc_standard_cell(number, letter, std_rows):
    """
    Pick the SC lattice variant of a space group and find how its standard
    primitive cell sits inside spglib's idealized standard conventional cell.

    :param number: space-group number
    :param letter: lattice-centering letter of the space-group symbol
        ("P", "A", "B", "C", "I", "F" or "R")
    :param std_rows: spglib's idealized standard conventional lattice, one
        lattice vector per row

    Return
    ------
    (name, params, M): the SC variant name (key of mkits.database.kpath_3d),
        its conventional-cell lattice parameters (a, b, c in the same unit as
        `std_rows`, alpha in degrees; only the ones its k-point formulas need
        are guaranteed meaningful), and the 3x3 matrix M such that
        (SC standard primitive lattice vectors) = (`std_rows` lattice
        vectors) @ M, all as columns.
    """
    _cols = np.array(std_rows, dtype=float).T
    _lens = np.linalg.norm(_cols, axis=0)

    if number <= 2:
        raise functions.MkitsError("_sc_standard_cell does not handle triclinic cells.")

    # --- monoclinic: spglib's standard cell has the unique axis along b ---
    if number <= 15:
        if letter == "P":
            _in_plane = sorted([0, 2], key=lambda _i: _lens[_i])
            _perm = [1, _in_plane[0], _in_plane[1]]
            _prim = _SC_PRIM_MATRIX["P"]
        elif letter == "C":
            _perm = [1, 0, 2]
            _prim = _SC_PRIM_MATRIX["C_MONO"]
        else:
            raise functions.MkitsError(
                "Unsupported monoclinic lattice centering '%s' (spglib's standard cell is expected "
                "to be P or C centered)." % letter
            )
        # SC wants the angle alpha between the new b and c to be acute: flip
        # the new a and b together (keeps the lattice and the C centering)
        _cos = _sc_cos_angle(_cols[:, _perm[1]], _cols[:, _perm[2]])
        _signs = (-1.0, -1.0, 1.0) if _cos < 0.0 else (1.0, 1.0, 1.0)
        _params = {
            "a": _lens[_perm[0]], "b": _lens[_perm[1]], "c": _lens[_perm[2]],
            "alpha": float(np.degrees(np.arccos(abs(_cos)))),
        }
        _m = _sc_signed_permutation(_perm, _signs) @ _prim
        if letter == "P":
            return "MCL", _params, _m

        # base-centered monoclinic: MCLC1-5 are told apart by the angle
        # k_gamma between the first two reciprocal primitive vectors and,
        # when that is acute, by one more combination of the parameters
        _primitive = _sc_primitive_vectors("MCLC1", _params)
        _gram_inv = np.linalg.inv(_primitive @ _primitive.T)
        _cos_kgamma = _gram_inv[0, 1] / np.sqrt(_gram_inv[0, 0] * _gram_inv[1, 1])
        if _cos_kgamma < -_SC_TOL:
            return "MCLC1", _params, _m
        if abs(_cos_kgamma) <= _SC_TOL:
            return "MCLC2", _params, _m
        _alpha = np.radians(_params["alpha"])
        _a, _b, _c = _params["a"], _params["b"], _params["c"]
        _disc = _b * np.cos(_alpha) / _c + _b ** 2 * np.sin(_alpha) ** 2 / _a ** 2
        if _disc < 1.0 - _SC_TOL:
            return "MCLC3", _params, _m
        if _disc > 1.0 + _SC_TOL:
            return "MCLC5", _params, _m
        return "MCLC4", _params, _m

    # --- orthorhombic ---
    if number <= 74:
        if letter in ("P", "F", "I"):
            _perm = [int(_i) for _i in np.argsort(_lens, kind="stable")]
        elif letter == "C":
            _perm = sorted([0, 1], key=lambda _i: _lens[_i]) + [2]
        elif letter == "A":
            _perm = sorted([1, 2], key=lambda _i: _lens[_i]) + [0]
        elif letter == "B":
            _perm = sorted([0, 2], key=lambda _i: _lens[_i]) + [1]
        else:
            raise functions.MkitsError("Unsupported orthorhombic lattice centering '%s'." % letter)
        _a, _b, _c = (_lens[_i] for _i in _perm)
        _params = {"a": _a, "b": _b, "c": _c}
        _q = _sc_signed_permutation(_perm)
        if letter == "P":
            return "ORC", _params, _q
        if letter == "I":
            return "ORCI", _params, _q @ _SC_PRIM_MATRIX["I"]
        if letter == "F":
            _x, _y = 1.0 / _a ** 2, 1.0 / _b ** 2 + 1.0 / _c ** 2
            if abs(_x - _y) <= _SC_TOL * max(_x, _y):
                _name = "ORCF3"
            else:
                _name = "ORCF1" if _x > _y else "ORCF2"
            return _name, _params, _q @ _SC_PRIM_MATRIX["F"]
        return "ORCC", _params, _q @ _SC_PRIM_MATRIX["C_ORTHO"]

    # --- tetragonal: spglib's standard cell has c as the unique axis ---
    if number <= 142:
        _params = {"a": _lens[0], "c": _lens[2]}
        if letter == "P":
            return "TET", _params, np.eye(3)
        return ("BCT1" if _lens[2] < _lens[0] else "BCT2"), _params, _SC_PRIM_MATRIX["I"]

    # --- trigonal / hexagonal ---
    if number <= 194:
        if number in _SC_RHOMBOHEDRAL_GROUPS:
            _prim = _cols @ _SC_PRIM_MATRIX["R"]
            _a = float(np.linalg.norm(_prim[:, 0]))
            _cos = np.mean([
                _sc_cos_angle(_prim[:, 0], _prim[:, 1]),
                _sc_cos_angle(_prim[:, 0], _prim[:, 2]),
                _sc_cos_angle(_prim[:, 1], _prim[:, 2]),
            ])
            _alpha = float(np.degrees(np.arccos(_cos)))
            return ("RHL1" if _alpha < 90.0 else "RHL2"), {"a": _a, "alpha": _alpha}, _SC_PRIM_MATRIX["R"]
        return "HEX", {"a": _lens[0], "c": _lens[2]}, np.eye(3)

    # --- cubic ---
    if letter not in ("P", "F", "I"):
        raise functions.MkitsError("Unsupported cubic lattice centering '%s'." % letter)
    _name = {"P": "CUB", "F": "FCC", "I": "BCC"}[letter]
    return _name, {"a": _lens[0]}, _SC_PRIM_MATRIX[letter]


def _sc_triclinic_cell(std_rows):
    """
    SC standard primitive cell of a triclinic lattice: choose the
    primitive basis whose *reciprocal* vectors are Niggli-reduced and have
    pairwise angles that are all acute-or-right (SC "TRIb" variants) or all
    obtuse-or-right ("TRIa"), which is what makes the SC points -- half a
    reciprocal vector, or a half-integer sum of them, the only high-symmetry
    points a triclinic lattice has -- the shortest member of their
    equivalence class (ie inside the first Brillouin zone).

    :param std_rows: the (primitive) triclinic lattice, one vector per row

    Return
    ------
    (name, M): "TRI1a"/"TRI2a"/"TRI1b"/"TRI2b" (the "2" variants have one
        right reciprocal angle), and the 3x3 matrix M with (standard
        primitive lattice vectors) = (`std_rows` lattice vectors) @ M, as
        columns.
    """
    _cols = np.array(std_rows, dtype=float).T
    _rec_cols = np.linalg.inv(_cols).T
    _reduced_rows = spg.niggli_reduce(_rec_cols.T.copy())
    if _reduced_rows is None:
        raise functions.MkitsError("spglib failed to Niggli-reduce the reciprocal lattice.")
    _u_float = np.array(_reduced_rows) @ np.linalg.inv(_rec_cols.T)
    _u = np.rint(_u_float)
    if not np.allclose(_u_float, _u, atol=1e-3) or abs(abs(np.linalg.det(_u)) - 1.0) > 1e-6:
        raise functions.MkitsError(
            "The Niggli-reduced reciprocal lattice is not an integer (unimodular) combination of "
            "the original one:\n%s" % _u_float
        )
    _w = _u.T  # reciprocal columns: new = old @ _w

    # try no flip and each pair flip (keeps the handedness); some variant
    # always has the three angles all on one side of 90 degrees, since a
    # pair flip negates exactly two of the three angle cosines
    _best = None
    for _signs in ((1.0, 1.0, 1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0), (1.0, -1.0, -1.0)):
        _w_try = _w @ np.diag(_signs)
        _rec_try = _rec_cols @ _w_try
        _cos = np.array([
            _sc_cos_angle(_rec_try[:, 1], _rec_try[:, 2]),
            _sc_cos_angle(_rec_try[:, 0], _rec_try[:, 2]),
            _sc_cos_angle(_rec_try[:, 0], _rec_try[:, 1]),
        ])
        if np.all(_cos <= _SC_TOL):
            _best = (_w_try, "a", _cos)
        elif np.all(_cos >= -_SC_TOL):
            _best = (_w_try, "b", _cos)
    if _best is None:
        raise functions.MkitsError("Cannot find a triclinic reciprocal basis with all angles on one side of 90 degrees.")

    _w_best, _side, _cos = _best
    _name = "TRI%d%s" % (2 if np.any(np.abs(_cos) <= _SC_TOL) else 1, _side)
    # real-space standard primitive basis dual to the chosen reciprocal basis
    return _name, np.linalg.inv(_w_best).T


class seekpath(struct):
    """
    Generate a standard high-symmetry k-path for band-structure calculations.

    3D structures follow the Setyawan-Curtarolo (SC) convention for all 14
    Bravais lattices and their lattice-parameter-dependent variants
    (mkits.database.kpath_3d): spglib identifies the space group, its
    standard conventional cell and the transformation from the input cell to
    it; that picks the SC variant (centering included, eg a face-centered
    cubic Si cell is "FCC" whether it is given as the 2-atom primitive or
    the 8-atom conventional cell) and the transformation to the SC standard
    primitive cell, whose reciprocal-basis fractional k-points are then
    mapped back onto the input cell's own reciprocal basis. The k-points
    written out are therefore always valid for the cell actually passed in,
    standard or not, primitive or not (for a non-primitive cell they are
    the same Brillouin-zone points seen through that cell's smaller
    Brillouin zone, ie the bands come out folded).

    2D structures still use the first, self-built mkits.database.kpath_2d
    approximation, which only distinguishes the four primitive 2D Bravais
    lattice types (no centered rectangular lattice).

    :param inp: input structure file
    :param is2d: force 2D/3D path selection; auto-detected if None
    :param vacuum_ratio: c is treated as a vacuum axis if c > vacuum_ratio * max(a, b)
    :param vacuum_min: minimum absolute length (angstrom) of the vacuum axis
    :param symprec: symmetry-detection tolerance passed to spglib

    Attributes
    ----------
    lattice_type: str
        3D: the SC lattice variant (a key of mkits.database.kpath_3d, eg
        "FCC", "BCT1", "ORCF2", "MCLC3"); 2D: square / rectangular /
        hexagonal / oblique.
    sc_params: dict
        3D only: the standard conventional cell's lattice parameters that
        the variant's k-point formulas use (a, b, c, alpha; lengths in
        angstrom, alpha in degrees).
    sc_transform: (3, 3) numpy array
        3D only: T with (SC standard primitive lattice vectors) = (this
        structure's lattice vectors) @ T, all as columns. The identity for
        a structure already given as its SC standard primitive cell.
    kpath_points: dict
        label -> fractional coordinates in this structure's own reciprocal
        basis.
    kpath_segments: list of list
        the path as continuous segments of labels.
    """

    def __init__(self, inp, is2d=None, vacuum_ratio=2.0, vacuum_min=12.0, symprec=1e-3):
        super().__init__(inp)
        self.symprec = symprec
        self.is2d = self.is_2d(vacuum_ratio, vacuum_min) if is2d is None else is2d
        self.sc_params = {}
        self.sc_transform = np.eye(3)
        if self.is2d:
            self.lattice_type = self.__classify_2d()
        else:
            self.lattice_type, self.sc_params, self.sc_transform = self.__standard_primitive_3d()
        self.kpath_points = {}
        self.kpath_segments = []
        self.__build_kpath()

    def __classify_2d(self):
        """Classify the in-plane (a, b) lattice into a 2D Bravais lattice type."""
        _a, _b, _, _, _, _gamma = self.lattice6
        if abs(_a - _b) < 1e-2 * _a:
            if abs(_gamma - 90.0) < 1.0:
                return "square"
            if abs(_gamma - 120.0) < 1.0 or abs(_gamma - 60.0) < 1.0:
                return "hexagonal"
        if abs(_gamma - 90.0) < 1.0:
            return "rectangular"
        return "oblique"

    def __standard_primitive_3d(self):
        """
        Identify the SC lattice variant of this structure and the
        transformation from this structure's cell to the variant's standard
        primitive cell.

        Return
        ------
        (name, params, T): see the class docstring for `lattice_type`,
            `sc_params` and `sc_transform`.
        """
        _dataset = self.get_symmetry(symprec=self.symprec)
        if _dataset["number"] is None:
            # spglib found no symmetry at all -- treat the input as a P1 cell
            _number, _letter = 1, "P"
            _std_rows = np.array(self.lattice9)
            _p_inv = np.eye(3)
        else:
            _number = _dataset["number"]
            _letter = _dataset["international"].strip()[0]
            _std_rows = np.array(_dataset["std_lattice"])
            _p_inv = np.linalg.inv(np.array(_dataset["transformation_matrix"]))

        if _number <= 2:
            _name, _m = _sc_triclinic_cell(_std_rows)
            _params = {}
        else:
            _name, _params, _m = _sc_standard_cell(_number, _letter, _std_rows)

        _transform = _p_inv @ _m
        self.__check_standard_cell(_name, _params, _transform)
        return _name, _params, _transform

    def __check_standard_cell(self, name, params, transform):
        """
        Sanity-check the transformation found by __standard_primitive_3d:
        this structure's lattice transformed into the SC standard primitive
        cell must have the metric tensor that variant's own lattice
        parameters imply. A gross mismatch means a bug (or a wildly
        distorted input), not a subtle tolerance issue, so only a loose
        tolerance is used. Triclinic has no parameter-defined standard cell
        to compare against.
        """
        if name.startswith("TRI"):
            return
        _cols = self.lattice9.T @ transform
        _gram = _cols.T @ _cols
        _expected_rows = _sc_primitive_vectors(name, params)
        _expected = _expected_rows @ _expected_rows.T
        if not np.allclose(_gram, _expected, rtol=0.0, atol=0.1 * np.max(np.diag(_expected))):
            raise functions.MkitsError(
                "Internal error: the transformation to the SC standard primitive cell of lattice "
                "variant %s does not reproduce its metric tensor (got\n%s\nexpected\n%s)."
                % (name, _gram, _expected)
            )

    def __build_kpath(self):
        """Populate self.kpath_points / self.kpath_segments from the tables."""
        if self.is2d:
            _entry = database.kpath_2d[self.lattice_type]
            self.kpath_points = _entry["points"]
            self.kpath_segments = _entry["path"]
            return

        _entry = database.kpath_3d[self.lattice_type]
        _points = _entry["points"]
        if callable(_points):
            _points = _points(self.sc_params)
        if self.lattice_type.startswith("TRI"):
            _points = self.__shortest_triclinic_points(_points)
        # fractional coordinates of a reciprocal-space point transform with
        # the inverse transpose of the real-space basis change
        _to_input = np.linalg.inv(self.sc_transform).T
        self.kpath_points = {}
        for _label, _kpoint in _points.items():
            _mapped = np.round(_to_input @ np.array(_kpoint, dtype=float), 10) + 0.0
            self.kpath_points[_label] = _mapped.tolist()
        self.kpath_segments = [list(_segment) for _segment in _entry["path"]]

    def __shortest_triclinic_points(self, points):
        """
        Replace every triclinic SC point by its equivalent (same point modulo
        a reciprocal lattice vector) closest to Gamma. The SC table fixes the
        signs of the half-integer coefficients for an idealized reciprocal
        basis, but which sign pattern of a three-vector point like (1/2, 1/2,
        1/2) is the shortest one depends on which pair of reciprocal vectors
        has the largest dot product -- so for an actual lattice the table's
        choice can fall outside the first Brillouin zone while another
        equivalent one does not. Picking the shortest one keeps the straight
        Gamma-to-point lines of the triclinic path inside the zone.
        """
        _rec = np.linalg.inv(self.lattice9.T @ self.sc_transform).T
        _metric = _rec.T @ _rec
        _shifts = np.array(np.meshgrid([0, -1, 1], [0, -1, 1], [0, -1, 1])).reshape(3, -1).T
        _shifts = _shifts[np.lexsort((np.abs(_shifts).sum(axis=1),))]  # zero shift first
        _result = {}
        for _label, _kpoint in points.items():
            _frac = np.array(_kpoint, dtype=float)
            _best, _best_norm = _frac, _frac @ _metric @ _frac
            for _shift in _shifts:
                _trial = _frac + _shift
                _norm = _trial @ _metric @ _trial
                if _norm < _best_norm - 1e-9 * max(1.0, _best_norm):
                    _best, _best_norm = _trial, _norm
            _result[_label] = _best.tolist()
        return _result

    def write_kpath(self, fpath="./", fname="KPOINTS_band", code="vasp", kpoints_per_segment=20,
                     write2file=True):
        """
        Write the high-symmetry k-path to a band-structure k-points file, or
        return its lines (`write2file=False`) so a caller (eg mkits.qe) can
        embed them directly into a larger input file instead of a standalone
        KPOINTS/K_POINTS file.

        :param code: "vasp" (KPOINTS, line mode) or "qe" (K_POINTS crystal_b)
        :param kpoints_per_segment: number of k-points interpolated per segment
        """
        if code == "vasp":
            _lines = ["k-path generated by mkits.structure.seekpath\n",
                      "%d\n" % kpoints_per_segment, "Line-mode\n", "Reciprocal\n"]
            for _segment in self.kpath_segments:
                for _label1, _label2 in zip(_segment[:-1], _segment[1:]):
                    _p1 = self.kpath_points[_label1]
                    _p2 = self.kpath_points[_label2]
                    _lines.append("%12.8f%12.8f%12.8f  ! %s\n" % (_p1[0], _p1[1], _p1[2], _label1))
                    _lines.append("%12.8f%12.8f%12.8f  ! %s\n\n" % (_p2[0], _p2[1], _p2[2], _label2))
        elif code == "qe":
            _lines = []
            _all_labels = [label for _segment in self.kpath_segments for label in _segment]
            _lines.append("K_POINTS crystal_b\n")
            _lines.append("%d\n" % len(_all_labels))
            for _segment in self.kpath_segments:
                for _label in _segment[:-1]:
                    _p = self.kpath_points[_label]
                    _lines.append("%12.8f%12.8f%12.8f %d  ! %s\n" % (_p[0], _p[1], _p[2], kpoints_per_segment, _label))
                _last = _segment[-1]
                _p = self.kpath_points[_last]
                _lines.append("%12.8f%12.8f%12.8f %d  ! %s\n" % (_p[0], _p[1], _p[2], 1, _last))
        else:
            raise functions.MkitsError("Unsupported band-structure k-points format: %s" % code)
            return

        if write2file:
            with open(fpath + "/" + fname, "w", newline="\n") as f:
                f.writelines(_lines)
        else:
            return _lines


class trajectory(struct):
    """
    Multi-frame structure trajectory: an MD run or the ionic steps of a
    relaxation. `self` is itself a normal, fully usable `struct`
    representing the LAST frame -- `super().__init__(inp)` parses it via
    struct's own existing qeout/xdatcar readers (both already keep only
    the final geometry), so nothing new was needed there. `get_frame(i)`
    materializes any other frame as its own independent `struct`, built
    directly from the arrays already parsed here (no re-parsing of the
    source file).

    Supported input formats: QE output (`calculation` = "relax"/
    "vc-relax"/"md"/"vc-md", auto-detected as "qeout" same as struct's
    own format detection) and VASP XDATCAR ("xdatcar", fixed-cell only --
    see `_read_xdatcar_frames`'s own docstring for the same limitation
    struct's own "xdatcar" format has).

    :param inp: the trajectory file path

    Attributes
    ----------
    n_frames: int
    positions_frac: (n_frames, total_atom, 3) numpy array
    lattices: (3, 3) numpy array if the cell is fixed across the whole
        trajectory, else (n_frames, 3, 3)
    """

    def __init__(self, inp):
        super().__init__(inp)
        if self.calculator not in ("qeout", "xdatcar"):
            raise functions.MkitsError(
                "mkits.structure.trajectory only supports 'qeout' and 'xdatcar' input, got '%s'."
                % self.calculator
            )
        self.n_frames = 0
        self.positions_frac = np.zeros((0, self.total_atom, 3))
        self.lattices = self.lattice9.copy()

        with open(inp, "r") as f:
            _lines = f.readlines()
        if self.calculator == "qeout":
            self.__parse_qeout_trajectory(_lines)
        else:
            _, _, self.positions_frac = _read_xdatcar_frames(_lines)
            self.n_frames = self.positions_frac.shape[0]

    def __parse_qeout_trajectory(self, lines):
        """
        Parse every ionic step's lattice + atomic positions from a QE
        output file.

        **LOW/MEDIUM CONFIDENCE on the per-step lattice**: pw.x reprints a
        "CELL_PARAMETERS" block ahead of every "ATOMIC_POSITIONS" block
        for calculation='relax'/'vc-relax' regardless of whether the cell
        actually changed, so when their counts match one-to-one they are
        paired up positionally here; for calculation='md' (fixed-cell,
        does not reprint an unchanged cell every step) the counts will not
        match, in which case every frame falls back to sharing the single
        lattice `struct.__parse_qeout` already found (the last
        CELL_PARAMETERS block if one was printed at all, else the initial
        "crystal axes" preamble block) -- not verified against a live
        calculation='md'/'vc-md' output, only 'relax'; check your
        installed QE version's actual output before trusting per-step
        lattices from an MD run.
        """
        _latticeunit = 1.0
        for _line in lines:
            if "lattice parameter" in _line:
                _latticeunit = float(_line.split()[-2])
                break

        _atomic_position_idx = [_i for _i, _l in enumerate(lines) if "ATOMIC_POSITIONS" in _l]
        _cell_parameter_idx = [_i for _i, _l in enumerate(lines) if "CELL_PARAMETERS" in _l]
        if not _atomic_position_idx:
            raise functions.MkitsError("No ATOMIC_POSITIONS blocks found for a trajectory read of %s." % self.inp)

        _nframe = len(_atomic_position_idx)
        _fixed_cell = len(_cell_parameter_idx) != _nframe
        if _fixed_cell:
            _lattices = np.tile(self.lattice9, (_nframe, 1, 1))
        else:
            _lattices = np.array([
                _qeout_cell_block(lines, _i, _latticeunit) for _i in _cell_parameter_idx
            ])

        _positions = np.zeros((_nframe, self.total_atom, 3))
        for _f, _i in enumerate(_atomic_position_idx):
            _symbols, _coords, _is_crystal = _qeout_positions_block(lines, _i, self.total_atom)
            if _is_crystal:
                _positions[_f] = _coords
            else:
                _positions[_f] = functions.cart2frac(functions.lattice_conversion(_lattices[_f]), _coords)

        self.n_frames = _nframe
        self.positions_frac = _positions
        self.lattices = _lattices[0] if _fixed_cell else _lattices

    def get_frame(self, i):
        """
        Materialize frame `i` (0-based) as its own independent `struct`,
        built directly from the already-parsed trajectory arrays -- no
        re-parsing of the source file. Element identity/order (and
        selective-dynamics/magmom columns) is reused as-is from the LAST
        frame (`self.position`, from `super().__init__`), since only
        coordinates (and, for a variable cell, the lattice) actually vary
        frame to frame.
        """
        if not 0 <= i < self.n_frames:
            raise functions.MkitsError("Frame index %d out of range (0..%d)." % (i, self.n_frames - 1))

        _frame = struct("none")
        _frame.title = "%s frame %d" % (self.title, i)
        _lattice9 = self.lattices if self.lattices.ndim == 2 else self.lattices[i]
        _frame.lattice9 = _lattice9
        _frame.lattice6 = functions.lattice_conversion(_lattice9)
        _frame.total_atom = self.total_atom

        _atom_index = self.position[1:, 0:1]
        _frac = self.positions_frac[i]
        _cart = functions.frac2cart(_lattice9, _frac)
        _dyn = self.position[1:, 7:10]
        _magmom = self.position[1:, 10:11]

        _block = np.hstack((_atom_index, _frac, _cart, _dyn, _magmom))
        _frame.position = np.vstack((_frame.position, _block))
        return _frame
