# -*- coding: utf-8 -*-
"""Shared constant tables used by mkits.structure and other mkits modules
(mkits.builder's adsorption molecule geometries included).

DFT-code-specific templates (INCAR/QE keyword tables, gnuplot scripts, ...)
live in their respective modules (vasp.py, qe.py, ...) instead.
"""

import math

import numpy as np


# ============================================================== #
# unit conversion
# ============================================================== #
uc_bohr2ang = 0.529177  # bohr to angstrom
uc_ry2ev = 13.605693  # rydberg to eV
uc_ha2ev = 27.211386  # hartree to eV


# ============================================================== #
# periodic table
# ------------------------------------------------------------ #
# atom_data[Z] = [Z, symbol, name, atomic_mass, magmom]
#   atomic_mass: standard atomic weight in amu, None if unknown/unstable.
#   magmom: recommended initial magnetic moment (mu_B) used to seed
#           spin-polarized DFT calculations; 0.0 for elements without a
#           commonly used non-zero starting guess.
# Z = 0 is reserved as a placeholder for an unrecognized element ("X").
# ============================================================== #
atom_data = [
    [0, "X", "Unknown", None, 0.0],
    [1, "H", "Hydrogen", 1.00794, 0.0],
    [2, "He", "Helium", 4.002602, 0.0],
    [3, "Li", "Lithium", 6.941, 0.0],
    [4, "Be", "Beryllium", 9.012182, 0.0],
    [5, "B", "Boron", 10.811, 0.0],
    [6, "C", "Carbon", 12.0107, 1.0],
    [7, "N", "Nitrogen", 14.0067, 1.0],
    [8, "O", "Oxygen", 15.9994, 0.6],
    [9, "F", "Fluorine", 18.9984032, 0.6],
    [10, "Ne", "Neon", 20.1797, 0.0],
    [11, "Na", "Sodium", 22.98976928, 0.0],
    [12, "Mg", "Magnesium", 24.3050, 0.0],
    [13, "Al", "Aluminium", 26.9815386, 0.0],
    [14, "Si", "Silicon", 28.0855, 0.0],
    [15, "P", "Phosphorus", 30.973762, 0.0],
    [16, "S", "Sulfur", 32.065, 0.0],
    [17, "Cl", "Chlorine", 35.453, 0.0],
    [18, "Ar", "Argon", 39.948, 0.0],
    [19, "K", "Potassium", 39.0983, 0.0],
    [20, "Ca", "Calcium", 40.078, 0.0],
    [21, "Sc", "Scandium", 44.955912, 0.0],
    [22, "Ti", "Titanium", 47.867, 0.6],
    [23, "V", "Vanadium", 50.9415, 1.0],
    [24, "Cr", "Chromium", 51.9961, -2.0],
    [25, "Mn", "Manganese", 54.938045, 3.0],
    [26, "Fe", "Iron", 55.845, 3.0],
    [27, "Co", "Cobalt", 58.933195, 3.0],
    [28, "Ni", "Nickel", 58.6934, 1.0],
    [29, "Cu", "Copper", 63.546, 0.0],
    [30, "Zn", "Zinc", 65.38, 0.0],
    [31, "Ga", "Gallium", 69.723, 0.0],
    [32, "Ge", "Germanium", 72.64, 0.0],
    [33, "As", "Arsenic", 74.92160, 0.0],
    [34, "Se", "Selenium", 78.96, 0.0],
    [35, "Br", "Bromine", 79.904, 0.0],
    [36, "Kr", "Krypton", 83.798, 0.0],
    [37, "Rb", "Rubidium", 85.4678, 0.0],
    [38, "Sr", "Strontium", 87.62, 0.0],
    [39, "Y", "Yttrium", 88.90585, 0.0],
    [40, "Zr", "Zirconium", 91.224, 0.0],
    [41, "Nb", "Niobium", 92.90638, 0.0],
    [42, "Mo", "Molybdenum", 95.96, 2.0],
    [43, "Tc", "Technetium", None, 0.0],
    [44, "Ru", "Ruthenium", 101.07, 2.0],
    [45, "Rh", "Rhodium", 102.90550, 0.0],
    [46, "Pd", "Palladium", 106.42, 0.0],
    [47, "Ag", "Silver", 107.8682, 0.0],
    [48, "Cd", "Cadmium", 112.411, 0.0],
    [49, "In", "Indium", 114.818, 0.0],
    [50, "Sn", "Tin", 118.710, 0.0],
    [51, "Sb", "Antimony", 121.760, 0.0],
    [52, "Te", "Tellurium", 127.60, 0.0],
    [53, "I", "Iodine", 126.90447, 0.0],
    [54, "Xe", "Xenon", 131.293, 0.0],
    [55, "Cs", "Caesium", 132.9054519, 0.0],
    [56, "Ba", "Barium", 137.327, 0.0],
    [57, "La", "Lanthanum", 138.90547, 0.0],
    [58, "Ce", "Cerium", 140.116, 0.0],
    [59, "Pr", "Praseodymium", 140.90765, 0.0],
    [60, "Nd", "Neodymium", 144.242, 0.0],
    [61, "Pm", "Promethium", None, 0.0],
    [62, "Sm", "Samarium", 150.36, 0.0],
    [63, "Eu", "Europium", 151.964, 0.0],
    [64, "Gd", "Gadolinium", 157.25, 0.0],
    [65, "Tb", "Terbium", 158.92535, 0.0],
    [66, "Dy", "Dysprosium", 162.500, 0.0],
    [67, "Ho", "Holmium", 164.93032, 0.0],
    [68, "Er", "Erbium", 167.259, 0.0],
    [69, "Tm", "Thulium", 168.93421, 0.0],
    [70, "Yb", "Ytterbium", 173.054, 0.0],
    [71, "Lu", "Lutetium", 174.9668, 0.0],
    [72, "Hf", "Hafnium", 178.49, 0.0],
    [73, "Ta", "Tantalum", 180.94788, 0.0],
    [74, "W", "Tungsten", 183.84, 2.0],
    [75, "Re", "Rhenium", 186.207, 0.0],
    [76, "Os", "Osmium", 190.23, 0.0],
    [77, "Ir", "Iridium", 192.217, 0.0],
    [78, "Pt", "Platinum", 195.084, 0.0],
    [79, "Au", "Gold", 196.966569, 0.0],
    [80, "Hg", "Mercury", 200.59, 0.0],
    [81, "Tl", "Thallium", 204.3833, 0.0],
    [82, "Pb", "Lead", 207.2, 0.0],
    [83, "Bi", "Bismuth", 208.98040, 0.0],
    [84, "Po", "Polonium", None, 0.0],
    [85, "At", "Astatine", None, 0.0],
    [86, "Rn", "Radon", None, 0.0],
    [87, "Fr", "Francium", None, 0.0],
    [88, "Ra", "Radium", None, 0.0],
    [89, "Ac", "Actinium", None, 0.0],
    [90, "Th", "Thorium", 232.03806, 0.0],
    [91, "Pa", "Protactinium", 231.03588, 0.0],
    [92, "U", "Uranium", 238.02891, 0.0],
    [93, "Np", "Neptunium", None, 0.0],
    [94, "Pu", "Plutonium", None, 0.0],
    [95, "Am", "Americium", None, 0.0],
    [96, "Cm", "Curium", None, 0.0],
    [97, "Bk", "Berkelium", None, 0.0],
    [98, "Cf", "Californium", None, 0.0],
    [99, "Es", "Einsteinium", None, 0.0],
    [100, "Fm", "Fermium", None, 0.0],
    [101, "Md", "Mendelevium", None, 0.0],
    [102, "No", "Nobelium", None, 0.0],
    [103, "Lr", "Lawrencium", None, 0.0],
    [104, "Rf", "Rutherfordium", None, 0.0],
    [105, "Db", "Dubnium", None, 0.0],
    [106, "Sg", "Seaborgium", None, 0.0],
    [107, "Bh", "Bohrium", None, 0.0],
    [108, "Hs", "Hassium", None, 0.0],
    [109, "Mt", "Meitnerium", None, 0.0],
    [110, "Ds", "Darmstadtium", None, 0.0],
    [111, "Rg", "Roentgenium", None, 0.0],
    [112, "Cn", "Copernicium", None, 0.0],
    [113, "Uut", "Ununtrium", None, 0.0],
    [114, "Uuq", "Ununquadium", None, 0.0],
    [115, "Uup", "Ununpentium", None, 0.0],
    [116, "Uuh", "Ununhexium", None, 0.0],
    [117, "Uus", "Ununseptium", None, 0.0],
    [118, "Uuo", "Ununoctium", None, 0.0],
]

symbol_map = {row[1]: row[0] for row in atom_data if row[0] != 0}


# ============================================================== #
# mkits.builder.adsorpt_molecue: adsorbate molecule geometries
# ------------------------------------------------------------ #
# Row 0 of each table is a placeholder "central position" (unused by
# adsorpt_molecue, kept for layout parity with atom_data); real atoms are
# rows 1: [atomic_number, x, y, z] in angstrom, already centered so that
# distance/orientation offsets can be applied directly.
# ============================================================== #
mol_h2o = np.array([
    [0, 0, 0, 0],
    [8, 0, 0, 0],
    [1, 0, -0.76913, 0.59479],
    [1, 0, 0.76913, 0.59479],
])

mol_h2 = np.array([
    [0, 0, 0, 0],
    [1, 0, -0.35913, 0],
    [1, 0, 0.35913, 0],
])

mol_oh = np.array([
    [0, 0, 0, 0],
    [8, 0, 0, 0],
    [1, 0, 0.76913, 0.59479],
])

mol_ooh = np.array([
    [0, 0, 0, 0],
    [8, 0, 0, 0],
    [8, 0.86900, 1.07800, 0.52270],
    [1, 1.67520, 1.39700, 1.21580],
])

mol_h = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
])

mol_o = np.array([
    [0, 0, 0, 0],
    [8, 0, 0, 0],
])

mol_o2 = np.array([
    [0, 0, 0, 0],
    [8, 0, -0.61913, 0],
    [8, 0, 0.61913, 0],
])


# ============================================================== #
# WIEN2k templates
# ------------------------------------------------------------ #
# Identity local rotation matrix, used when writing case.struct files.
# Symmetry-adapted local rotation matrices are left to WIEN2k's own
# symmetry utilities (x nn / symmetso) run after the file is generated.
# ============================================================== #
local_rot_matrix = """LOCAL ROT MATRIX:    1.0000000 0.0000000 0.0000000
                     0.0000000 1.0000000 0.0000000
                     0.0000000 0.0000000 1.0000000
"""


# ============================================================== #
# high-symmetry k-paths
# ------------------------------------------------------------ #
# kpath_3d: the Setyawan-Curtarolo (SC) paths for all 14 Bravais lattices
# and their lattice-parameter-dependent variants (Comput. Mater. Sci. 49,
# 299 (2010)), keyed by variant name (CUB, FCC, BCC, TET, BCT1, BCT2, ORC,
# ORCF1-3, ORCI, ORCC, HEX, RHL1, RHL2, MCL, MCLC1-5, TRI1a, TRI1b, TRI2a,
# TRI2b). Unlike a table keyed by crystal system alone, the variant already
# encodes the lattice centering (P/I/F/C/R) and, where the Brillouin-zone
# topology depends on it, the lattice-parameter regime (eg c<a vs c>a for
# BCT).
#
# Every point is given in fractional coordinates of the reciprocal basis of
# the SC *standard primitive cell* (not of an arbitrary input cell) --
# mkits.structure.seekpath finds the transformation from the input cell to
# that standard primitive cell and maps the points back, so a table entry
# is only meaningful together with that mapping.
#
# Each entry is {"points": ..., "path": ...}:
#   points: a dict label -> [x, y, z] when the points are fixed numbers,
#       or a function of the standard cell's lattice parameters
#       (p["a"], p["b"], p["c"] in any common length unit, p["alpha"] in
#       degrees) returning such a dict when they depend on them
#   path: list of continuous segments, each a list of point labels; a new
#       segment starts wherever the SC path has a "|" discontinuity
# Greek-letter labels are spelled out ("GAMMA", "SIGMA", "SIGMA1").
#
# kpath_2d (below) is still a first, self-built approximation that only
# distinguishes the four primitive 2D Bravais lattice types (no centered
# rectangular lattice).
# ============================================================== #
def _kpoints_bct1(p):
    _eta = (1.0 + p["c"] ** 2 / p["a"] ** 2) / 4.0
    return {
        "GAMMA": [0.0, 0.0, 0.0],
        "M": [-0.5, 0.5, 0.5],
        "N": [0.0, 0.5, 0.0],
        "P": [0.25, 0.25, 0.25],
        "X": [0.0, 0.0, 0.5],
        "Z": [_eta, _eta, -_eta],
        "Z1": [-_eta, 1.0 - _eta, _eta],
    }


def _kpoints_bct2(p):
    _eta = (1.0 + p["a"] ** 2 / p["c"] ** 2) / 4.0
    _zeta = p["a"] ** 2 / (2.0 * p["c"] ** 2)
    return {
        "GAMMA": [0.0, 0.0, 0.0],
        "N": [0.0, 0.5, 0.0],
        "P": [0.25, 0.25, 0.25],
        "SIGMA": [-_eta, _eta, _eta],
        "SIGMA1": [_eta, 1.0 - _eta, -_eta],
        "X": [0.0, 0.0, 0.5],
        "Y": [-_zeta, _zeta, 0.5],
        "Y1": [0.5, 0.5, -_zeta],
        "Z": [0.5, 0.5, -0.5],
    }


def _kpoints_orcf1(p):
    _a, _b, _c = p["a"], p["b"], p["c"]
    _zeta = (1.0 + _a ** 2 / _b ** 2 - _a ** 2 / _c ** 2) / 4.0
    _eta = (1.0 + _a ** 2 / _b ** 2 + _a ** 2 / _c ** 2) / 4.0
    return {
        "GAMMA": [0.0, 0.0, 0.0],
        "A": [0.5, 0.5 + _zeta, _zeta],
        "A1": [0.5, 0.5 - _zeta, 1.0 - _zeta],
        "L": [0.5, 0.5, 0.5],
        "T": [1.0, 0.5, 0.5],
        "X": [0.0, _eta, _eta],
        "X1": [1.0, 1.0 - _eta, 1.0 - _eta],
        "Y": [0.5, 0.0, 0.5],
        "Z": [0.5, 0.5, 0.0],
    }


def _kpoints_orcf2(p):
    _a, _b, _c = p["a"], p["b"], p["c"]
    _phi = (1.0 + _c ** 2 / _b ** 2 - _c ** 2 / _a ** 2) / 4.0
    _eta = (1.0 + _a ** 2 / _b ** 2 - _a ** 2 / _c ** 2) / 4.0
    _delta = (1.0 + _b ** 2 / _a ** 2 - _b ** 2 / _c ** 2) / 4.0
    return {
        "GAMMA": [0.0, 0.0, 0.0],
        "C": [0.5, 0.5 - _eta, 1.0 - _eta],
        "C1": [0.5, 0.5 + _eta, _eta],
        "D": [0.5 - _delta, 0.5, 1.0 - _delta],
        "D1": [0.5 + _delta, 0.5, _delta],
        "L": [0.5, 0.5, 0.5],
        "H": [1.0 - _phi, 0.5 - _phi, 0.5],
        "H1": [_phi, 0.5 + _phi, 0.5],
        "X": [0.0, 0.5, 0.5],
        "Y": [0.5, 0.0, 0.5],
        "Z": [0.5, 0.5, 0.0],
    }


def _kpoints_orci(p):
    _a, _b, _c = p["a"], p["b"], p["c"]
    _zeta = (1.0 + _a ** 2 / _c ** 2) / 4.0
    _eta = (1.0 + _b ** 2 / _c ** 2) / 4.0
    _delta = (_b ** 2 - _a ** 2) / (4.0 * _c ** 2)
    _mu = (_a ** 2 + _b ** 2) / (4.0 * _c ** 2)
    return {
        "GAMMA": [0.0, 0.0, 0.0],
        "L": [-_mu, _mu, 0.5 - _delta],
        "L1": [_mu, -_mu, 0.5 + _delta],
        "L2": [0.5 - _delta, 0.5 + _delta, -_mu],
        "R": [0.0, 0.5, 0.0],
        "S": [0.5, 0.0, 0.0],
        "T": [0.0, 0.0, 0.5],
        "W": [0.25, 0.25, 0.25],
        "X": [-_zeta, _zeta, _zeta],
        "X1": [_zeta, 1.0 - _zeta, -_zeta],
        "Y": [_eta, -_eta, _eta],
        "Y1": [1.0 - _eta, _eta, -_eta],
        "Z": [0.5, 0.5, -0.5],
    }


def _kpoints_orcc(p):
    _zeta = (1.0 + p["a"] ** 2 / p["b"] ** 2) / 4.0
    return {
        "GAMMA": [0.0, 0.0, 0.0],
        "A": [_zeta, _zeta, 0.5],
        "A1": [-_zeta, 1.0 - _zeta, 0.5],
        "R": [0.0, 0.5, 0.5],
        "S": [0.0, 0.5, 0.0],
        "T": [-0.5, 0.5, 0.5],
        "X": [_zeta, _zeta, 0.0],
        "X1": [-_zeta, 1.0 - _zeta, 0.0],
        "Y": [-0.5, 0.5, 0.0],
        "Z": [0.0, 0.0, 0.5],
    }


def _kpoints_rhl1(p):
    _cos = math.cos(math.radians(p["alpha"]))
    _eta = (1.0 + 4.0 * _cos) / (2.0 + 4.0 * _cos)
    _nu = 0.75 - _eta / 2.0
    return {
        "GAMMA": [0.0, 0.0, 0.0],
        "B": [_eta, 0.5, 1.0 - _eta],
        "B1": [0.5, 1.0 - _eta, _eta - 1.0],
        "F": [0.5, 0.5, 0.0],
        "L": [0.5, 0.0, 0.0],
        "L1": [0.0, 0.0, -0.5],
        "P": [_eta, _nu, _nu],
        "P1": [1.0 - _nu, 1.0 - _nu, 1.0 - _eta],
        "P2": [_nu, _nu, _eta - 1.0],
        "Q": [1.0 - _nu, _nu, 0.0],
        "X": [_nu, 0.0, -_nu],
        "Z": [0.5, 0.5, 0.5],
    }


def _kpoints_rhl2(p):
    _eta = 1.0 / (2.0 * math.tan(math.radians(p["alpha"]) / 2.0) ** 2)
    _nu = 0.75 - _eta / 2.0
    return {
        "GAMMA": [0.0, 0.0, 0.0],
        "F": [0.5, -0.5, 0.0],
        "L": [0.5, 0.0, 0.0],
        "P": [1.0 - _nu, -_nu, 1.0 - _nu],
        "P1": [_nu, _nu - 1.0, _nu - 1.0],
        "Q": [_eta, _eta, _eta],
        "Q1": [1.0 - _eta, -_eta, -_eta],
        "Z": [0.5, -0.5, 0.5],
    }


def _kpoints_mcl(p):
    _alpha = math.radians(p["alpha"])
    _eta = (1.0 - p["b"] * math.cos(_alpha) / p["c"]) / (2.0 * math.sin(_alpha) ** 2)
    _nu = 0.5 - _eta * p["c"] * math.cos(_alpha) / p["b"]
    return {
        "GAMMA": [0.0, 0.0, 0.0],
        "A": [0.5, 0.5, 0.0],
        "C": [0.0, 0.5, 0.5],
        "D": [0.5, 0.0, 0.5],
        "D1": [0.5, 0.0, -0.5],
        "E": [0.5, 0.5, 0.5],
        "H": [0.0, _eta, 1.0 - _nu],
        "H1": [0.0, 1.0 - _eta, _nu],
        "H2": [0.0, _eta, -_nu],
        "M": [0.5, _eta, 1.0 - _nu],
        "M1": [0.5, 1.0 - _eta, _nu],
        "M2": [0.5, _eta, -_nu],
        "X": [0.0, 0.5, 0.0],
        "Y": [0.0, 0.0, 0.5],
        "Y1": [0.0, 0.0, -0.5],
        "Z": [0.5, 0.0, 0.0],
    }


def _kpoints_mclc12(p, with_f3):
    # shared by MCLC1 (k_gamma > 90) and MCLC2 (k_gamma = 90); only MCLC2
    # additionally has F3. (pymatgen's MCLC2 table gives F1 = (zeta,
    # zeta - 1, eta) instead of the (zeta, zeta, eta) used here -- that
    # point lies outside the first Brillouin zone for a kgamma = 90 lattice
    # while (zeta, zeta, eta), which is also what ASE uses, lies inside it,
    # so it looks like a transcription slip on pymatgen's side.)
    _a, _b, _c = p["a"], p["b"], p["c"]
    _alpha = math.radians(p["alpha"])
    _zeta = (2.0 - _b * math.cos(_alpha) / _c) / (4.0 * math.sin(_alpha) ** 2)
    _eta = 0.5 + 2.0 * _zeta * _c * math.cos(_alpha) / _b
    _psi = 0.75 - _a ** 2 / (4.0 * _b ** 2 * math.sin(_alpha) ** 2)
    _phi = _psi + (0.75 - _psi) * _b * math.cos(_alpha) / _c
    _points = {
        "GAMMA": [0.0, 0.0, 0.0],
        "N": [0.5, 0.0, 0.0],
        "N1": [0.0, -0.5, 0.0],
        "F": [1.0 - _zeta, 1.0 - _zeta, 1.0 - _eta],
        "F1": [_zeta, _zeta, _eta],
        "F2": [-_zeta, -_zeta, 1.0 - _eta],
        "I": [_phi, 1.0 - _phi, 0.5],
        "I1": [1.0 - _phi, _phi - 1.0, 0.5],
        "L": [0.5, 0.5, 0.5],
        "M": [0.5, 0.0, 0.5],
        "X": [1.0 - _psi, _psi - 1.0, 0.0],
        "X1": [_psi, 1.0 - _psi, 0.0],
        "X2": [_psi - 1.0, -_psi, 0.0],
        "Y": [0.5, 0.5, 0.0],
        "Y1": [-0.5, -0.5, 0.0],
        "Z": [0.0, 0.0, 0.5],
    }
    if with_f3:
        _points["F3"] = [1.0 - _zeta, -_zeta, 1.0 - _eta]
    return _points


def _kpoints_mclc1(p):
    return _kpoints_mclc12(p, False)


def _kpoints_mclc2(p):
    return _kpoints_mclc12(p, True)


def _kpoints_mclc34(p):
    # shared by MCLC3 (b cos(alpha)/c + b^2 sin^2(alpha)/a^2 < 1) and MCLC4
    # (= 1): identical points, different paths
    _a, _b, _c = p["a"], p["b"], p["c"]
    _alpha = math.radians(p["alpha"])
    _mu = (1.0 + _b ** 2 / _a ** 2) / 4.0
    _delta = _b * _c * math.cos(_alpha) / (2.0 * _a ** 2)
    _zeta = _mu - 0.25 + (1.0 - _b * math.cos(_alpha) / _c) / (4.0 * math.sin(_alpha) ** 2)
    _eta = 0.5 + 2.0 * _zeta * _c * math.cos(_alpha) / _b
    _phi = 1.0 + _zeta - 2.0 * _mu
    _psi = _eta - 2.0 * _delta
    return {
        "GAMMA": [0.0, 0.0, 0.0],
        "F": [1.0 - _phi, 1.0 - _phi, 1.0 - _psi],
        "F1": [_phi, _phi - 1.0, _psi],
        "F2": [1.0 - _phi, -_phi, 1.0 - _psi],
        "H": [_zeta, _zeta, _eta],
        "H1": [1.0 - _zeta, -_zeta, 1.0 - _eta],
        "H2": [-_zeta, -_zeta, 1.0 - _eta],
        "I": [0.5, -0.5, 0.5],
        "M": [0.5, 0.0, 0.5],
        "N": [0.5, 0.0, 0.0],
        "N1": [0.0, -0.5, 0.0],
        "X": [0.5, -0.5, 0.0],
        "Y": [_mu, _mu, _delta],
        "Y1": [1.0 - _mu, -_mu, -_delta],
        "Y2": [-_mu, -_mu, -_delta],
        "Y3": [_mu, _mu - 1.0, _delta],
        "Z": [0.0, 0.0, 0.5],
    }


def _kpoints_mclc5(p):
    _a, _b, _c = p["a"], p["b"], p["c"]
    _alpha = math.radians(p["alpha"])
    _zeta = (_b ** 2 / _a ** 2 + (1.0 - _b * math.cos(_alpha) / _c) / math.sin(_alpha) ** 2) / 4.0
    _eta = 0.5 + 2.0 * _zeta * _c * math.cos(_alpha) / _b
    _mu = _eta / 2.0 + _b ** 2 / (4.0 * _a ** 2) - _b * _c * math.cos(_alpha) / (2.0 * _a ** 2)
    _nu = 2.0 * _mu - _zeta
    _rho = 1.0 - _zeta * _a ** 2 / _b ** 2
    _omega = (4.0 * _nu - 1.0 - _b ** 2 * math.sin(_alpha) ** 2 / _a ** 2) * _c / (2.0 * _b * math.cos(_alpha))
    _delta = _zeta * _c * math.cos(_alpha) / _b + _omega / 2.0 - 0.25
    return {
        "GAMMA": [0.0, 0.0, 0.0],
        "F": [_nu, _nu, _omega],
        "F1": [1.0 - _nu, 1.0 - _nu, 1.0 - _omega],
        "F2": [_nu, _nu - 1.0, _omega],
        "H": [_zeta, _zeta, _eta],
        "H1": [1.0 - _zeta, -_zeta, 1.0 - _eta],
        "H2": [-_zeta, -_zeta, 1.0 - _eta],
        "I": [_rho, 1.0 - _rho, 0.5],
        "I1": [1.0 - _rho, _rho - 1.0, 0.5],
        "L": [0.5, 0.5, 0.5],
        "M": [0.5, 0.0, 0.5],
        "N": [0.5, 0.0, 0.0],
        "N1": [0.0, -0.5, 0.0],
        "X": [0.5, -0.5, 0.0],
        "Y": [_mu, _mu, _delta],
        "Y1": [1.0 - _mu, -_mu, -_delta],
        "Y2": [-_mu, -_mu, -_delta],
        "Y3": [_mu, _mu - 1.0, _delta],
        "Z": [0.0, 0.0, 0.5],
    }


def _kpoints_orcf3(p):
    # same points as ORCF1 (the two only differ in which side of
    # 1/a^2 = 1/b^2 + 1/c^2 the lattice sits on), shorter path
    return _kpoints_orcf1(p)


_KPOINTS_TRI_A = {
    "GAMMA": [0.0, 0.0, 0.0],
    "L": [0.5, 0.5, 0.0],
    "M": [0.0, 0.5, 0.5],
    "N": [0.5, 0.0, 0.5],
    "R": [0.5, 0.5, 0.5],
    "X": [0.5, 0.0, 0.0],
    "Y": [0.0, 0.5, 0.0],
    "Z": [0.0, 0.0, 0.5],
}

_KPOINTS_TRI_B = {
    "GAMMA": [0.0, 0.0, 0.0],
    "L": [0.5, -0.5, 0.0],
    "M": [0.0, 0.0, 0.5],
    "N": [-0.5, -0.5, 0.5],
    "R": [0.0, -0.5, 0.5],
    "X": [0.0, -0.5, 0.0],
    "Y": [0.5, 0.0, 0.0],
    "Z": [-0.5, 0.0, 0.5],
}

_PATH_TRI = [["X", "GAMMA", "Y"], ["L", "GAMMA", "Z"], ["N", "GAMMA", "M"], ["R", "GAMMA"]]

kpath_3d = {
    "CUB": {
        "points": {
            "GAMMA": [0.0, 0.0, 0.0],
            "M": [0.5, 0.5, 0.0],
            "R": [0.5, 0.5, 0.5],
            "X": [0.0, 0.5, 0.0],
        },
        "path": [["GAMMA", "X", "M", "GAMMA", "R", "X"], ["M", "R"]],
    },
    "FCC": {
        "points": {
            "GAMMA": [0.0, 0.0, 0.0],
            "K": [3.0 / 8.0, 3.0 / 8.0, 0.75],
            "L": [0.5, 0.5, 0.5],
            "U": [5.0 / 8.0, 0.25, 5.0 / 8.0],
            "W": [0.5, 0.25, 0.75],
            "X": [0.5, 0.0, 0.5],
        },
        "path": [["GAMMA", "X", "W", "K", "GAMMA", "L", "U", "W", "L", "K"], ["U", "X"]],
    },
    "BCC": {
        "points": {
            "GAMMA": [0.0, 0.0, 0.0],
            "H": [0.5, -0.5, 0.5],
            "P": [0.25, 0.25, 0.25],
            "N": [0.0, 0.0, 0.5],
        },
        "path": [["GAMMA", "H", "N", "GAMMA", "P", "H"], ["P", "N"]],
    },
    "TET": {
        "points": {
            "GAMMA": [0.0, 0.0, 0.0],
            "A": [0.5, 0.5, 0.5],
            "M": [0.5, 0.5, 0.0],
            "R": [0.0, 0.5, 0.5],
            "X": [0.0, 0.5, 0.0],
            "Z": [0.0, 0.0, 0.5],
        },
        "path": [["GAMMA", "X", "M", "GAMMA", "Z", "R", "A", "Z"], ["X", "R"], ["M", "A"]],
    },
    "BCT1": {
        "points": _kpoints_bct1,
        "path": [["GAMMA", "X", "M", "GAMMA", "Z", "P", "N", "Z1", "M"], ["X", "P"]],
    },
    "BCT2": {
        "points": _kpoints_bct2,
        "path": [["GAMMA", "X", "Y", "SIGMA", "GAMMA", "Z", "SIGMA1", "N", "P", "Y1", "Z"], ["X", "P"]],
    },
    "ORC": {
        "points": {
            "GAMMA": [0.0, 0.0, 0.0],
            "R": [0.5, 0.5, 0.5],
            "S": [0.5, 0.5, 0.0],
            "T": [0.0, 0.5, 0.5],
            "U": [0.5, 0.0, 0.5],
            "X": [0.5, 0.0, 0.0],
            "Y": [0.0, 0.5, 0.0],
            "Z": [0.0, 0.0, 0.5],
        },
        "path": [["GAMMA", "X", "S", "Y", "GAMMA", "Z", "U", "R", "T", "Z"], ["Y", "T"], ["U", "X"], ["S", "R"]],
    },
    "ORCF1": {
        "points": _kpoints_orcf1,
        "path": [["GAMMA", "Y", "T", "Z", "GAMMA", "X", "A1", "Y"], ["T", "X1"], ["X", "A", "Z"], ["L", "GAMMA"]],
    },
    "ORCF2": {
        "points": _kpoints_orcf2,
        "path": [
            ["GAMMA", "Y", "C", "D", "X", "GAMMA", "Z", "D1", "H", "C"],
            ["C1", "Z"], ["X", "H1"], ["H", "Y"], ["L", "GAMMA"],
        ],
    },
    "ORCF3": {
        "points": _kpoints_orcf3,
        "path": [["GAMMA", "Y", "T", "Z", "GAMMA", "X", "A1", "Y"], ["X", "A", "Z"], ["L", "GAMMA"]],
    },
    "ORCI": {
        "points": _kpoints_orci,
        "path": [["GAMMA", "X", "L", "T", "W", "R", "X1", "Z", "GAMMA", "Y", "S", "W"], ["L1", "Y"], ["Y1", "Z"]],
    },
    "ORCC": {
        "points": _kpoints_orcc,
        "path": [["GAMMA", "X", "S", "R", "A", "Z", "GAMMA", "Y", "X1", "A1", "T", "Y"], ["Z", "T"]],
    },
    "HEX": {
        "points": {
            "GAMMA": [0.0, 0.0, 0.0],
            "A": [0.0, 0.0, 0.5],
            "H": [1.0 / 3.0, 1.0 / 3.0, 0.5],
            "K": [1.0 / 3.0, 1.0 / 3.0, 0.0],
            "L": [0.5, 0.0, 0.5],
            "M": [0.5, 0.0, 0.0],
        },
        "path": [["GAMMA", "M", "K", "GAMMA", "A", "L", "H", "A"], ["L", "M"], ["K", "H"]],
    },
    "RHL1": {
        "points": _kpoints_rhl1,
        "path": [["GAMMA", "L", "B1"], ["B", "Z", "GAMMA", "X"], ["Q", "F", "P1", "Z"], ["L", "P"]],
    },
    "RHL2": {
        "points": _kpoints_rhl2,
        "path": [["GAMMA", "P", "Z", "Q", "GAMMA", "F", "P1", "Q1", "L", "Z"]],
    },
    "MCL": {
        "points": _kpoints_mcl,
        "path": [["GAMMA", "Y", "H", "C", "E", "M1", "A", "X", "H1"], ["M", "D", "Z"], ["Y", "D"]],
    },
    "MCLC1": {
        "points": _kpoints_mclc1,
        "path": [["GAMMA", "Y", "F", "L", "I"], ["I1", "Z", "F1"], ["Y", "X1"], ["X", "GAMMA", "N"], ["M", "GAMMA"]],
    },
    "MCLC2": {
        "points": _kpoints_mclc2,
        "path": [["GAMMA", "Y", "F", "L", "I"], ["I1", "Z", "F1"], ["N", "GAMMA", "M"]],
    },
    "MCLC3": {
        "points": _kpoints_mclc34,
        "path": [["GAMMA", "Y", "F", "H", "Z", "I", "F1"], ["H1", "Y1", "X", "GAMMA", "N"], ["M", "GAMMA"]],
    },
    "MCLC4": {
        "points": _kpoints_mclc34,
        "path": [["GAMMA", "Y", "F", "H", "Z", "I"], ["H1", "Y1", "X", "GAMMA", "N"], ["M", "GAMMA"]],
    },
    "MCLC5": {
        "points": _kpoints_mclc5,
        "path": [["GAMMA", "Y", "F", "L", "I"], ["I1", "Z", "H", "F1"], ["H1", "Y1", "X", "GAMMA", "N"], ["M", "GAMMA"]],
    },
    "TRI1a": {"points": _KPOINTS_TRI_A, "path": _PATH_TRI},
    "TRI2a": {"points": _KPOINTS_TRI_A, "path": _PATH_TRI},
    "TRI1b": {"points": _KPOINTS_TRI_B, "path": _PATH_TRI},
    "TRI2b": {"points": _KPOINTS_TRI_B, "path": _PATH_TRI},
}

kpath_2d = {
    "square": {
        "points": {
            "GAMMA": [0.0, 0.0, 0.0],
            "X": [0.5, 0.0, 0.0],
            "M": [0.5, 0.5, 0.0],
        },
        "path": [["GAMMA", "X", "M", "GAMMA"]],
    },
    "rectangular": {
        "points": {
            "GAMMA": [0.0, 0.0, 0.0],
            "X": [0.5, 0.0, 0.0],
            "S": [0.5, 0.5, 0.0],
            "Y": [0.0, 0.5, 0.0],
        },
        "path": [["GAMMA", "X", "S", "Y", "GAMMA"]],
    },
    "hexagonal": {
        "points": {
            "GAMMA": [0.0, 0.0, 0.0],
            "K": [1.0 / 3.0, 1.0 / 3.0, 0.0],
            "M": [0.5, 0.0, 0.0],
        },
        "path": [["GAMMA", "M", "K", "GAMMA"]],
    },
    "oblique": {
        "points": {
            "GAMMA": [0.0, 0.0, 0.0],
            "X": [0.5, 0.0, 0.0],
            "Y": [0.0, 0.5, 0.0],
        },
        "path": [["X", "GAMMA", "Y"]],
    },
}
