# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for mconvert.

Produces a single-file, windowed executable meant for students who have
never installed Python: no console window, and every dependency -- including
spglib's dynamically-loaded native library, see collect_all() below -- is
bundled into the one .exe.

PyInstaller does not cross-compile: this must be run ON the target OS to
produce a matching binary (run on Windows for a Windows .exe). See
.github/workflows/build-windows.yml for a CI build that does this without
needing a personal Windows machine.

Usage:
    pyinstaller build.spec
"""

from PyInstaller.utils.hooks import collect_all

datas = [("mconvert/resources", "mconvert/resources")]
binaries = []
hiddenimports = []

# spglib loads a shared library (lib64/libsymspg.so on Linux, a .dll on
# Windows) from a path relative to its own package directory at runtime.
# PyInstaller's automatic dependency scan does not reliably follow that kind
# of relative, non-import-time load, so pull in the whole installed package
# tree (submodules, data files, binaries) exactly as installed instead of
# hand-listing individual files.
for package_name in ("spglib",):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    ["mconvert/__main__.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="mconvert",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="mconvert/resources/icon.ico",
)
