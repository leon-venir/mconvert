#!/usr/bin/env python3
"""Vendor mkits.structure (and its dependencies) into mconvert/engine/.

mconvert does not depend on mkits at runtime (see design.md): the structure
parsing/writing engine is copied from the sibling mkits repository into
mconvert/engine/ so the packaged app is self-contained. Files under
mconvert/engine/ are generated output and must not be hand-edited; fix bugs
upstream in mkits and re-run this script.

Usage:
    python tools/sync_engine.py [--source PATH] [--dry-run]
"""

import argparse
import datetime
import re
import subprocess
from pathlib import Path

VENDORED_FILES = ["structure.py", "functions.py", "database.py"]

MCONVERT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path("/home/leon/Public/z_research/mkits/mkits")
ENGINE_DIR = MCONVERT_ROOT / "mconvert" / "engine"

# Rewrite the package-absolute imports used inside the mkits repo to
# package-relative ones, since the vendored files live in mconvert.engine
# instead of the mkits package.
IMPORT_PATTERN = re.compile(r"^from mkits import (\w+)$", re.MULTILINE)

ENGINE_INIT_CONTENT = '''"""Vendored structure-conversion engine, synced from mkits.

Do not edit files in this package by hand: they are copied verbatim (with
import paths rewritten) from the mkits repository by tools/sync_engine.py.
Fix bugs upstream in mkits, then re-run the sync script.
"""
'''


def rewrite_imports(source_text: str) -> str:
    """Rewrite `from mkits import X` to the package-relative `from . import X`."""
    return IMPORT_PATTERN.sub(r"from . import \1", source_text)


def git_commit_info(repo_dir: Path) -> str:
    """Return a human-readable '<short-hash>[-dirty] (<date>)' for repo_dir."""
    try:
        commit_hash = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        commit_date = subprocess.run(
            ["git", "-C", str(repo_dir), "log", "-1", "--format=%cI"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return "%s%s (%s)" % (commit_hash, "-dirty" if dirty else "", commit_date)
    except (OSError, subprocess.CalledProcessError):
        return "unknown (source is not a git repository or git is unavailable)"


def sync(source_dir: Path, dry_run: bool) -> None:
    if not source_dir.is_dir():
        raise SystemExit("Source directory not found: %s" % source_dir)

    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    changed = []

    for filename in VENDORED_FILES:
        src_path = source_dir / filename
        if not src_path.is_file():
            raise SystemExit("Expected source file missing: %s" % src_path)

        new_text = rewrite_imports(src_path.read_text(encoding="utf-8"))
        dst_path = ENGINE_DIR / filename
        old_text = dst_path.read_text(encoding="utf-8") if dst_path.is_file() else None

        if new_text != old_text:
            changed.append(filename)
            if not dry_run:
                dst_path.write_text(new_text, encoding="utf-8", newline="\n")

    init_path = ENGINE_DIR / "__init__.py"
    if not init_path.is_file() or init_path.read_text(encoding="utf-8") != ENGINE_INIT_CONTENT:
        changed.append("__init__.py")
        if not dry_run:
            init_path.write_text(ENGINE_INIT_CONTENT, encoding="utf-8", newline="\n")

    if dry_run:
        print("Would update: %s" % ", ".join(changed) if changed else "Engine is already up to date with %s" % source_dir)
        return

    mkits_repo_root = source_dir.parent
    vendored_from = (
        "source: %s\n"
        "mkits commit: %s\n"
        "synced at: %s\n"
        "files: %s\n"
        % (
            source_dir,
            git_commit_info(mkits_repo_root),
            datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            ", ".join(VENDORED_FILES),
        )
    )
    (ENGINE_DIR / "VENDORED_FROM.txt").write_text(vendored_from, encoding="utf-8", newline="\n")

    # Make mconvert importable so the vendored engine can be exercised
    # end-to-end (`from mconvert.engine.structure import struct`) before the
    # rest of the app scaffold exists.
    package_init = MCONVERT_ROOT / "mconvert" / "__init__.py"
    if not package_init.is_file():
        package_init.write_text("", encoding="utf-8")

    print("Synced: %s" % ", ".join(changed) if changed else "Engine is already up to date with %s" % source_dir)
    print("Wrote %s" % (ENGINE_DIR / "VENDORED_FROM.txt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE,
        help="Path to the mkits package directory to vendor from (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing any files",
    )
    args = parser.parse_args()
    sync(args.source.resolve(), args.dry_run)


if __name__ == "__main__":
    main()
