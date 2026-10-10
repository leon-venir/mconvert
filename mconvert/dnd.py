"""Drag-and-drop payload helpers shared by every tab that accepts drops."""

import os


def single_local_file(mime_data):
    """The local file path carried by a drag-and-drop payload, or None if it
    doesn't carry exactly one local file (a remote URL, several files, plain
    text, ...)."""
    urls = mime_data.urls()
    if len(urls) != 1:
        return None
    path = urls[0].toLocalFile()
    return path if path and os.path.isfile(path) else None


def single_local_directory(mime_data):
    """Same as single_local_file but for a single local directory."""
    urls = mime_data.urls()
    if len(urls) != 1:
        return None
    path = urls[0].toLocalFile()
    return path if path and os.path.isdir(path) else None
