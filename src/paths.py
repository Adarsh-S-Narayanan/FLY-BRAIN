"""Resource path resolution for source runs AND frozen bundles (PyInstaller).

Read-only resources (datasets, shaders, UI assets, manifests) ship inside the
bundle; user/scientific data (diagnostics/, checkpoints, models) stays relative
to the working directory so a portable build never writes into its own program
files.
"""
import os
import sys


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def base_dir() -> str:
    """Directory containing bundled read-only resources."""
    if is_frozen():
        # PyInstaller onefile: extracted bundle root
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return meipass
        return os.path.dirname(sys.executable)
    # source run: project root (two levels above this file)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource(rel: str) -> str:
    """Absolute path to a bundled read-only resource."""
    return os.path.join(base_dir(), *rel.split("/"))


def data_dir(rel: str = "") -> str:
    """Absolute path for user/scientific data (relative to CWD)."""
    return os.path.join(os.getcwd(), rel) if rel else os.getcwd()
