"""Compute the version number and store it in the `__version__` variable.

Based on <https://github.com/maresb/hatch-vcs-footgun-example>.

Do not put this directly into `__init__.py` because it can lead to circular imports.
Instead, name this module something like `version.py` and import `__version__`
from this module into your `__init__.py` file and elsewhere in your project.
"""

import os
from pathlib import Path


def _cleanup_stale_egg_info() -> bool:
    """Rename stale celeri.egg-info/ to celeri.egg-info.delme/ if it exists.

    Old setuptools installs leave behind a celeri.egg-info/ directory that
    causes importlib.metadata to return the wrong version. This function
    renames it out of the way so the correct hatch-vcs version is used.

    Returns True if the directory was renamed, False otherwise.
    """
    stale_egg_info = Path(__file__).parent.parent / "celeri.egg-info"
    if stale_egg_info.is_dir():
        stale_egg_info.rename(stale_egg_info.with_suffix(".egg-info.delme"))
        return True
    return False


def _get_hatch_version():
    """Compute the most up-to-date version number in a development environment.

    For more details, see <https://github.com/maresb/hatch-vcs-footgun-example/>.
    """
    from hatchling.metadata.core import ProjectMetadata
    from hatchling.plugin.manager import PluginManager
    from hatchling.utils.fs import locate_file

    pyproject_toml = locate_file(__file__, "pyproject.toml")
    if pyproject_toml is None:
        raise RuntimeError("pyproject.toml not found although hatchling is installed")
    root = str(Path(pyproject_toml).parent)
    metadata = ProjectMetadata(root=root, plugin_manager=PluginManager())
    # Version can be either statically set in pyproject.toml or computed dynamically:
    return metadata.core.version or metadata.hatch.version.cached


def _get_importlib_metadata_version():
    """Compute the version number using importlib.metadata.

    This is the official Pythonic way to get the version number of an installed
    package. However, it is only updated when a package is installed. Thus, if a
    package is installed in editable mode, and a different version is checked out,
    then the version number will not be updated.
    """
    from importlib.metadata import version

    if __package__ is None:
        raise RuntimeError(
            f"__package__ not set in '{__file__}' - ensure that you are running this "
            "module as part of a package, e.g. 'python -m celeri.version' instead "
            "of 'python celeri/version.py'."
        )
    __version__ = version(__package__)
    return __version__


_cleanup_stale_egg_info()
__version__ = _get_importlib_metadata_version()
if os.environ.get("CELERI_HATCH_VCS_RUNTIME_VERSION"):
    __version__ = _get_hatch_version()
