#!/usr/bin/env python3

"""This is a dirty hack to install celeri commands globally.

I wanted to do this instead via

`pixi global install --path /path/to/clone/of/celeri`

but this feature is very new and currently very poorly supported.

But it's easy enough to hack something by hand!
"""

import argparse
import sys
import tomllib
from pathlib import Path

SHIM_CONTENT = """
#!/bin/sh

# This file is a dirty hack created by pixi_global_install_shim_hack.py

exec "{pixi_bin_path}" run --manifest-path="{manifest_path}" -- {command_name} "$@"
""".strip()


def get_project_root() -> Path:
    result = Path(__file__).parent.parent.parent.resolve()
    if not (result / ".git").is_dir():
        raise FileNotFoundError(f"Expected .git directory at {result / '.git'}")
    return result


def get_pyproject_toml_path() -> Path:
    result = get_project_root() / "pyproject.toml"
    if not result.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {result}")
    return result


def get_pixi_toml_path() -> Path:
    result = get_project_root() / "pixi.toml"
    if not result.exists():
        raise FileNotFoundError(f"pixi.toml not found at {result}")
    return result


def get_scripts_from_pyproject_toml() -> list[str]:
    pyproject_toml_path = get_pyproject_toml_path()
    with pyproject_toml_path.open("rb") as f:
        pyproject_toml = tomllib.load(f)
    try:
        scripts_table = pyproject_toml["project"]["scripts"]
    except KeyError as exc:
        raise KeyError(
            f"No [project.scripts] table found in {pyproject_toml_path}"
        ) from exc
    if not scripts_table:
        raise ValueError(
            f"No scripts defined under [project.scripts] in {pyproject_toml_path}"
        )
    return list(scripts_table.keys())


def get_pixi_bin_path() -> Path:
    if is_windows():
        raise NotImplementedError("Windows is not supported. Consider Linux.")
    result = Path.home() / ".pixi" / "bin" / "pixi"
    if not result.is_file():
        raise FileNotFoundError(f"Expected pixi executable at {result}")
    return result


def is_windows() -> bool:
    return sys.platform == "win32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create or remove global shims for celeri console scripts under ~/.pixi/bin"
        )
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove previously created shims instead of creating them",
    )
    return parser.parse_args()


def get_shim_content(
    pixi_bin_path: Path, manifest_path: Path, command_name: str
) -> str:
    return (
        SHIM_CONTENT.format(
            pixi_bin_path=pixi_bin_path,
            manifest_path=manifest_path,
            command_name=command_name,
        )
        + "\n"
    )


def remove_shim_if_owned(dest_path: Path) -> bool:
    """Remove shim if it contains our marker. Return True if removed."""
    if not dest_path.exists():
        return False
    try:
        text = dest_path.read_text(errors="ignore")
    except Exception:
        return False
    if "pixi_global_install_shim_hack.py" not in text:
        return False
    dest_path.unlink()
    print(f"Removed shim: {dest_path}")
    return True


def main() -> None:
    args = parse_args()
    scripts = get_scripts_from_pyproject_toml()
    pixi_bin_path = get_pixi_bin_path()
    manifest_path = get_pixi_toml_path()
    bin_dir = pixi_bin_path.parent
    if args.remove:
        removed_any = False
        for command_name in scripts:
            dest_path = bin_dir / command_name
            removed_any = remove_shim_if_owned(dest_path) or removed_any
        if not removed_any:
            print("No shims removed (none found or not owned by this script)")
        return
    for command_name in scripts:
        shim_content = get_shim_content(pixi_bin_path, manifest_path, command_name)
        dest_path = bin_dir / command_name
        dest_path.write_text(shim_content)
        dest_path.chmod(0o755)
        print(f"Wrote shim: {dest_path}")


if __name__ == "__main__":
    main()
