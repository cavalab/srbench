#!/usr/bin/env python3
"""Type-check every Python file, including importlib-loaded hyphenated methods.

Mypy's CLI rejects package names such as gp-elite and ps-tree. Supply explicit,
unique module names so these adapters are checked alongside the rest of the
repository without renaming the methods or excluding their source code.
"""

import subprocess
from pathlib import Path

from mypy.build import BuildSource, build
from mypy.errors import CompileError
from mypy.main import process_options


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    # Include tracked and new source files anywhere in the repo, honoring gitignore.
    listing = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "*.py",
        ],
        cwd=root,
    )
    paths = sorted({root / name.decode() for name in listing.split(b"\0") if name})
    paths = [path for path in paths if path.is_file()]

    sources = []
    for path in paths:
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        module = ".".join(part.replace("-", "_") for part in parts) or "srbench"
        sources.append(BuildSource(str(path), module, None))

    _, options = process_options(
        ["--config-file", str(root / "pyproject.toml")], require_targets=False
    )
    try:
        result = build(sources=sources, options=options)
        messages = result.errors
    except CompileError as exc:
        messages = exc.messages
    for message in messages:
        print(message)
    if messages:
        return 1
    print(f"Success: no issues found in {len(sources)} source files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
