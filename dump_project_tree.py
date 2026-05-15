#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dump project directory structure:
- Always show ALL directories
- Only show .py and .sh files
- Hide other file types (csv/json/jsonl/etc.)
"""

import os
import argparse
from typing import Set

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".tox",
}


def should_exclude_dir(name: str, exclude_dirs: Set[str], include_hidden: bool) -> bool:
    if not include_hidden and name.startswith("."):
        return True
    return name in exclude_dirs


def dump_tree(
    root: str,
    exts: Set[str],
    exclude_dirs: Set[str],
    include_hidden: bool,
) -> str:
    lines = []
    root = os.path.abspath(root)
    root_name = os.path.basename(root) or root
    lines.append(root_name)

    def walk(current_path: str, prefix: str):
        try:
            entries = sorted(os.listdir(current_path))
        except PermissionError:
            return

        # split dirs / files
        dirs = []
        files = []

        for e in entries:
            full = os.path.join(current_path, e)
            if os.path.isdir(full):
                if not should_exclude_dir(e, exclude_dirs, include_hidden):
                    dirs.append(e)
            else:
                if not include_hidden and e.startswith("."):
                    continue
                if os.path.splitext(e)[1].lower() in exts:
                    files.append(e)

        items = [(d, "dir") for d in dirs] + [(f, "file") for f in files]

        for idx, (name, kind) in enumerate(items):
            is_last = idx == len(items) - 1
            branch = "└── " if is_last else "├── "
            lines.append(prefix + branch + name)

            if kind == "dir":
                next_prefix = prefix + ("    " if is_last else "│   ")
                walk(os.path.join(current_path, name), next_prefix)

    walk(root, "")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Export project tree (directories + .py/.sh only)."
    )
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--out", default="project_tree.txt", help="Output text file")
    parser.add_argument(
        "--ext",
        action="append",
        default=[".py", ".sh"],
        help="File extensions to include (default: .py, .sh)",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and directories",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Extra directory names to exclude",
    )

    args = parser.parse_args()

    exts = {e if e.startswith(".") else "." + e for e in args.ext}
    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS) | set(args.exclude_dir)

    tree_text = dump_tree(
        root=args.root,
        exts=exts,
        exclude_dirs=exclude_dirs,
        include_hidden=args.include_hidden,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(tree_text)

    print(f"[OK] Project structure written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
