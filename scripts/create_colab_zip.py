#!/usr/bin/env python3
"""Create project.zip for Google Colab training."""

import os
import zipfile
from pathlib import Path


def create_colab_zip():
    """Create a zip file containing only what's needed for Colab training."""

    # Files and directories to include
    includes = [
        "pyproject.toml",
        "requirements.txt",
        "styleforge/",
        "styles/",
        "configs/",
        "StyleForge.ipynb",
    ]

    # Patterns to exclude
    excludes = [
        "__pycache__",
        ".pyc",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    ]

    zip_name = "project.zip"

    print(f"Creating {zip_name} for Google Colab...")
    print()

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        file_count = 0
        total_size = 0

        for item in includes:
            path = Path(item)

            if not path.exists():
                print(f"⚠ Warning: {item} not found, skipping")
                continue

            if path.is_file():
                # Single file
                size = path.stat().st_size
                zipf.write(path, path)
                print(f"  ✓ {path} ({size:,} bytes)")
                file_count += 1
                total_size += size

            elif path.is_dir():
                # Directory - walk and add all files
                for root, dirs, files in os.walk(path):
                    # Filter out excluded directories
                    dirs[:] = [d for d in dirs if d not in excludes]

                    for file in files:
                        # Skip excluded file patterns
                        if any(excl in file for excl in excludes):
                            continue

                        file_path = Path(root) / file
                        arcname = file_path  # Keep relative path

                        size = file_path.stat().st_size
                        zipf.write(file_path, arcname)
                        print(f"  ✓ {arcname} ({size:,} bytes)")
                        file_count += 1
                        total_size += size

        print()
        print(f"✓ Created {zip_name}")
        print(f"  Files: {file_count}")
        print(f"  Uncompressed: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")

        # Get actual zip size
        zip_size = Path(zip_name).stat().st_size
        print(f"  Compressed: {zip_size:,} bytes ({zip_size / 1024 / 1024:.2f} MB)")
        print()
        print("Upload this file to Google Colab and run the notebook.")


if __name__ == "__main__":
    create_colab_zip()
