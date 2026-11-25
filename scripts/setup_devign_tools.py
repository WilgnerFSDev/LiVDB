#!/usr/bin/env python3
import os
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

TEMURIN_JDK11_URL = (
    "https://github.com/adoptium/temurin11-binaries/releases/download/"
    "jdk-11.0.22+7/OpenJDK11U-jdk_x64_linux_hotspot_11.0.22_7.tar.gz"
)

JOERN_1_1_260_URL = (
    "https://github.com/joernio/joern/releases/download/v1.1.260/joern-cli.zip"
)

TOOLS_DIR = _ROOT / "tools"


def _download_with_progress(url: str, dest_path: Path, desc: str) -> None:
    print(f"[*] Downloading {desc}...")
    print(f"    Source: {url}")
    print(f"    Target: {dest_path}")

    def report(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            pct = min(100.0, (downloaded / total_size) * 100)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r    Progress: {pct:5.1f}% ({mb_down:6.1f} / {mb_total:6.1f} MB)")
            sys.stdout.flush()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest_path, reporthook=report)
    print("\n    [✓] Download finished.")


def setup_jdk11() -> Path:
    candidate_paths = [
        TOOLS_DIR / "jdk-11",
        _ROOT / "plugins" / "plugin_cfvd" / "jdk-11",
    ]
    for p in candidate_paths:
        if (p / "bin" / "java").exists():
            print(f"[✓] JDK 11 found at: {p}")
            return p

    target_dir = TOOLS_DIR / "jdk-11"
    tar_path = TOOLS_DIR / "jdk11.tar.gz"

    _download_with_progress(TEMURIN_JDK11_URL, tar_path, "Eclipse Temurin OpenJDK 11")

    print(f"[*] Extracting JDK 11 to {target_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        top_dir = tar.getmembers()[0].name.split("/")[0]
        tar.extractall(path=TOOLS_DIR)

    extracted_dir = TOOLS_DIR / top_dir
    if extracted_dir.exists() and extracted_dir != target_dir:
        extracted_dir.rename(target_dir)

    tar_path.unlink(missing_ok=True)
    print(f"[✓] JDK 11 installed at: {target_dir}")
    return target_dir


def setup_joern() -> Path:
    candidate_paths = [
        TOOLS_DIR / "joern-cli",
        _ROOT / "plugins" / "plugin_cfvd" / "joern-cli",
    ]
    for p in candidate_paths:
        if (p / "joern").exists():
            print(f"[✓] Joern 1.1.260 found at: {p}")
            return p

    target_dir = TOOLS_DIR / "joern-cli"
    zip_path = TOOLS_DIR / "joern-cli.zip"

    _download_with_progress(JOERN_1_1_260_URL, zip_path, "Joern v1.1.260")

    print(f"[*] Extracting Joern to {TOOLS_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=TOOLS_DIR)

    zip_path.unlink(missing_ok=True)

    for item in (target_dir / "bin").glob("*"):
        os.chmod(item, 0o755)
    for name in ["joern", "joern-parse", "joern-export", "joern-scan"]:
        fpath = target_dir / name
        if fpath.exists():
            os.chmod(fpath, 0o755)

    print(f"[✓] Joern installed at: {target_dir}")
    return target_dir


def main() -> None:
    print("═" * 70)
    print("  LiVDB - Environment Setup for Canonical Devign & Literature Baselines")
    print("═" * 70)
    jdk_path = setup_jdk11()
    joern_path = setup_joern()
    print("═" * 70)
    print("[✓] All tools configured for 100% reproducible execution:")
    print(f"    JDK 11    : {jdk_path}")
    print(f"    Joern CLI : {joern_path}")
    print("═" * 70)


if __name__ == "__main__":
    main()
