# Test Fixtures

This directory contains minimal, deterministic CVE fixtures used by the automated test suite (`tests/test_dataset_snapshot.py` and `tests/test_cve_filters.py`) so all tests run out of the box on a fresh clone without requiring NVD API calls or a full Linux kernel repository clone.

## Contents

- `raw_data/` — Vulnerable and patched C/H source file snapshots for 8 historical Linux kernel CVEs (`CVE-2007-6762`, `CVE-2010-5331`, `CVE-2011-0006`, `CVE-2011-1477`, `CVE-2021-46909`, `CVE-2022-49925`, `CVE-2024-26600`, `CVE-2024-39484`).
- `cache/` — Corresponding NIST NVD API v2.0 JSON metadata records for 5 sample CVEs.

## Provenance & Licensing

- Source files under `raw_data/` originate from the **Linux kernel** and remain subject to the **GNU General Public License v2.0 (GPL-2.0)** as indicated in their respective headers.
- Metadata files under `cache/` originate from the **NIST National Vulnerability Database (NVD) API v2.0** (public domain data; *this product uses data from the NVD API but is not endorsed or certified by the NVD*).
