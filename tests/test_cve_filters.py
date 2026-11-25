import json
import os
import shutil
import tempfile
import pandas as pd
import pytest

from scripts.dataset_builder import process_cve_dataset, _process_single_cve
from scripts.utils import (
    extract_cve_base_score,
    extract_cve_cwes,
    normalize_cwe_filter,
    matches_cve_filters,
    strip_c_comments,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
FIXTURES_RAW_DIR = os.path.join(FIXTURES_DIR, "raw_data")
FIXTURES_CACHE_DIR = os.path.join(FIXTURES_DIR, "cache")


REAL_SAMPLE_CVES = [
    "CVE-2024-26600",
    "CVE-2007-6762",
    "CVE-2010-5331",
    "CVE-2011-0006",
    "CVE-2011-1477",
]

KERNEL_MACRO_CVES = [
    "CVE-2022-49925",
    "CVE-2021-46909",
    "CVE-2024-39484",
]


@pytest.fixture(scope="module")
def real_cve_env():
    missing = [
        cve_id
        for cve_id in REAL_SAMPLE_CVES
        if not os.path.exists(os.path.join(FIXTURES_RAW_DIR, cve_id))
        or not os.path.exists(os.path.join(FIXTURES_CACHE_DIR, f"{cve_id}.json"))
    ]
    if missing:
        pytest.fail(f"Missing required CVE test fixtures: {missing}")

    tmp_dir = tempfile.mkdtemp()
    raw_dir = os.path.join(tmp_dir, "raw_data")
    cache_dir = os.path.join(tmp_dir, "cache")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    for cve_id in REAL_SAMPLE_CVES:
        src_raw = os.path.join(FIXTURES_RAW_DIR, cve_id)
        src_cache = os.path.join(FIXTURES_CACHE_DIR, f"{cve_id}.json")
        shutil.copytree(src_raw, os.path.join(raw_dir, cve_id))
        shutil.copyfile(src_cache, os.path.join(cache_dir, f"{cve_id}.json"))

    yield raw_dir, cache_dir, tmp_dir

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_real_cache_metadata_cve_2024_26600(real_cve_env):
    _, cache_dir, _ = real_cve_env
    cache_file = os.path.join(cache_dir, "CVE-2024-26600.json")
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    score = extract_cve_base_score(data)
    cwes = extract_cve_cwes(data)

    assert score == 5.5
    assert cwes == ["CWE-476"]


def test_real_cache_metadata_cve_2007_6762(real_cve_env):
    _, cache_dir, _ = real_cve_env
    cache_file = os.path.join(cache_dir, "CVE-2007-6762.json")
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    score = extract_cve_base_score(data)
    cwes = extract_cve_cwes(data)

    assert score == 9.8
    assert cwes == ["CWE-119"]


def test_real_cache_metadata_cve_2010_5331(real_cve_env):
    _, cache_dir, _ = real_cve_env
    cache_file = os.path.join(cache_dir, "CVE-2010-5331.json")
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    score = extract_cve_base_score(data)
    cwes = extract_cve_cwes(data)

    assert score == 7.8
    assert cwes == ["CWE-193"]


def test_real_cache_metadata_cve_2011_0006(real_cve_env):
    _, cache_dir, _ = real_cve_env
    cache_file = os.path.join(cache_dir, "CVE-2011-0006.json")
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    score = extract_cve_base_score(data)
    cwes = extract_cve_cwes(data)

    assert score == 1.9
    assert cwes == ["CWE-264"]


def test_real_cache_metadata_cve_2011_1477_multiple_cwes(real_cve_env):
    _, cache_dir, _ = real_cve_env
    cache_file = os.path.join(cache_dir, "CVE-2011-1477.json")
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    score = extract_cve_base_score(data)
    cwes = extract_cve_cwes(data)

    assert score == 7.2
    assert "CWE-119" in cwes
    assert "CWE-264" in cwes


def test_normalize_cwe_filter_formats():
    res = normalize_cwe_filter("615, 123, CWE-354")
    assert {"615", "CWE-615", "123", "CWE-123", "354", "CWE-354"}.issubset(res)

    res_list = normalize_cwe_filter(["476", "CWE-120"])
    assert {"476", "CWE-476", "120", "CWE-120"}.issubset(res_list)

    assert normalize_cwe_filter("") == set()
    assert normalize_cwe_filter(None) == set()


def test_matches_cve_filters_logic():
    data = {
        "cve": {
            "id": "CVE-SYNTHETIC-1",
            "metrics": {
                "cvssMetricV31": [
                    {
                        "type": "Primary",
                        "cvssData": {"version": "3.1", "baseScore": 7.5},
                    }
                ]
            },
            "weaknesses": [
                {"description": [{"lang": "en", "value": "CWE-476"}]}
            ],
        }
    }

    assert matches_cve_filters(data, min_score=7.0) is True
    assert matches_cve_filters(data, min_score=8.0) is False
    assert matches_cve_filters(data, max_score=8.0) is True
    assert matches_cve_filters(data, max_score=7.0) is False
    assert matches_cve_filters(data, min_score=7.0, max_score=8.0) is True
    assert matches_cve_filters(data, cwes="476") is True
    assert matches_cve_filters(data, cwes="CWE-476") is True
    assert matches_cve_filters(data, cwes="615, 123, 476") is True
    assert matches_cve_filters(data, cwes="119, 120") is False


def test_real_cve_extraction_paired_mode(real_cve_env):
    raw_dir, _, _ = real_cve_env

    res_26600 = _process_single_cve(("CVE-2024-26600", raw_dir, True))
    assert len(res_26600["rows"]) == 4
    labels_26600 = [r["label"] for r in res_26600["rows"]]
    assert labels_26600.count(1) == 2
    assert labels_26600.count(0) == 2

    res_6762 = _process_single_cve(("CVE-2007-6762", raw_dir, True))
    assert len(res_6762["rows"]) == 2
    labels_6762 = [r["label"] for r in res_6762["rows"]]
    assert labels_6762.count(1) == 1
    assert labels_6762.count(0) == 1


def test_real_cve_extraction_full_context_mode(real_cve_env):
    raw_dir, _, _ = real_cve_env

    res_26600 = _process_single_cve(("CVE-2024-26600", raw_dir, False))
    assert len(res_26600["rows"]) == 19
    labels_26600 = [r["label"] for r in res_26600["rows"]]
    assert labels_26600.count(1) == 2
    assert labels_26600.count(0) == 17

    res_6762 = _process_single_cve(("CVE-2007-6762", raw_dir, False))
    assert len(res_6762["rows"]) == 11
    labels_6762 = [r["label"] for r in res_6762["rows"]]
    assert labels_6762.count(1) == 1
    assert labels_6762.count(0) == 10


def test_real_cve_hash_integrity(real_cve_env):
    import hashlib

    raw_dir, _, _ = real_cve_env
    res = _process_single_cve(("CVE-2024-26600", raw_dir, False))

    for row in res["rows"]:
        code = row["function"]
        clean_code = strip_c_comments(code)
        normalized = "".join(clean_code.split())
        computed_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        assert computed_hash == row["hash"], f"Hash mismatch for function in {row['cve_id']}"


def test_real_dataset_build_filter_min_score(real_cve_env, monkeypatch):
    raw_dir, cache_dir, tmp_dir = real_cve_env
    from scripts import dataset_builder
    monkeypatch.setattr(dataset_builder, "CACHE_DIR", cache_dir)

    out_csv = os.path.join(tmp_dir, "dataset_min_score_8.csv")
    process_cve_dataset(raw_dir, out_csv, min_score=8.0, interactive=False)

    assert os.path.exists(out_csv)
    df = pd.read_csv(out_csv)
    assert set(df["cve_id"].unique()) == {"CVE-2007-6762"}


def test_real_dataset_build_filter_max_score(real_cve_env, monkeypatch):
    raw_dir, cache_dir, tmp_dir = real_cve_env
    from scripts import dataset_builder
    monkeypatch.setattr(dataset_builder, "CACHE_DIR", cache_dir)

    out_csv = os.path.join(tmp_dir, "dataset_max_score_5.csv")
    process_cve_dataset(raw_dir, out_csv, max_score=5.0, interactive=False)

    assert os.path.exists(out_csv)
    df = pd.read_csv(out_csv)
    assert set(df["cve_id"].unique()) == {"CVE-2011-0006"}


def test_real_dataset_build_filter_score_range(real_cve_env, monkeypatch):
    raw_dir, cache_dir, tmp_dir = real_cve_env
    from scripts import dataset_builder
    monkeypatch.setattr(dataset_builder, "CACHE_DIR", cache_dir)

    out_csv = os.path.join(tmp_dir, "dataset_score_range_7_to_8.csv")
    process_cve_dataset(raw_dir, out_csv, min_score=7.0, max_score=8.0, interactive=False)

    assert os.path.exists(out_csv)
    df = pd.read_csv(out_csv)
    assert set(df["cve_id"].unique()) == {"CVE-2010-5331", "CVE-2011-1477"}


def test_real_dataset_build_filter_cwe_numeric(real_cve_env, monkeypatch):
    raw_dir, cache_dir, tmp_dir = real_cve_env
    from scripts import dataset_builder
    monkeypatch.setattr(dataset_builder, "CACHE_DIR", cache_dir)

    out_csv = os.path.join(tmp_dir, "dataset_cwe_476.csv")
    process_cve_dataset(raw_dir, out_csv, cwes="476", interactive=False)

    assert os.path.exists(out_csv)
    df = pd.read_csv(out_csv)
    assert set(df["cve_id"].unique()) == {"CVE-2024-26600"}


def test_real_dataset_build_filter_cwe_prefixed(real_cve_env, monkeypatch):
    raw_dir, cache_dir, tmp_dir = real_cve_env
    from scripts import dataset_builder
    monkeypatch.setattr(dataset_builder, "CACHE_DIR", cache_dir)

    out_csv = os.path.join(tmp_dir, "dataset_cwe_119.csv")
    process_cve_dataset(raw_dir, out_csv, cwes="CWE-119", interactive=False)

    assert os.path.exists(out_csv)
    df = pd.read_csv(out_csv)
    assert set(df["cve_id"].unique()) == {"CVE-2007-6762", "CVE-2011-1477"}


def test_real_dataset_build_filter_multiple_cwes(real_cve_env, monkeypatch):
    raw_dir, cache_dir, tmp_dir = real_cve_env
    from scripts import dataset_builder
    monkeypatch.setattr(dataset_builder, "CACHE_DIR", cache_dir)

    out_csv = os.path.join(tmp_dir, "dataset_cwe_multiple.csv")
    process_cve_dataset(raw_dir, out_csv, cwes="264, 193", interactive=False)

    assert os.path.exists(out_csv)
    df = pd.read_csv(out_csv)
    assert set(df["cve_id"].unique()) == {"CVE-2011-0006", "CVE-2011-1477", "CVE-2010-5331"}


def test_real_dataset_build_filter_combined_score_and_cwe(real_cve_env, monkeypatch):
    raw_dir, cache_dir, tmp_dir = real_cve_env
    from scripts import dataset_builder
    monkeypatch.setattr(dataset_builder, "CACHE_DIR", cache_dir)

    out_csv = os.path.join(tmp_dir, "dataset_combined_min7_cwe119.csv")
    process_cve_dataset(raw_dir, out_csv, min_score=7.0, cwes="119", interactive=False)

    assert os.path.exists(out_csv)
    df = pd.read_csv(out_csv)
    assert set(df["cve_id"].unique()) == {"CVE-2007-6762", "CVE-2011-1477"}


def test_real_cve_single_function_only_worker_flag(real_cve_env):
    raw_dir, _, _ = real_cve_env

    res_26600 = _process_single_cve(("CVE-2024-26600", raw_dir, True, True))
    assert res_26600["is_tangled"] is True
    assert len(res_26600["rows"]) == 0
    assert res_26600["functions_changed"] == 2

    res_6762 = _process_single_cve(("CVE-2007-6762", raw_dir, True, True))
    assert res_6762["is_tangled"] is False
    assert len(res_6762["rows"]) == 2
    assert res_6762["functions_changed"] == 1


def test_real_dataset_build_filter_single_function_only(real_cve_env, monkeypatch):
    raw_dir, cache_dir, tmp_dir = real_cve_env
    from scripts import dataset_builder
    monkeypatch.setattr(dataset_builder, "CACHE_DIR", cache_dir)

    out_csv = os.path.join(tmp_dir, "dataset_single_function_only.csv")
    process_cve_dataset(raw_dir, out_csv, single_function_only=True, interactive=False)

    assert os.path.exists(out_csv)
    df = pd.read_csv(out_csv)
    included = set(df["cve_id"].unique())

    assert included == {"CVE-2007-6762", "CVE-2010-5331", "CVE-2011-0006"}
    assert "CVE-2024-26600" not in included
    assert "CVE-2011-1477" not in included


def test_real_dataset_build_filter_single_function_combined_with_cwe(real_cve_env, monkeypatch):
    raw_dir, cache_dir, tmp_dir = real_cve_env
    from scripts import dataset_builder
    monkeypatch.setattr(dataset_builder, "CACHE_DIR", cache_dir)

    out_csv = os.path.join(tmp_dir, "dataset_cwe119_single_func.csv")
    process_cve_dataset(raw_dir, out_csv, cwes="119", single_function_only=True, interactive=False)

    assert os.path.exists(out_csv)
    df = pd.read_csv(out_csv)
    assert set(df["cve_id"].unique()) == {"CVE-2007-6762"}


def test_calculate_ast_hash_immunity_to_comments_and_whitespace():
    from scripts.c_parser import calculate_ast_hash
    from scripts.dataset_builder import calculate_text_hash

    code_with_comments = """
    int calculate_total(int count, int price) {
        // Line comment explaining calculation
        /* Multi-line
           block comment */
        int tax = 5;
        return (count * price) + tax;
    }
    """

    code_without_comments = """
    int calculate_total(int count, int price) {
        int tax = 5;
        return (count * price) + tax;
    }
    """

    code_modified = """
    int calculate_total(int count, int price) {
        int tax = 10;
        return (count * price) + tax;
    }
    """

    assert calculate_ast_hash(code_with_comments) == calculate_ast_hash(code_without_comments)
    assert calculate_text_hash(code_with_comments) == calculate_text_hash(code_without_comments)

    assert calculate_ast_hash(code_with_comments) != calculate_ast_hash(code_modified)
    assert calculate_text_hash(code_with_comments) != calculate_text_hash(code_modified)


def test_ast_hash_with_different_inline_and_block_comments():
    from scripts.c_parser import calculate_ast_hash
    from scripts.dataset_builder import calculate_text_hash

    code_vulnerable_comments = """
    int process_buffer(int *buffer, int length) {
        // [OLD] Check if pointer is valid before proceeding
        if (!buffer || length <= 0) {
            return -1; /* error code */
        }
        int sum = 0; // accumulator variable
        /*
         * Iterate through buffer elements
         * and compute summation
         */
        for (int i = 0; i < length; i++) {
            sum += buffer[i]; // add current element
        }
        // Return computed sum to caller
        return sum;
    }
    """

    code_patched_comments = """
    int process_buffer(int *buffer, int length) {
        /* [NEW] Validate input parameters */
        if (!buffer || length <= 0) {
            return -1;
        }
        int sum = 0;
        // Replaced loop comment with single-line note
        for (int i = 0; i < length; i++) {
            sum += buffer[i];
        }
        /* All done */
        return sum;
    }
    """

    assert calculate_ast_hash(code_vulnerable_comments) == calculate_ast_hash(code_patched_comments)
    assert calculate_text_hash(code_vulnerable_comments) == calculate_text_hash(code_patched_comments)


def test_ast_hash_preserves_strings_containing_comment_delimiters():
    from scripts.c_parser import calculate_ast_hash

    code_orig = """
    void log_event(void) {
        printf("DEBUG: /* this is a string literal, not a comment // */");
    }
    """

    code_with_real_comment = """
    void log_event(void) {
        // Real comment above
        printf("DEBUG: /* this is a string literal, not a comment // */"); /* Real block comment */
    }
    """

    code_with_modified_string = """
    void log_event(void) {
        printf("DEBUG: /* modified string content // */");
    }
    """

    assert calculate_ast_hash(code_orig) == calculate_ast_hash(code_with_real_comment)
    assert calculate_ast_hash(code_orig) != calculate_ast_hash(code_with_modified_string)


def test_dataset_builder_ignores_comment_only_diffs_in_cve():
    tmp_dir = tempfile.mkdtemp()
    cve_id = "CVE-SYNTHETIC-COMMENT-DIFF"
    v_dir = os.path.join(tmp_dir, cve_id, "vulnerable")
    p_dir = os.path.join(tmp_dir, cve_id, "patched")
    os.makedirs(v_dir, exist_ok=True)
    os.makedirs(p_dir, exist_ok=True)

    v_code = """
    int helper_func(int x) {
        // Initial comment before calculation
        return x * 2;
    }
    """

    p_code = """
    int helper_func(int x) {
        /* Updated multi-line comment
           explaining x * 2 */
        return x * 2;
    }
    """

    with open(os.path.join(v_dir, "helper.c"), "w") as f:
        f.write(v_code)
    with open(os.path.join(p_dir, "helper.c"), "w") as f:
        f.write(p_code)

    try:
        res_ast = _process_single_cve((cve_id, tmp_dir, True, False, None, "ast"))
        assert res_ast["functions_changed"] == 0
        assert res_ast["functions_unchanged"] == 1
        assert len(res_ast["rows"]) == 0

        res_text = _process_single_cve((cve_id, tmp_dir, True, False, None, "text"))
        assert res_text["functions_changed"] == 0
        assert res_text["functions_unchanged"] == 1
        assert len(res_text["rows"]) == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_text_hash_vs_ast_hash_equivalence_on_real_cves(real_cve_env):
    raw_dir, _, _ = real_cve_env

    for cve_id in REAL_SAMPLE_CVES:
        res_text = _process_single_cve((cve_id, raw_dir, False, False, None, "text"))
        res_ast = _process_single_cve((cve_id, raw_dir, False, False, None, "ast"))

        assert res_text["functions_found"] == res_ast["functions_found"]
        assert res_text["functions_changed"] == res_ast["functions_changed"]
        assert res_text["functions_unchanged"] == res_ast["functions_unchanged"]
        assert len(res_text["rows"]) == len(res_ast["rows"])

        text_labels = [(r["filename"], r["function"][:40], r["label"]) for r in res_text["rows"]]
        ast_labels = [(r["filename"], r["function"][:40], r["label"]) for r in res_ast["rows"]]
        assert text_labels == ast_labels, f"Label discrepancy found in {cve_id}"


def test_real_dataset_build_with_ast_hash_strategy(real_cve_env, monkeypatch):
    raw_dir, cache_dir, tmp_dir = real_cve_env
    from scripts import dataset_builder
    monkeypatch.setattr(dataset_builder, "CACHE_DIR", cache_dir)

    out_csv = os.path.join(tmp_dir, "dataset_ast_hash.csv")
    process_cve_dataset(raw_dir, out_csv, hash_strategy="ast", interactive=False)

    assert os.path.exists(out_csv)
    df = pd.read_csv(out_csv)
    assert set(df["cve_id"].unique()) == set(REAL_SAMPLE_CVES)
    assert all(df["hash"].str.len() == 64)


@pytest.fixture(scope="module")
def kernel_macro_cve_env():
    missing = [
        cve_id
        for cve_id in KERNEL_MACRO_CVES
        if not os.path.exists(os.path.join(FIXTURES_RAW_DIR, cve_id))
    ]
    if missing:
        pytest.fail(f"Missing required kernel macro CVE fixtures: {missing}")

    tmp_dir = tempfile.mkdtemp()
    raw_dir = os.path.join(tmp_dir, "raw_data")
    cache_dir = os.path.join(tmp_dir, "cache")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    for cve_id in KERNEL_MACRO_CVES:
        src_raw = os.path.join(FIXTURES_RAW_DIR, cve_id)
        shutil.copytree(src_raw, os.path.join(raw_dir, cve_id))
        src_cache = os.path.join(FIXTURES_CACHE_DIR, f"{cve_id}.json")
        if os.path.exists(src_cache):
            shutil.copyfile(src_cache, os.path.join(cache_dir, f"{cve_id}.json"))

    yield raw_dir, cache_dir, tmp_dir

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_text_hash_vs_ast_hash_divergence_on_kernel_macro_cves(kernel_macro_cve_env):
    raw_dir, _, _ = kernel_macro_cve_env

    for cve_id in KERNEL_MACRO_CVES:
        res_text = _process_single_cve((cve_id, raw_dir, True, False, None, "text"))
        res_ast = _process_single_cve((cve_id, raw_dir, True, False, None, "ast"))

        assert res_text["functions_changed"] == res_ast["functions_changed"], (
            f"Divergence detected in {cve_id}: "
            f"Text Hash identified {res_text['functions_changed']} changed functions, "
            f"while AST Hash identified {res_ast['functions_changed']}."
        )
