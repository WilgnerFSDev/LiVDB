import os
import shutil
import tempfile
import pandas as pd
import pytest

from scripts.dataset_builder import process_cve_dataset, _process_single_cve


@pytest.fixture
def sample_cve_env():
    tmp_dir = tempfile.mkdtemp()
    cve_id = "CVE-TEST-0001"
    vuln_dir = os.path.join(tmp_dir, cve_id, "vulnerable")
    patched_dir = os.path.join(tmp_dir, cve_id, "patched")
    os.makedirs(vuln_dir, exist_ok=True)
    os.makedirs(patched_dir, exist_ok=True)

    vuln_code = """
int func_changed(int a) {
    // vulnerable buffer overflow
    char buf[10];
    return a + 1;
}

int func_unchanged(int b) {
    return b * 2;
}

int func_deleted(int c) {
    return c - 5;
}
"""

    patched_code = """
int func_changed(int a) {
    // fixed buffer handling
    char buf[20];
    if (a < 0) return 0;
    return a + 1;
}

int func_unchanged(int b) {
    return b * 2;
}
"""

    with open(os.path.join(vuln_dir, "test_file.c"), "w", encoding="utf-8") as f:
        f.write(vuln_code)

    with open(os.path.join(patched_dir, "test_file.c"), "w", encoding="utf-8") as f:
        f.write(patched_code)

    yield tmp_dir, cve_id

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_process_single_cve_full_context(sample_cve_env):
    root_dir, cve_id = sample_cve_env
    res = _process_single_cve((cve_id, root_dir, False))

    assert len(res["rows"]) == 3
    labels = [r["label"] for r in res["rows"]]
    assert labels.count(1) == 1
    assert labels.count(0) == 2
    assert res["functions_removed"] == 1


def test_process_single_cve_paired_only(sample_cve_env):
    root_dir, cve_id = sample_cve_env
    res = _process_single_cve((cve_id, root_dir, True))

    assert len(res["rows"]) == 2
    labels = [r["label"] for r in res["rows"]]
    assert labels.count(1) == 1
    assert labels.count(0) == 1
    assert res["functions_changed"] == 1


def test_process_cve_dataset_paired_pipeline(sample_cve_env, monkeypatch):
    from scripts.ui import UI

    monkeypatch.setattr(UI, "ask_boolean", lambda *args, **kwargs: False)

    root_dir, cve_id = sample_cve_env
    output_csv = os.path.join(root_dir, "test_paired.csv")

    process_cve_dataset(root_dir, output_csv, paired_only=True)

    assert os.path.exists(output_csv)
    df = pd.read_csv(output_csv)

    assert len(df) == 2
    assert (df["label"] == 1).sum() == (df["label"] == 0).sum()
    assert set(df.columns) == {"cve_id", "filename", "hash", "lines", "label", "function"}


def test_strip_c_comments_preserves_strings():
    from scripts.utils import strip_c_comments
    from scripts.dataset_builder import calculate_hash

    c_code_clean = """int foo(void) {
        return 0;
    }"""

    c_code_with_comments = """int foo(void) {
        /* This is a block comment */
        // This is a line comment
        return 0;
    }"""

    c_code_with_string_containing_comment = """void print_test(void) {
        printf("/* not a comment */ // also not");
        char c = '"';
    }"""

    assert calculate_hash(c_code_clean) == calculate_hash(c_code_with_comments)

    stripped_strings = strip_c_comments(c_code_with_string_containing_comment)
    assert '/* not a comment */' in stripped_strings
    assert '// also not' in stripped_strings
