import os
import tempfile
import pandas as pd
import pytest
from unittest.mock import patch

from scripts.curator import curate_dataset


@pytest.fixture
def dummy_csv():
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, "dataset_test.csv")

    data = {
        "cve_id": ["CVE-1", "CVE-2", "CVE-3", "CVE-4", "CVE-5"],
        "filename": ["f1.c", "f2.c", "f3.c", "f4.c", "f5.c"],
        "label": [0, 1, 0, 1, 0],
        "function": [
            "void func_dup() { int a = 1; }",
            "void func_unique1() { int b = 2; }",
            "void func_unique2() { int c = 3; }",
            "void func_dup() { int a = 1; }",
            "void func_unique3() { int d = 4; }",
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    yield csv_path, tmp_dir

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_curate_preserves_original_order_without_shuffle(dummy_csv):
    csv_path, _ = dummy_csv

    with patch("scripts.ui.UI.ask_choice", return_value="0"), \
         patch("scripts.ui.UI.ask_boolean", return_value=False):
        out_path = curate_dataset(csv_path)

    assert out_path is not None
    assert os.path.exists(out_path)

    df_out = pd.read_csv(out_path)
    assert len(df_out) == 4

    dup_row = df_out[df_out["cve_id"] == "CVE-4"]
    assert len(dup_row) == 1
    assert dup_row.iloc[0]["label"] == 1

    expected_cves = ["CVE-2", "CVE-3", "CVE-4", "CVE-5"]
    assert list(df_out["cve_id"]) == expected_cves


def test_curate_with_shuffle_seed(dummy_csv):
    csv_path, _ = dummy_csv

    with patch("scripts.ui.UI.ask_choice", return_value="0"), \
         patch("scripts.ui.UI.ask_boolean", return_value=True), \
         patch("scripts.ui.UI.ask_string", return_value="42"):
        out_path = curate_dataset(csv_path)

    assert out_path is not None
    df_out = pd.read_csv(out_path)
    assert len(df_out) == 4
    assert len(df_out[df_out["label"] == 1]) == 2
    assert len(df_out[df_out["label"] == 0]) == 2


def test_curate_with_shuffle_blank_seed(dummy_csv):
    csv_path, _ = dummy_csv

    with patch("scripts.ui.UI.ask_choice", return_value="0"), \
         patch("scripts.ui.UI.ask_boolean", return_value=True), \
         patch("scripts.ui.UI.ask_string", return_value="   "):
        out_path = curate_dataset(csv_path)

    assert out_path is not None
    df_out = pd.read_csv(out_path)
    assert len(df_out) == 4
    assert len(df_out[df_out["label"] == 1]) == 2
    assert len(df_out[df_out["label"] == 0]) == 2
