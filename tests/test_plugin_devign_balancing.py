import os
import shutil
import tempfile
import pandas as pd
import pytest
from unittest.mock import patch

from plugins.plugin_devign.main_plugin import DevignPlugin


@pytest.fixture
def imbalanced_csv():
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, "dataset_imbalanced.csv")

    data = {
        "cve_id": [f"CVE-V{i}" for i in range(10)] + [f"CVE-C{i}" for i in range(100)],
        "filename": [f"v{i}.c" for i in range(10)] + [f"c{i}.c" for i in range(100)],
        "label": [1] * 10 + [0] * 100,
        "function": [f"void vuln_{i}() {{}}" for i in range(10)]
        + [f"void clean_{i}() {{}}" for i in range(100)],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    yield csv_path, tmp_dir

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_devign_balancing_1to1(imbalanced_csv):
    csv_path, tmp_dir = imbalanced_csv
    plugin = DevignPlugin()
    plugin.workspace_dir = os.path.join(tmp_dir, "workspace")

    raw_code_dir = os.path.join(tmp_dir, "raw_code")
    os.makedirs(raw_code_dir, exist_ok=True)

    with patch("scripts.ui.UI.ask_choice", return_value="1"), \
         patch("scripts.ui.UI.ask_string", return_value="42"), \
         patch("plugins.plugin_devign.main_plugin.RAW_CODE_DIR", raw_code_dir), \
         patch("plugins.plugin_devign.main_plugin.export_to_raw") as mock_export, \
         patch("plugins.plugin_devign.main_plugin.generate_cpg", return_value=False):
        plugin.run(csv_path)

    expected_balanced_file = os.path.join(tmp_dir, "dataset_imbalanced_balanced1to1.csv")
    assert os.path.exists(expected_balanced_file)

    df_balanced = pd.read_csv(expected_balanced_file)
    assert len(df_balanced) == 20
    assert (df_balanced["label"] == 1).sum() == 10
    assert (df_balanced["label"] == 0).sum() == 10

    mock_export.assert_called_once()
    assert mock_export.call_args[1]["dataset_path"] == expected_balanced_file


def test_devign_balancing_skipped_when_already_balanced(tmp_path):
    csv_path = str(tmp_path / "already_balanced.csv")
    data = {
        "cve_id": ["CVE-1", "CVE-2"],
        "filename": ["f1.c", "f2.c"],
        "label": [1, 0],
        "function": ["void f1() {}", "void f2() {}"],
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)

    plugin = DevignPlugin()
    plugin.workspace_dir = str(tmp_path / "workspace")

    raw_code_dir = str(tmp_path / "raw_code")
    os.makedirs(raw_code_dir, exist_ok=True)

    with patch("scripts.ui.UI.ask_choice") as mock_choice, \
         patch("scripts.ui.UI.ask_string", return_value="42"), \
         patch("plugins.plugin_devign.main_plugin.RAW_CODE_DIR", raw_code_dir), \
         patch("plugins.plugin_devign.main_plugin.export_to_raw") as mock_export, \
         patch("plugins.plugin_devign.main_plugin.generate_cpg", return_value=False):
        plugin.run(csv_path)

    mock_choice.assert_not_called()
    mock_export.assert_called_once_with(
        output_dir=mock_export.call_args[1]["output_dir"],
        dataset_path=csv_path
    )
