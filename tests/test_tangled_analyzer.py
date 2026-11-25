import os
import tempfile
import pandas as pd
import pytest

from scripts.tangled_analyzer import (
    compute_tangled_distribution,
    print_terminal_histogram,
    generate_tangled_histogram_plot,
    run_tangled_analysis,
)


@pytest.fixture
def sample_dataset_csv():
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, "synthetic_dataset.csv")

    rows = []
    rows.append({"cve_id": "CVE-2021-0001", "filename": "a.c", "hash": "h1", "label": 1, "function": "void f1(){}"})
    rows.append({"cve_id": "CVE-2021-0001", "filename": "a.c", "hash": "h2", "label": 0, "function": "void f1_p(){}"})

    rows.append({"cve_id": "CVE-2021-0002", "filename": "b.c", "hash": "h3", "label": 1, "function": "void f2(){}"})
    rows.append({"cve_id": "CVE-2021-0002", "filename": "b.c", "hash": "h4", "label": 0, "function": "void f2_p(){}"})

    rows.append({"cve_id": "CVE-2021-0003", "filename": "c.c", "hash": "h5", "label": 1, "function": "void f3a(){}"})
    rows.append({"cve_id": "CVE-2021-0003", "filename": "c.c", "hash": "h6", "label": 1, "function": "void f3b(){}"})
    rows.append({"cve_id": "CVE-2021-0003", "filename": "c.c", "hash": "h7", "label": 0, "function": "void f3a_p(){}"})

    rows.append({"cve_id": "CVE-2021-0004", "filename": "d.c", "hash": "h8", "label": 1, "function": "void f4a(){}"})
    rows.append({"cve_id": "CVE-2021-0004", "filename": "d.c", "hash": "h9", "label": 1, "function": "void f4b(){}"})
    rows.append({"cve_id": "CVE-2021-0004", "filename": "d.c", "hash": "h10", "label": 1, "function": "void f4c(){}"})

    rows.append({"cve_id": "CVE-2021-0001", "filename": "a.c", "hash": "h11", "label": 0, "function": "void f_clean(){}"})

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    yield csv_path, tmp_dir

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_compute_tangled_distribution_metrics(sample_dataset_csv):
    csv_path, _ = sample_dataset_csv
    data = compute_tangled_distribution(csv_path)

    assert data["total_cves"] == 4
    assert data["total_functions_changed"] == 7

    assert data["single_func_count"] == 2
    assert data["single_func_pct"] == 50.0

    assert data["multi_func_count"] == 2
    assert data["multi_func_pct"] == 50.0

    assert data["counts_by_num"][1] == 2
    assert data["counts_by_num"][2] == 1
    assert data["counts_by_num"][3] == 1
    assert data["percentages"][1] == 50.0
    assert data["percentages"][2] == 25.0
    assert data["percentages"][3] == 25.0

    assert data["mean_funcs"] == 7 / 4
    assert data["max_funcs"] == 3


def test_compute_tangled_distribution_empty_and_error_cases():
    tmp_dir = tempfile.mkdtemp()

    with pytest.raises(FileNotFoundError):
        compute_tangled_distribution(os.path.join(tmp_dir, "non_existent.csv"))

    invalid_csv = os.path.join(tmp_dir, "invalid.csv")
    pd.DataFrame([{"foo": "bar"}]).to_csv(invalid_csv, index=False)
    with pytest.raises(ValueError):
        compute_tangled_distribution(invalid_csv)

    no_vuln_csv = os.path.join(tmp_dir, "no_vuln.csv")
    pd.DataFrame([{"cve_id": "CVE-1", "label": 0, "function": "void f(){}"}]).to_csv(no_vuln_csv, index=False)
    data = compute_tangled_distribution(no_vuln_csv)
    assert data["total_cves"] == 0
    assert data["total_functions_changed"] == 0

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_generate_tangled_histogram_plot(sample_dataset_csv):
    csv_path, tmp_dir = sample_dataset_csv
    data = compute_tangled_distribution(csv_path)

    out_png = os.path.join(tmp_dir, "test_histogram.png")
    result_path = generate_tangled_histogram_plot(data, out_png, "Test Dataset")

    assert os.path.exists(out_png)
    assert result_path == out_png
    assert os.path.getsize(out_png) > 1000


def test_run_tangled_analysis_end_to_end(sample_dataset_csv):
    csv_path, tmp_dir = sample_dataset_csv
    reports_dir = os.path.join(tmp_dir, "reports")

    png_path = run_tangled_analysis(csv_path, output_dir=reports_dir)

    assert png_path is not None
    assert os.path.exists(png_path)
    assert png_path.endswith(".png")
