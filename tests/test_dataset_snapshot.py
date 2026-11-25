import hashlib
import os

import pandas as pd
import pytest

from scripts.dataset_builder import process_cve_dataset

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "raw_data")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(RAW_DATA_DIR, "CVE-2024-26600")),
    reason="tests/fixtures/raw_data/CVE-2024-26600 required for golden master snapshot tests",
)


import shutil
import tempfile

@pytest.fixture(scope="module")
def dataset_df() -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp_dir:
        src_cve = os.path.join(RAW_DATA_DIR, "CVE-2024-26600")
        dst_cve = os.path.join(tmp_dir, "CVE-2024-26600")
        shutil.copytree(src_cve, dst_cve)
        output_csv = os.path.join(tmp_dir, "snapshot_test.csv")

        process_cve_dataset(tmp_dir, output_csv, interactive=False)
        df = pd.read_csv(output_csv)
        yield df


EXPECTED_COLUMNS = ["cve_id", "filename", "hash", "lines", "label", "function"]

EXPECTED_ROW_COUNT = 19


class TestCsvStructure:
    def test_columns_exact(self, dataset_df):
        assert list(dataset_df.columns) == EXPECTED_COLUMNS

    def test_row_count(self, dataset_df):
        assert len(dataset_df) == EXPECTED_ROW_COUNT

    def test_no_null_fields(self, dataset_df):
        assert not dataset_df.isnull().any().any(), "Found null fields in CSV"

    def test_labels_binary(self, dataset_df):
        unique_labels = set(dataset_df["label"].unique())
        assert unique_labels.issubset({0, 1}), f"Unexpected labels: {unique_labels}"

    def test_no_duplicate_rows(self, dataset_df):
        assert dataset_df["hash"].is_unique, "Duplicate hashes found in CSV"


class TestMetadata:
    def test_cve_id_all_correct(self, dataset_df):
        assert (dataset_df["cve_id"] == "CVE-2024-26600").all()

    def test_filename_all_correct(self, dataset_df):
        assert (dataset_df["filename"] == "phy-omap-usb2.c").all()


PATCHED_FUNCTIONS_SNAPSHOT = {
    "omap_usb_set_vbus_vuln": (
        "041a5737718331c8126b8b993dd9cb39113be5ff244544e1bfd7f0f96faa076b",
        1,
        9,
    ),
    "omap_usb_set_vbus_patched": (
        "20a4d5d88fd77646c012450c36d7fef7cbc3d42c8d3c9a970b35be39efca4d6e",
        0,
        9,
    ),
    "omap_usb_start_srp_vuln": (
        "6033685fa4fe8762eca69a92896c4e0f3e6ca0a780050739a305701f79e06bfd",
        1,
        9,
    ),
    "omap_usb_start_srp_patched": (
        "1ba43677382b0e65a7c58813d66bc56de6d41d0394e31871d31abd2cc343bcc0",
        0,
        9,
    ),
}


class TestPatchedFunctionLabels:
    def test_vulnerable_count(self, dataset_df):
        assert len(dataset_df[dataset_df["label"] == 1]) == 2

    @pytest.mark.parametrize("name,snapshot", PATCHED_FUNCTIONS_SNAPSHOT.items())
    def test_patched_hash_and_label(self, dataset_df, name, snapshot):
        expected_hash, expected_label, expected_size = snapshot
        row = dataset_df[dataset_df["hash"] == expected_hash]
        assert len(row) == 1, f"Hash not found for {name}: {expected_hash}"
        assert row.iloc[0]["label"] == expected_label, f"Wrong label for {name}"
        assert row.iloc[0]["lines"] == expected_size, f"Wrong line count for {name}"


class TestHashIntegrity:
    def test_all_hashes_match_function_content(self, dataset_df):
        from scripts.utils import strip_c_comments

        mismatches = []
        for idx, row in dataset_df.iterrows():
            clean_code = strip_c_comments(row["function"])
            normalized = "".join(clean_code.split())
            computed = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if computed != row["hash"]:
                mismatches.append(
                    f"ROW {idx}: stored={row['hash'][:16]}... computed={computed[:16]}..."
                )
        assert not mismatches, "Hash mismatches:\n" + "\n".join(mismatches)


LINES_SNAPSHOT = [
    ("b5058a8305e1cdee9b2cf0387a1570ca8febc637cf7e8f12c22dc34bc95f5a25", 4),
    ("3f772beda6fe0457ddb2ce4c5ae2a49a9f632f540de59c7cb9060addd083cd85", 5),
    ("df2720a69eecbcedaaa88148fd648de6c0055028125ad5ba7e4d24631b075a86", 12),
    ("041a5737718331c8126b8b993dd9cb39113be5ff244544e1bfd7f0f96faa076b", 9),
    ("20a4d5d88fd77646c012450c36d7fef7cbc3d42c8d3c9a970b35be39efca4d6e", 9),
    ("6033685fa4fe8762eca69a92896c4e0f3e6ca0a780050739a305701f79e06bfd", 9),
    ("1ba43677382b0e65a7c58813d66bc56de6d41d0394e31871d31abd2cc343bcc0", 9),
    ("f3f1ab289f044d9c38b3a6f9633bef8e0439923c78851b8d0020028285eca592", 8),
    ("aea9e6f25394979b6ec9cfb232fb9edf88ddd2c8caa6915eb902bdced93f2b20", 9),
    ("9ba251955880833c55d25685ea067f0834042d877ca0805e687c68c8a2775de4", 19),
    ("d365db71a001c86d9dbe7c4ffec496a132ffa8aed076915a8a2f708e032c6bb2", 6),
    ("cbb8b83fac0766b7ef73afd6b797e29b78d0ceda8276070c517d3cb1b761d1a1", 6),
    ("7d5ff000e4d9e67e5a3a759a2ccdda21b075f3bda563b205618a5ebcaf5cf519", 8),
    ("5bf16bacb1f7c36d586e05fe49c522626bcd1e45b861f580d2a18d690dd7ae9f", 26),
    ("bf502bde988d046929b77afbc3c5d48174701a66f75eabded5c3f16f0847899d", 30),
    ("1c6855d54f523194c86e62def074a05602a69796f6eedcd8300fc7f237d839b9", 6),
    ("eb042e8f946af0b4f64335c63a1b0cbc03a22857f02f8e3b6f6cfbc5da82e211", 19),
    ("613fb211db5b035dcdf2bc611c49d45fef613583cc5f3094a811ed547b46ec5c", 137),
    ("7b6752bb6839d3ab4b859ede56650f48eb48d98aca5a62e7543b05b141edd781", 7),
]


class TestLinesSnapshot:
    @pytest.mark.parametrize("expected_hash,expected_size", LINES_SNAPSHOT)
    def test_function_size(self, dataset_df, expected_hash, expected_size):
        row = dataset_df[dataset_df["hash"] == expected_hash]
        assert len(row) == 1, f"Hash not found: {expected_hash[:16]}..."
        assert row.iloc[0]["lines"] == expected_size, (
            f"Wrong line count for hash {expected_hash[:16]}...: "
            f"expected={expected_size}, got={row.iloc[0]['lines']}"
        )


EXPECTED_FUNCTION_FIRST_LINES = {
    "static inline u32 omap_usb_readl(void __iomem *addr, unsigned int offset)",
    "static inline void omap_usb_writel(void __iomem *addr, unsigned int offset,",
    "int omap_usb2_set_comparator(struct phy_companion *comparator)",
    "static int omap_usb_set_vbus(struct usb_otg *otg, bool enabled)",
    "static int omap_usb_start_srp(struct usb_otg *otg)",
    "static int omap_usb_set_host(struct usb_otg *otg, struct usb_bus *host)",
    "static int omap_usb_set_peripheral(struct usb_otg *otg,",
    "static int omap_usb_phy_power(struct omap_usb *phy, int on)",
    "static int omap_usb_power_off(struct phy *x)",
    "static int omap_usb_power_on(struct phy *x)",
    "static int omap_usb2_disable_clocks(struct omap_usb *phy)",
    "static int omap_usb2_enable_clocks(struct omap_usb *phy)",
    "static int omap_usb_init(struct phy *x)",
    "static int omap_usb_exit(struct phy *x)",
    "static void omap_usb2_init_errata(struct omap_usb *phy)",
    "static int omap_usb2_probe(struct platform_device *pdev)",
    "static void omap_usb2_remove(struct platform_device *pdev)",
}


class TestFunctionPresence:
    def test_no_unexpected_extra_functions(self, dataset_df):
        actual_first_lines = set(
            dataset_df["function"].str.strip().str.split("\n").str[0].str.strip()
        )
        extra = actual_first_lines - EXPECTED_FUNCTION_FIRST_LINES
        assert not extra, f"Unexpected extra functions found: {extra}"

    def test_all_expected_functions_present(self, dataset_df):
        actual_first_lines = set(
            dataset_df["function"].str.strip().str.split("\n").str[0].str.strip()
        )
        missing = EXPECTED_FUNCTION_FIRST_LINES - actual_first_lines
        assert not missing, f"Expected functions missing: {missing}"

    def test_omap_usb_writel_present(self, dataset_df):
        actual_first_lines = set(
            dataset_df["function"].str.strip().str.split("\n").str[0].str.strip()
        )
        assert any("omap_usb_writel" in line for line in actual_first_lines)

    def test_omap_usb_set_peripheral_present(self, dataset_df):
        actual_first_lines = set(
            dataset_df["function"].str.strip().str.split("\n").str[0].str.strip()
        )
        assert any("omap_usb_set_peripheral" in line for line in actual_first_lines)
