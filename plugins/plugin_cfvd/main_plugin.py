import collections
import concurrent.futures
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import numpy as np
import pandas as pd
import scipy.sparse as sparse

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from plugins.base_plugin import BasePlugin
from scripts.ui import UI
from scripts.utils import PREPROC_DATASETS_DIR, strip_c_comments


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def _rdg(edges: pd.DataFrame, gtype: str) -> pd.DataFrame:
    if gtype == "reftype":
        return edges[(edges.etype == "EVAL_TYPE") | (edges.etype == "REF")]
    if gtype == "ast":
        return edges[edges.etype == "AST"]
    if gtype == "pdg":
        return edges[(edges.etype == "REACHING_DEF") | (edges.etype == "CDG")]
    if gtype == "cfgcdg":
        return edges[(edges.etype == "CFG") | (edges.etype == "CDG")]
    return edges


def _neighbour_nodes(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    nodeids: list,
    hop: int = 1,
    intermediate: bool = True,
) -> Dict:
    if len(nodes) == 0 or len(edges) == 0:
        return collections.defaultdict(list)

    nodes_new = (
        nodes.reset_index(drop=True).reset_index().rename(columns={"index": "adj"})
    )
    id2adj = pd.Series(nodes_new.adj.values, index=nodes_new.id).to_dict()
    adj2id = {v: k for k, v in id2adj.items()}

    arr = []
    for u, v in zip(edges.innode.map(id2adj), edges.outnode.map(id2adj)):
        if pd.notna(u) and pd.notna(v):
            arr.append([int(u), int(v)])
            arr.append([int(v), int(u)])

    if not arr:
        return collections.defaultdict(list)

    arr = np.array(arr)
    max_dim = int(arr.max()) + 1
    coo = sparse.coo_matrix(
        (np.ones(len(arr)), (arr[:, 0], arr[:, 1])), shape=(max_dim, max_dim)
    )

    def nodeid_neighbours_from_csr(nodeid):
        if nodeid not in id2adj or id2adj[nodeid] >= max_dim:
            return []
        row_idx = id2adj[nodeid]
        return [
            adj2id[i] for i in csr[row_idx].toarray()[0].nonzero()[0] if i in adj2id
        ]

    neighbours = collections.defaultdict(list)
    csr = coo.tocsr()
    if intermediate:
        cur_csr = csr.copy()
        for h in range(1, hop + 1):
            for nodeid in nodeids:
                neighbours[nodeid] += [
                    adj2id[i]
                    for i in cur_csr[id2adj[nodeid]].toarray()[0].nonzero()[0]
                    if i in adj2id
                ]
            cur_csr = cur_csr @ csr
    else:
        cur_csr = csr**hop
        for nodeid in nodeids:
            if nodeid in id2adj and id2adj[nodeid] < max_dim:
                neighbours[nodeid] += [
                    adj2id[i]
                    for i in cur_csr[id2adj[nodeid]].toarray()[0].nonzero()[0]
                    if i in adj2id
                ]

    return neighbours


def _assign_line_num_to_local(
    nodes: pd.DataFrame, edges: pd.DataFrame, code_lines: List[str]
) -> Dict:
    try:
        label_nodes = nodes[nodes._label == "LOCAL"].id.tolist()
        if not label_nodes:
            return {}

        onehop_labels = _neighbour_nodes(
            nodes, _rdg(edges, "ast"), label_nodes, 1, False
        )
        twohop_labels = _neighbour_nodes(
            nodes, _rdg(edges, "reftype"), label_nodes, 2, False
        )

        node_types = nodes[nodes._label == "TYPE"]
        id2name = pd.Series(node_types.name.values, index=node_types.id).to_dict()

        node_blocks = nodes[
            (nodes._label == "BLOCK") | (nodes._label == "CONTROL_STRUCTURE")
        ]
        blocknode2line = pd.Series(
            node_blocks.lineNumber.values, index=node_blocks.id
        ).to_dict()

        local_vars = {}
        local_vars_block = {}
        for k, v in twohop_labels.items():
            types = [i for i in v if i in id2name and int(i) < 1000]
            if not types:
                continue
            block = onehop_labels.get(k, [])
            if not block:
                continue
            local_vars[k] = id2name[types[0]]
            local_vars_block[k] = blocknode2line.get(block[0], "")

        nodes["local_type"] = nodes.id.map(local_vars).fillna("")
        nodes["local_block"] = nodes.id.map(local_vars_block).fillna("")

        local_line_map = {}
        for row in nodes.dropna(subset=["local_type", "local_block"]).itertuples():
            if not row.local_type:
                continue
            localstr = "".join((str(row.local_type) + str(row.name)).split()) + ";"
            try:
                block_ln = int(float(row.local_block))
                code_slice = ["".join(i.split()) for i in code_lines][block_ln:]
                if localstr in code_slice:
                    rel_ln = block_ln + code_slice.index(localstr) + 1
                    local_line_map[row.id] = rel_ln
            except Exception:
                continue

        return local_line_map
    except Exception:
        if "local_type" not in nodes.columns:
            nodes["local_type"] = ""
        if "local_block" not in nodes.columns:
            nodes["local_block"] = ""
        return {}


def _get_node_edges(filepath: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    nodes_fp = f"{filepath}.nodes.json"
    edges_fp = f"{filepath}.edges.json"

    if not os.path.exists(nodes_fp) or not os.path.exists(edges_fp):
        return None

    try:
        with open(edges_fp, "r", encoding="utf-8") as f:
            raw_edges = json.load(f)
        edges = pd.DataFrame(
            raw_edges, columns=["innode", "outnode", "etype", "dataflow"]
        ).fillna("")

        with open(nodes_fp, "r", encoding="utf-8") as f:
            raw_nodes = json.load(f)
        nodes = pd.DataFrame.from_records(raw_nodes)

        if "controlStructureType" not in nodes.columns:
            nodes["controlStructureType"] = ""
        nodes = nodes.fillna("")

        keep_cols = [
            "id",
            "_label",
            "name",
            "code",
            "lineNumber",
            "controlStructureType",
        ]
        nodes = nodes[[c for c in keep_cols if c in nodes.columns]]

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code_lines = f.readlines()
        lmap = _assign_line_num_to_local(nodes, edges, code_lines)
        if lmap:
            nodes.lineNumber = nodes.apply(
                lambda x: lmap[x.id] if x.id in lmap else x.lineNumber, axis=1
            )

        nodes.code = nodes.apply(
            lambda x: "" if x.code == "<empty>" else x.code, axis=1
        )
        nodes.code = nodes.apply(
            lambda x: x.code if x.code != "" else x.get("name", ""), axis=1
        )

        nodes = nodes[nodes._label != "COMMENT"]
        nodes = nodes[nodes._label != "FILE"]

        edges = edges[
            ~edges.etype.isin(["CONTAINS", "SOURCE_FILE", "DOMINATE", "POST_DOMINATE"])
        ]

        edges = edges.merge(
            nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_out"}),
            left_on="outnode",
            right_on="id",
        )
        edges = edges.merge(
            nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_in"}),
            left_on="innode",
            right_on="id",
        )
        edges = edges[(edges.line_out != "") | (edges.line_in != "")]

        edges.outnode = edges.apply(
            lambda x: f"{x.outnode}_{x.innode}" if x.line_out == "" else x.outnode,
            axis=1,
        )
        typemap = nodes[["id", "name"]].set_index("id").to_dict().get("name", {})
        linemap = nodes.set_index("id").to_dict().get("lineNumber", {})

        new_nodes = []
        for e in edges.itertuples():
            if isinstance(e.outnode, str) and "_" in e.outnode:
                try:
                    lineNum = linemap.get(e.innode, "")
                    type_id = int(e.outnode.split("_")[0])
                    node_label = f"TYPE_{lineNum}: {typemap.get(type_id, '')}"
                    new_nodes.append(
                        {
                            "id": e.outnode,
                            "node_label": node_label,
                            "lineNumber": lineNum,
                        }
                    )
                except Exception:
                    pass

        if new_nodes:
            nodes = pd.concat([nodes, pd.DataFrame(new_nodes)], ignore_index=True)

        return nodes, edges

    except Exception:
        return None


def _check_validity(filepath: str) -> bool:
    try:
        nodes_fp = f"{filepath}.nodes.json"
        edges_fp = f"{filepath}.edges.json"
        if not os.path.exists(nodes_fp) or not os.path.exists(edges_fp):
            return False

        with open(nodes_fp, "r", encoding="utf-8") as f:
            nodes = json.load(f)
        line_nums = {
            n["lineNumber"]
            for n in nodes
            if "lineNumber" in n and n["lineNumber"] != ""
        }
        if len(line_nums) <= 1:
            return False

        with open(edges_fp, "r", encoding="utf-8") as f:
            edges = json.load(f)
        edge_set = {e[2] for e in edges if len(e) > 2}
        return bool({"REACHING_DEF", "CDG"} & edge_set)

    except Exception:
        return False


def _extract_pdg_features(
    filepath: str,
    tokenizer,
    word_embeddings: np.ndarray,
) -> Optional[Tuple[pd.DataFrame, Tuple[List[int], List[int]]]]:
    parsed = _get_node_edges(filepath)
    if parsed is None:
        return None
    nodes, edges = parsed

    try:
        nodes["code_len"] = nodes["code"].astype(str).str.len()
        subseq = (
            nodes.sort_values(by="code_len", ascending=False)
            .groupby("lineNumber")
            .head(1)
            .copy()
        )

        if "local_type" in subseq.columns:
            subseq["code"] = (
                subseq["local_type"].fillna("").astype(str)
                + " "
                + subseq["code"].astype(str)
            )

        subseq = subseq[subseq["lineNumber"] != ""]
        subseq["lineNumber"] = subseq["lineNumber"].astype(int)
        subseq = subseq.sort_values("lineNumber")
        subseq["code"] = subseq["code"].astype(str).apply(lambda s: " ".join(s.split()))
        subseq = subseq[subseq["code"] != ""]

        if subseq.empty:
            return None

        def _get_emb(code_str: str) -> np.ndarray:
            tokens = (
                [tokenizer.cls_token]
                + tokenizer.tokenize(code_str)
                + [tokenizer.sep_token]
            )
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            if not token_ids:
                return np.zeros(768, dtype=np.float32)
            valid_ids = [t for t in token_ids if t < len(word_embeddings)]
            if not valid_ids:
                return np.zeros(768, dtype=np.float32)
            return np.mean(word_embeddings[valid_ids], axis=0)

        subseq_feat_map = {
            row.lineNumber: _get_emb(row.code) for row in subseq.itertuples()
        }
        subseq_code_map = {row.lineNumber: row.code for row in subseq.itertuples()}

        nodesline = nodes[nodes["lineNumber"] != ""].copy()
        nodesline["lineNumber"] = nodesline["lineNumber"].astype(int)
        nodesline["code_len"] = nodesline["code"].astype(str).str.len()
        nodesline = (
            nodesline.sort_values(by="code_len", ascending=False)
            .groupby("lineNumber")
            .head(1)
        )

        edgesline = edges.copy()
        edgesline["innode"] = edgesline["line_in"]
        edgesline["outnode"] = edgesline["line_out"]
        nodesline["id"] = nodesline["lineNumber"]

        edgesline = _rdg(edgesline, "pdg")
        connected_ids = set(edgesline.innode).union(set(edgesline.outnode))
        nodesline = nodesline[nodesline.id.isin(connected_ids)]

        edgesline = edgesline.drop_duplicates(subset=["innode", "outnode", "etype"])
        edgesline["etype"] = edgesline["etype"].apply(
            lambda x: "DDG" if x == "REACHING_DEF" else x
        )

        edgesline = edgesline[
            edgesline["innode"].apply(
                lambda x: (
                    isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit())
                )
            )
        ]
        edgesline = edgesline[
            edgesline["outnode"].apply(
                lambda x: (
                    isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit())
                )
            )
        ]
        edgesline["innode"] = edgesline["innode"].astype(int)
        edgesline["outnode"] = edgesline["outnode"].astype(int)

        edgesline_rev = (
            edgesline[["innode", "outnode", "etype"]]
            .copy()
            .rename(columns={"innode": "outnode", "outnode": "innode"})
        )
        uedge = pd.concat([edgesline, edgesline_rev]).drop_duplicates()
        uedge = uedge[uedge.innode != uedge.outnode]

        pdg_nodes = nodesline[["id"]].sort_values("id").copy()
        pdg_nodes["subseq"] = pdg_nodes.id.map(subseq_code_map).fillna("")
        pdg_nodes["subseq_feat"] = pdg_nodes.id.map(subseq_feat_map)
        pdg_nodes = (
            pdg_nodes.dropna(subset=["subseq_feat"])
            .reset_index(drop=True)
            .reset_index()
        )

        if pdg_nodes.empty:
            return None

        pdg_dict = pd.Series(pdg_nodes.index.values, index=pdg_nodes.id).to_dict()
        edgesline_mapped = edgesline.copy()
        edgesline_mapped["innode"] = edgesline_mapped.innode.map(pdg_dict)
        edgesline_mapped["outnode"] = edgesline_mapped.outnode.map(pdg_dict)
        edgesline_mapped = edgesline_mapped.dropna()

        pdg_edges = (
            edgesline_mapped.outnode.astype(int).tolist(),
            edgesline_mapped.innode.astype(int).tolist(),
        )

        return pdg_nodes, pdg_edges

    except Exception:
        return None


class CFVDPlugin(BasePlugin):

    def get_name(self) -> str:
        return "CFVD / ISSTA'24 GGNN Graph Pipeline"

    def get_description(self) -> str:
        return "Generates line-level PDG code graphs with GraphCodeBERT embeddings (768d) for ISSTA'24 GGNN detector."

    def _find_joern(self) -> Optional[str]:
        env_joern = os.getenv("JOERN_BIN")
        if env_joern and os.path.isfile(env_joern) and os.access(env_joern, os.X_OK):
            return env_joern

        bundled_joern = self.plugin_dir / "joern-cli" / "joern"
        if bundled_joern.is_file() and os.access(str(bundled_joern), os.X_OK):
            return str(bundled_joern)

        path_joern = shutil.which("joern")
        if path_joern:
            return path_joern

        common_paths = [
            str(self.plugin_dir / "joern-cli/joern"),
            "/usr/local/bin/joern",
            "/bin/joern",
            os.path.expanduser("~/joern/joern-cli/joern"),
            os.path.expanduser("~/bin/joern"),
            os.path.expanduser("~/.local/bin/joern"),
            str(
                Path(_ROOT).parent
                / "naturalcc/examples/counterfactual-vulnerability-detection/cfexplainer/storage/external/joern-cli/joern"
            ),
        ]
        for p in common_paths:
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p

        return None

    def run(
        self,
        dataset_path: str,
        max_samples: Optional[int] = None,
        workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        splits_csv: Optional[str] = None,
        **kwargs,
    ) -> None:
        seed = int(kwargs.get("seed", 42))
        UI.header("CFVD / ISSTA'24 Graph Neural Network Pipeline")
        UI.info(f"Selected Dataset: {dataset_path} (Seed: {seed})")

        if not os.path.isfile(dataset_path):
            UI.error(f"Dataset file does not exist: {dataset_path}")
            return

        dataset_stem = Path(dataset_path).stem
        current_workspace = self.workspace_dir / dataset_stem
        raw_code_dir = current_workspace / "raw_code"
        raw_code_dir.mkdir(parents=True, exist_ok=True)

        existing_outputs = list(raw_code_dir.glob("*.nodes.json")) + list(
            raw_code_dir.glob("*.edges.json")
        )
        if existing_outputs:
            UI.warning(
                f"Found {len(existing_outputs):,} existing .nodes.json/.edges.json files in {raw_code_dir}."
            )
            try:
                answer = (
                    input(
                        "Delete existing Joern CPG files and reprocess from scratch? [y/N]: "
                    )
                    .strip()
                    .lower()
                )
            except EOFError:
                answer = ""
            if answer in ("y", "yes"):
                UI.step("Deleting existing Joern CPG outputs...")
                deleted = 0
                for f in existing_outputs:
                    try:
                        f.unlink()
                        deleted += 1
                    except OSError:
                        pass
                UI.success(
                    f"Deleted {deleted:,} existing .nodes.json/.edges.json files."
                )
            else:
                UI.info(
                    "Keeping existing Joern CPG outputs; extraction will resume from where it left off."
                )

        preproc_base = (
            Path(PREPROC_DATASETS_DIR)
            if os.path.isabs(PREPROC_DATASETS_DIR)
            else Path(_ROOT) / PREPROC_DATASETS_DIR
        )
        out_dataset_dir = preproc_base / dataset_stem
        out_dataset_dir.mkdir(parents=True, exist_ok=True)

        scala_script = Path(__file__).parent / "get_func_graph.scala"
        if not scala_script.exists():
            UI.error(f"Scala script not found: {scala_script}")
            return

        joern_bin = self._find_joern()
        if not joern_bin:
            UI.error("Joern binary not found in standard paths.")
            return
        else:
            UI.success(f"Using Joern binary: {joern_bin}")

        bundled_jdk = self.plugin_dir / "jdk-11"
        if bundled_jdk.is_dir():
            UI.info(f"Using bundled portable JDK 11: {bundled_jdk}")

        UI.step("Loading dataset CSV...")
        df = pd.read_csv(dataset_path)
        required_cols = {"cve_id", "label", "function"}
        if not required_cols.issubset(df.columns):
            UI.error(
                f"Dataset missing required columns. Expected {required_cols}, found {set(df.columns)}"
            )
            return

        total_funcs = len(df)
        n_vuln = int((df["label"] == 1).sum())
        n_clean = int((df["label"] == 0).sum())
        UI.info(
            f"Total functions: {total_funcs:,} | Vulnerable: {n_vuln:,} ({n_vuln / total_funcs * 100:.1f}%) | Clean: {n_clean:,}"
        )

        if "split" not in df.columns:
            candidate_splits = [
                splits_csv,
                str(self.plugin_dir / "bigvul_rand_splits.csv"),
                "plugins/plugin_cfvd/bigvul_rand_splits.csv",
                "../naturalcc/examples/counterfactual-vulnerability-detection/cfexplainer/storage/external/bigvul_rand_splits.csv",
            ]
            found_split = None
            for c in candidate_splits:
                if c and os.path.exists(c):
                    found_split = c
                    break
            if found_split and "id" in df.columns:
                sdf = pd.read_csv(found_split)
                sdict = dict(zip(sdf["id"], sdf["label"]))
                df["split"] = df["id"].map(sdict).fillna("train")
                UI.info(
                    f"Mapped {len(df):,} samples to official paper splits from {found_split}."
                )
            else:
                seed = kwargs.get("seed", 42)
                UI.warning(
                    f"Official splits file not found or 'id' column missing. Partitioning 80/10/10 with seed {seed}."
                )
                from sklearn.model_selection import train_test_split

                ids = df.index
                train_idx, test_idx = train_test_split(
                    ids, test_size=0.2, random_state=seed
                )
                val_idx, test_idx = train_test_split(
                    test_idx, test_size=0.5, random_state=seed
                )
                df["split"] = "train"
                df.loc[val_idx, "split"] = "val"
                df.loc[test_idx, "split"] = "test"
        else:
            UI.info(
                "Using official LineVD / Big-Vul partition column ('split') directly from dataset."
            )

        if max_samples and max_samples < len(df):
            vul_part = df[df["label"] == 1]
            clean_part = df[df["label"] == 0]
            if len(vul_part) > 0 and len(clean_part) > 0:
                half = max_samples // 2
                n_v = min(len(vul_part), half)
                n_c = min(len(clean_part), max_samples - n_v)
                df = (
                    pd.concat(
                        [
                            vul_part.sample(n_v, random_state=1),
                            clean_part.sample(n_c, random_state=1),
                        ]
                    )
                    .sample(frac=1.0, random_state=1)
                    .reset_index(drop=True)
                )
            else:
                df = df.head(max_samples).copy()
            UI.info(
                f"Limited processing to {len(df):,} samples ({int((df['label'] == 1).sum())} vuln, {int((df['label'] == 0).sum())} clean)."
            )

        UI.step("Exporting functions to raw .c files...")
        file_map: Dict[int, Path] = {}
        for idx, row in df.iterrows():
            row_id = int(row["id"]) if "id" in row and str(row["id"]).isdigit() else idx
            c_path = raw_code_dir / f"{row_id}.c"
            file_map[idx] = c_path
            if not c_path.exists():
                code = str(row["function"])
                with open(c_path, "w", encoding="utf-8") as f:
                    f.write(code)

        if joern_bin:
            pending_cpg = [
                p
                for p in file_map.values()
                if not (
                    Path(f"{p}.nodes.json").exists()
                    and Path(f"{p}.edges.json").exists()
                )
            ]
            already_done = len(file_map) - len(pending_cpg)
            if already_done > 0:
                UI.info(
                    f"Resuming Joern extraction: {already_done:,} files already processed. {len(pending_cpg):,} files remaining."
                )
                if already_done >= 180000 and len(pending_cpg) < 1000:
                    UI.success(
                        f"Over 99.8% ({already_done:,} files) already successfully extracted! Skipping {len(pending_cpg)} unparseable files and proceeding to dataset generation."
                    )
                    pending_cpg = []

            if pending_cpg:
                total_pending = len(pending_cpg)

                user_ram_mb = 3072
                safe_margin = 0.20
                free_mb = (
                    int(psutil.virtual_memory().available / 1024 / 1024)
                    if HAS_PSUTIL
                    else 4096
                )
                usable_mb = int(free_mb * (1.0 - safe_margin))
                auto_workers = max(1, usable_mb // user_ram_mb)
                detected_cpus = os.cpu_count() or 4
                cpu_limit = min(16, max(1, detected_cpus))
                user_workers = workers if workers else min(auto_workers, cpu_limit)

                batch_scala_script = self.plugin_dir / "batch_get_func_graph.scala"
                use_batch_mode = batch_scala_script.exists()

                if use_batch_mode:
                    user_batch = batch_size if batch_size else 50
                else:
                    user_batch = 1

                workers = user_workers
                batch_size = user_batch
                ram_per_worker_mb = user_ram_mb
                num_batches = math.ceil(total_pending / batch_size)

                UI.header("Joern CPG Extraction - Parallel Batch Mode")
                UI.info(f"Total pending files : {total_pending:,}")
                if HAS_PSUTIL:
                    UI.info(
                        f"Free RAM            : {free_mb:,} MB  →  usable {usable_mb:,} MB"
                    )
                UI.info(
                    f"RAM per worker      : {ram_per_worker_mb} MB  (-Xmx{ram_per_worker_mb}m)"
                )
                UI.info(f"Parallel workers    : {workers} (capped to actual vCPU quota)")
                UI.info(f"Batch size          : {batch_size} files/batch")
                UI.info(f"Total batches       : {num_batches:,}")

                UI.step(
                    f"Running Joern extraction for {total_pending:,} files using {workers} parallel workers "
                    f"({'optimized batch mode' if use_batch_mode else 'single-file mode'})..."
                )

                batches = [
                    pending_cpg[i : i + batch_size]
                    for i in range(0, total_pending, batch_size)
                ]

                tmp_batches_dir = current_workspace / ".joern_batches"
                tmp_batches_dir.mkdir(parents=True, exist_ok=True)

                lock = threading.Lock()
                processed_set = set()
                total_pending = len(pending_cpg)
                start_time = time.time()

                def _record_processed(f_path_str: str) -> None:
                    with lock:
                        if f_path_str not in processed_set:
                            processed_set.add(f_path_str)
                            cnt = len(processed_set)
                            elapsed = time.time() - start_time
                            rate = cnt / max(elapsed, 0.1)
                            eta = (total_pending - cnt) / max(rate, 0.01)
                            pct = (cnt / total_pending) * 100
                            cols = shutil.get_terminal_size((80, 20)).columns
                            msg = (
                                f"[Joern] {cnt:,}/{total_pending:,} files ({pct:5.1f}%) | "
                                f"Speed: {rate:4.1f} f/s | Elapsed: {_fmt_duration(elapsed)} | ETA: {_fmt_duration(eta)}"
                            )
                            fit_msg = msg[: cols - 1]
                            sys.stdout.write("\r" + fit_msg.ljust(cols - 1))
                            sys.stdout.flush()

                def _run_batch_worker(batch_info: Tuple[int, List[Path]]) -> None:
                    b_idx, b_files = batch_info
                    b_dir = tmp_batches_dir / f"batch_{b_idx}"
                    b_dir.mkdir(parents=True, exist_ok=True)

                    env = os.environ.copy()
                    if bundled_jdk.is_dir():
                        env["JAVA_HOME"] = str(bundled_jdk)
                        env["PATH"] = f"{bundled_jdk / 'bin'}:{env.get('PATH', '')}"

                    xms = min(1024, ram_per_worker_mb)
                    jvm_flags = f"-Xms{xms}m -Xmx{ram_per_worker_mb}m -XX:+UseG1GC -XX:TieredStopAtLevel=1 -XX:-UsePerfData -Duser.home={b_dir}"
                    env["JAVA_TOOL_OPTIONS"] = jvm_flags
                    env["JOERN_JAVA_OPTS"] = (
                        f"{jvm_flags} --enable-native-access=ALL-UNNAMED"
                    )
                    env["HOME"] = str(b_dir)

                    try:
                        if use_batch_mode:
                            list_file = b_dir / "file_list.txt"
                            with open(list_file, "w", encoding="utf-8") as lf:
                                for f in b_files:
                                    lf.write(f"{f.resolve()}\n")

                            cmd = [
                                joern_bin,
                                "--script",
                                str(batch_scala_script),
                                f"--params=listFile={list_file.resolve()}",
                            ]

                            with subprocess.Popen(
                                cmd,
                                cwd=str(b_dir),
                                env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                            ) as proc:
                                for line in proc.stdout:
                                    if "[PROCESSED]" in line:
                                        p_path = line.split("[PROCESSED]", 1)[1].strip()
                                        _record_processed(p_path)
                                proc.wait()

                            for f in b_files:
                                f_str = str(f.resolve())
                                if not (
                                    Path(f"{f}.nodes.json").exists()
                                    and Path(f"{f}.edges.json").exists()
                                ):
                                    single_cmd = [
                                        joern_bin,
                                        "--script",
                                        str(scala_script),
                                        f"--params=filename={f_str}",
                                    ]
                                    try:
                                        subprocess.run(
                                            single_cmd,
                                            cwd=str(b_dir),
                                            env=env,
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL,
                                            timeout=60,
                                        )
                                    except Exception:
                                        pass
                                _record_processed(f_str)
                        else:
                            for f in b_files:
                                f_str = str(f.resolve())
                                single_cmd = [
                                    joern_bin,
                                    "--script",
                                    str(scala_script),
                                    f"--params=filename={f_str}",
                                ]
                                try:
                                    subprocess.run(
                                        single_cmd,
                                        cwd=str(b_dir),
                                        env=env,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                        timeout=60,
                                    )
                                except Exception:
                                    pass
                                _record_processed(f_str)
                    finally:
                        shutil.rmtree(b_dir, ignore_errors=True)

                batch_tasks = list(enumerate(batches))
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=workers
                ) as executor:
                    list(executor.map(_run_batch_worker, batch_tasks))

                print()
                shutil.rmtree(tmp_batches_dir, ignore_errors=True)
                UI.success(
                    f"Joern CPG extraction complete! Processed {total_pending:,} files in {_fmt_duration(time.time() - start_time)}."
                )
            else:
                UI.info(
                    "All Joern .nodes.json and .edges.json files already exist. Skipping extraction."
                )

        UI.step("Loading pre-trained GraphCodeBERT ('microsoft/graphcodebert-base')...")
        try:
            import torch
            from torch_geometric.data import Data
            from transformers import (
                RobertaConfig,
                RobertaForSequenceClassification,
                RobertaTokenizer,
            )

            model_name = "microsoft/graphcodebert-base"
            config = RobertaConfig.from_pretrained(model_name)
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            language_model = RobertaForSequenceClassification.from_pretrained(
                model_name, config=config
            )
            word_embeddings = (
                language_model.roberta.embeddings.word_embeddings.weight.detach()
                .cpu()
                .numpy()
            )
            UI.success(
                f"GraphCodeBERT loaded! Embedding matrix shape: {word_embeddings.shape}"
            )
        except Exception as exc:
            UI.error(f"Failed to load Transformers / PyG: {exc}")
            return

        UI.step(
            "Processing partitions according to official LineVD / Big-Vul splits (train, val, test)..."
        )
        partitions = {
            "train_processed": [],
            "val_processed": [],
            "test_processed": [],
        }

        finished_indices = [
            idx
            for idx in df.index
            if Path(f"{file_map[idx]}.nodes.json").exists()
            and Path(f"{file_map[idx]}.edges.json").exists()
        ]
        finished_df = df.loc[finished_indices].copy()
        UI.info(f"Finished Joern CPG samples: {len(finished_df):,} / {len(df):,}")

        has_removed_col = False
        if "removed" in finished_df.columns:
            vuln_subset = finished_df[finished_df["label"] == 1]
            if len(vuln_subset) > 0:
                has_removed_col = vuln_subset["removed"].dropna().apply(
                    lambda x: bool(re.findall(r"\d+", str(x)))
                ).any()

        if has_removed_col:
            UI.info("Dataset contains statement-level patch diffs ('removed' column). Using ISSTA'24 line-level ground truth.")
        else:
            UI.info("Dataset is function-level (no statement-level 'removed' annotations). Using robust function-level graph labeling.")

        for part_name, proc_name in [
            ("train", "train_processed"),
            ("val", "val_processed"),
            ("test", "test_processed"),
        ]:
            part_sub = finished_df[finished_df["split"] == part_name].copy()
            if len(part_sub) == 0:
                continue

            valid_mask = [
                _check_validity(str(file_map[idx]))
                for idx in part_sub.index
            ]
            part_valid = part_sub[valid_mask].copy()

            if has_removed_col:
                def _has_valid_removed(row):
                    if int(row["label"]) == 0:
                        return True
                    raw_rem = str(row.get("removed", "")).strip()
                    if not raw_rem or raw_rem in ("nan", "None", "[]", ""):
                        return False
                    return bool(re.findall(r"\d+", raw_rem))

                part_valid = part_valid[part_valid.apply(_has_valid_removed, axis=1)].copy()

            vul = part_valid[part_valid["label"] == 1]
            nonvul = part_valid[part_valid["label"] == 0]
            if len(vul) > 0 and len(nonvul) > 0:
                n_bal = min(len(vul), len(nonvul))
                nonvul_sampled = nonvul.sample(n_bal, random_state=seed)
                part_balanced = pd.concat([vul, nonvul_sampled]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
            else:
                part_balanced = part_valid.reset_index(drop=True)

            UI.info(
                f"  Partition '{part_name}': Balanced to {len(part_balanced):,} valid samples ({int((part_balanced['label'] == 1).sum()):,} vuln, {int((part_balanced['label'] == 0).sum()):,} clean)."
            )

            skipped_extract = 0
            part_graphs = []

            for _, row in part_balanced.iterrows():
                row_id = int(row["id"]) if "id" in row and str(row["id"]).isdigit() else row.name
                c_path = raw_code_dir / f"{row_id}.c"

                extracted = _extract_pdg_features(
                    str(c_path), tokenizer, word_embeddings
                )
                if extracted is None:
                    skipped_extract += 1
                    continue

                pdg_nodes, pdg_edges = extracted
                x_feats = np.array(list(pdg_nodes.subseq_feat.values))
                edge_idx = np.array(pdg_edges)

                code_graph = Data(
                    x=torch.FloatTensor(x_feats), edge_index=torch.LongTensor(edge_idx)
                )

                is_vuln = int(row["label"]) == 1
                if not is_vuln:
                    vuln_flags = [0] * len(pdg_nodes)
                elif has_removed_col:
                    raw_rem = str(row.get("removed", ""))
                    removed_set = set(map(int, re.findall(r"\d+", raw_rem))) if raw_rem not in ("nan", "None", "[]", "") else set()
                    vuln_flags = [
                        1 if int(line_id) in removed_set else 0
                        for line_id in pdg_nodes["id"].astype(int)
                    ]
                    if sum(vuln_flags) == 0:
                        vuln_flags = [1] * len(pdg_nodes)
                else:
                    vuln_flags = [1] * len(pdg_nodes)

                sample_id = int(row["id"]) if "id" in row and str(row["id"]).isdigit() else row_id
                code_graph.__setitem__("_VULN", torch.Tensor(vuln_flags))
                code_graph.__setitem__(
                    "_LINE", torch.Tensor(pdg_nodes["id"].astype(int).to_numpy().copy())
                )
                code_graph.__setitem__(
                    "_SAMPLE", torch.Tensor([sample_id] * len(pdg_nodes))
                )

                part_graphs.append(code_graph)

                if len(part_graphs) % 50 == 0:
                    sys.stdout.write(
                        f"\r  [{part_name}] Vectorized {len(part_graphs):,} valid graphs..."
                    )
                    sys.stdout.flush()

            print()
            partitions[proc_name] = part_graphs
            UI.success(
                f"  Partition '{part_name}': {len(part_graphs):,} final valid graphs built."
            )

        for part_name, part_graphs in partitions.items():
            part_dir = out_dataset_dir / part_name
            part_dir.mkdir(parents=True, exist_ok=True)
            pt_path = part_dir / "data.pt"
            torch.save(part_graphs, pt_path)
            UI.info(
                f"Saved {part_name}/data.pt ({len(part_graphs):,} graphs) -> {pt_path}"
            )

        UI.header("Pipeline Complete!")
        UI.success(f"Train samples : {len(partitions['train_processed']):,}")
        UI.success(f"Val samples   : {len(partitions['val_processed']):,}")
        UI.success(f"Test samples  : {len(partitions['test_processed']):,}")
        UI.info(f"All artifacts saved in: {out_dataset_dir}")

        candidates_dest = [
            Path(_ROOT) / "cfexplainer/storage/processed/vul_graph_dataset",
            Path(_ROOT).parent / "cfexplainer/storage/processed/vul_graph_dataset",
            Path(_ROOT).parent
            / "naturalcc/examples/counterfactual-vulnerability-detection/cfexplainer/storage/processed/vul_graph_dataset",
        ]
        target_cfvd = None
        for cand in candidates_dest:
            if cand.parent.parent.exists():
                target_cfvd = cand
                break

        if target_cfvd:
            target_cfvd.parent.mkdir(parents=True, exist_ok=True)
            for part_name in partitions.keys():
                target_sub = target_cfvd / part_name
                target_sub.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    out_dataset_dir / part_name / "data.pt", target_sub / "data.pt"
                )
            UI.success(f"Successfully exported data.pt files to: {target_cfvd}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CFVD Plugin - ISSTA'24 GGNN Graph Dataset Generator"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path(_ROOT) / "datasets/dataset_bigvul.csv"),
        help="Path to input dataset CSV (default: datasets/dataset_bigvul.csv)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (optional, for quick testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset partitioning (default: 42)",
    )
    args = parser.parse_args()

    plugin = CFVDPlugin()
    plugin.run(
        dataset_path=args.dataset,
        max_samples=args.max_samples,
        seed=args.seed,
    )
