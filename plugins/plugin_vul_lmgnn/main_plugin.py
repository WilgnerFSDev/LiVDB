import codecs
import concurrent.futures
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from plugins.base_plugin import BasePlugin
from scripts.ui import UI
from scripts.utils import PREPROC_DATASETS_DIR

_NODE_LABELS = [
    "Block", "Call", "Comment", "ControlStructure", "File", "Identifier",
    "FieldIdentifier", "Literal", "Local", "Member", "MetaData", "Method",
    "MethodInst", "MethodParameterIn", "MethodParameterOut", "MethodReturn",
    "Namespace", "NamespaceBlock", "Return", "Type", "TypeDecl", "Unknown"
]
_OPERATORS = [
    'addition', 'addressOf', 'and', 'arithmeticShiftRight', 'assignment',
    'assignmentAnd', 'assignmentArithmeticShiftRight', 'assignmentDivision',
    'assignmentMinus', 'assignmentMultiplication', 'assignmentOr', 'assignmentPlus',
    'assignmentShiftLeft', 'assignmentXor', 'cast', 'conditionalExpression',
    'division', 'equals', 'fieldAccess', 'greaterEqualsThan', 'greaterThan',
    'indirectFieldAccess', 'indirectIndexAccess', 'indirection', 'lessEqualsThan',
    'lessThan', 'logicalAnd', 'logicalNot', 'logicalOr', 'minus', 'modulo', 'multiplication',
    'not', 'notEquals', 'or', 'postDecrement', 'plus', 'postIncrement', 'preDecrement',
    'preIncrement', 'shiftLeft', 'sizeOf', 'subtraction'
]
_ALL_NODE_TYPES = _NODE_LABELS + _OPERATORS
NODE_TYPE_MAP: Dict[str, int] = {label: i for i, label in enumerate(_ALL_NODE_TYPES)}
UNKNOWN_TYPE_ID = len(_ALL_NODE_TYPES) + 1

DEFAULT_NODES_DIM = 205
DEFAULT_EMB_SIZE = 769
CHUNK_SIZE = 500

_KEYWORDS = frozenset({
    '__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export', '__far16', '__far32',
    '__fastcall', '__finally', '__import', '__inline', '__int16', '__int32', '__int64', '__int8',
    '__leave', '__optlink', '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
    '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except', '_Export', '_Far16',
    '_Far32', '_Fastcall', '_finally', '_Import', '_inline', '_int16', '_int32', '_int64',
    '_int8', '_leave', '_Optlink', '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas',
    'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
    'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const', 'const_cast', 'constexpr',
    'continue', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
    'explicit', 'export', 'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if',
    'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr',
    'operator', 'or', 'or_eq', 'override', 'private', 'protected', 'public', 'register',
    'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert',
    'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try',
    'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
    'wchar_t', 'while', 'xor', 'xor_eq', 'NULL'
})
_MAIN_SET = frozenset({'main'})
_MAIN_ARGS = frozenset({'argc', 'argv'})

_OPERATORS3 = {'<<=', '>>='}
_OPERATORS2 = {
    '->', '++', '--', '**',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
_OPERATORS1 = {
    '(', ')', '[', ']', '.',
    '+', '&',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':',
    '{', '}', '!', '~'
}

def _to_regex(lst):
    return r'|'.join([f"({re.escape(el)})" for el in lst])

_REGEX_SPLIT_OPERATORS = _to_regex(_OPERATORS3) + '|' + _to_regex(_OPERATORS2) + '|' + _to_regex(_OPERATORS1)


def clean_gadget(gadget: List[str]) -> List[str]:
    fun_symbols = {}
    var_symbols = {}
    fun_count = 1
    var_count = 1

    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b((?!\s*\**\w+))(?!\s*\()')
    cleaned_gadget = []

    for line in gadget:
        ascii_line = re.sub(r'[^\x00-\x7f]', r'', line)
        hex_line = re.sub(r'0[xX][0-9a-fA-F]+', "HEX", ascii_line)
        user_fun = rx_fun.findall(hex_line)
        user_var = rx_var.findall(hex_line)

        for fun_name in user_fun:
            if fun_name not in _MAIN_SET and fun_name not in _KEYWORDS:
                if fun_name not in fun_symbols:
                    fun_symbols[fun_name] = f'FUN{fun_count}'
                    fun_count += 1
                hex_line = re.sub(r'\b(' + re.escape(fun_name) + r')\b(?=\s*\()', fun_symbols[fun_name], hex_line)

        for var_name in user_var:
            v0 = var_name[0]
            if v0 not in _KEYWORDS and v0 not in _MAIN_ARGS:
                if v0 not in var_symbols:
                    var_symbols[v0] = f'VAR{var_count}'
                    var_count += 1
                hex_line = re.sub(r'\b(' + re.escape(v0) + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()',
                                  var_symbols[v0], hex_line)

        cleaned_gadget.append(hex_line)
    return cleaned_gadget


def vul_lmgnn_tokenizer(code: str, flag: bool = True) -> List[str]:
    if not code:
        return []
    no_str_lit_line = re.sub(r'["]([^"\\\n]|\\.|\\\n)*["]', '', code)
    no_char_lit_line = re.sub(r"'.*?'", "", no_str_lit_line)
    cleaned_code = no_char_lit_line

    if flag:
        try:
            cleaned_code = codecs.getdecoder("unicode_escape")(no_char_lit_line)[0]
        except Exception:
            cleaned_code = no_char_lit_line

    gadget: List[str] = []
    for line in cleaned_code.splitlines():
        stripped = line.strip()
        if stripped:
            gadget.append(stripped)

    clean = clean_gadget(gadget)
    tokenized: List[str] = []

    for cg in clean:
        if not cg:
            continue
        pat = re.compile(r'(/\*([^*]|(\*+[^*\/]))*\*+\/)|(\/\/.*)')
        cg = re.sub(pat, '', cg)
        cg = re.sub(r'(\n)|(\\\\n)|(\\\\)|(\\t)|(\\r)', '', cg)
        splitter = r' +|' + _REGEX_SPLIT_OPERATORS + r'|(\/)|(\;)|(\-)|(\*)'
        tokens = re.split(splitter, cg)
        tokens = [t for t in tokens if t and t.strip()]
        tokenized.extend(tokens)

    return tokenized


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m:02d}m {sec:02d}s"


class VulLMGNNPlugin(BasePlugin):

    def get_name(self) -> str:
        return "Vul-LMGNN (1:1 GNN + CodeBERT Pipeline)"

    def get_description(self) -> str:
        return (
            "Extracts Joern CPGs, applies clean_gadget normalization, temporal line ordering, "
            "and computes CodeBERT node embeddings (dim=769) strictly matching the original Vul-LMGNN paper."
        )

    def _find_joern(self) -> Tuple[Optional[Path], Optional[Path]]:
        bundled_jdk = Path(_ROOT) / "plugins" / "plugin_cfvd" / "jdk-11"
        if not bundled_jdk.is_dir():
            bundled_jdk = None

        joern_candidates = [
            self.plugin_dir / "joern-cli" / "joern",
            Path(_ROOT) / "plugins" / "plugin_cfvd" / "joern-cli" / "joern",
            Path.home() / "bin" / "joern",
            Path("/usr/local/bin/joern"),
        ]
        joern_bin = None
        for c in joern_candidates:
            if c.is_file() and os.access(c, os.X_OK):
                joern_bin = c.resolve()
                break

        if not joern_bin:
            sys_j = shutil.which("joern")
            if sys_j:
                joern_bin = Path(sys_j).resolve()

        return joern_bin, bundled_jdk

    def run(self, dataset_path: str) -> None:
        UI.header("Vul-LMGNN: 1:1 GNN + CodeBERT Data Preparation Pipeline")

        UI.step("Step 1/6: Pre-flight environment check...")
        joern_bin, bundled_jdk = self._find_joern()
        if not joern_bin:
            UI.error("Joern CLI binary ('joern') not found!")
            return
        UI.success(f"Joern binary: {joern_bin}")
        if bundled_jdk:
            UI.info(f"Using bundled portable JDK 11: {bundled_jdk}")

        try:
            from torch_geometric.data import Data
            from transformers import RobertaModel, RobertaTokenizer
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            UI.success(f"PyTorch Geometric & Transformers ready. Device: {device}")
        except ImportError as exc:
            UI.error(f"Missing required Python libraries: {exc}")
            return

        UI.step(f"Step 2/6: Loading dataset from {dataset_path}...")
        if not os.path.exists(dataset_path):
            UI.error(f"Dataset not found: {dataset_path}")
            return

        df = pd.read_csv(dataset_path)
        total_raw = len(df)
        df = df.dropna(subset=["function"])
        df = df[df["function"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
        cleaned_total = len(df)

        vuln_count = int((df["label"] == 1).sum())
        clean_count = cleaned_total - vuln_count
        ratio = (clean_count / vuln_count) if vuln_count > 0 else 0

        UI.info(f"Loaded: {total_raw:,} rows -> {cleaned_total:,} valid functions.")
        UI.info(f"  Vulnerable (1): {vuln_count:,} ({vuln_count/cleaned_total*100:.1f}%)")
        UI.info(f"  Clean      (0): {clean_count:,} ({clean_count/cleaned_total*100:.1f}%)")
        UI.info(f"  Class ratio   : 1:{ratio:.1f}")

        dataset_stem = Path(dataset_path).stem
        current_workspace = self.workspace_dir / dataset_stem
        out_chunks_dir = Path(PREPROC_DATASETS_DIR) / "vul_lmgnn" / dataset_stem

        if current_workspace.exists() or out_chunks_dir.exists():
            clean_choice = UI.ask_choice(
                "Previous artifacts detected for this dataset. Choose execution mode:",
                [
                    ("0", "Clean Run (Delete previous workspace & chunks to generate from scratch 1:1)", ""),
                    ("1", "Resume / Keep existing files (Reuse existing raw files & Joern export)", ""),
                ]
            )
            if clean_choice == "0":
                UI.info("Clearing previous workspace and output chunks...")
                if current_workspace.exists():
                    shutil.rmtree(current_workspace, ignore_errors=True)
                if out_chunks_dir.exists():
                    shutil.rmtree(out_chunks_dir, ignore_errors=True)
                UI.success("Cleared previous artifacts! Starting clean 1:1 run from scratch.")

        raw_dir = current_workspace / "raw_code"
        raw_dir.mkdir(parents=True, exist_ok=True)

        UI.step("Step 3/6: Exporting source code to raw .c files...")
        idx_to_func = {}
        file_map = {}
        for idx, row in df.iterrows():
            row_id = int(row["id"]) if "id" in row and str(row["id"]).isdigit() else idx
            func_code = str(row["function"])
            lbl = int(row["label"])
            filename = f"{row_id}_{lbl}.c"
            file_path = raw_dir / filename
            file_map[idx] = file_path
            idx_to_func[row_id] = func_code

            if not file_path.exists():
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(func_code)

        UI.success(f"Exported {len(file_map):,} .c files to {raw_dir}")

        UI.step("Step 4/6: Extracting CPG graphs with Joern...")
        graphs_dir = current_workspace / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

        scala_script = self.plugin_dir / "joern_to_vul_lmgnn.sc"
        if not scala_script.exists():
            UI.error(f"Joern export script not found: {scala_script}")
            return

        cpg_bin = current_workspace / "cpg.bin"

        ammonite_home = current_workspace / ".ammonite_home"
        ammonite_home.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if bundled_jdk:
            env["JAVA_HOME"] = str(bundled_jdk)
            env["PATH"] = f"{bundled_jdk / 'bin'}:{env.get('PATH', '')}"
        env["JAVA_OPTS"] = f"-Duser.home={ammonite_home} -Xmx8g"

        joern_parse_bin = joern_bin.parent / "joern-parse"
        if not joern_parse_bin.exists():
            joern_parse_bin = Path(shutil.which("joern-parse"))

        existing_jsons = list(graphs_dir.glob("*.json"))
        if not cpg_bin.exists() and len(existing_jsons) < len(file_map):
            UI.info("Generating CPG binary using joern-parse...")
            start_cpg = time.time()
            parse_cmd = [str(joern_parse_bin), str(raw_dir), "--out", str(cpg_bin)]
            proc = subprocess.Popen(parse_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                line_str = line.strip()
                if line_str:
                    print(f"  [joern-parse] {line_str}")
            proc.wait()
            if proc.returncode != 0:
                UI.error(f"joern-parse failed with exit code {proc.returncode}")
                return
            UI.success(f"CPG binary generated in {_fmt_duration(time.time() - start_cpg)}.")

        if len(existing_jsons) < len(file_map):
            UI.info("Extracting AST/CFG/DDG graphs to JSON with 1:1 attributes...")
            start_export = time.time()
            export_cmd = [
                str(joern_bin),
                "--script", str(scala_script.resolve()),
                f"--params=cpgPath={cpg_bin.resolve()},outputDir={graphs_dir.resolve()},minNodes=1,maxNodes=499,minEdges=1"
            ]
            proc_export = subprocess.Popen(export_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc_export.stdout:
                line_str = line.strip()
                if line_str:
                    print(f"  [joern-export] {line_str}")
            proc_export.wait()
            if proc_export.returncode != 0:
                UI.error(f"Joern graph export failed with exit code {proc_export.returncode}")
                return
            UI.success(f"Joern graph export finished in {_fmt_duration(time.time() - start_export)}.")

        json_files = sorted(list(graphs_dir.glob("*.json")))
        UI.success(f"Total function graph JSONs ready: {len(json_files):,}.")
        if not json_files:
            UI.error("No graph JSON files found.")
            return

        UI.step("Step 5/6: Generating 1:1 Vul-LMGNN Node Embeddings (CodeBERT + Gadget Clean)...")
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        bert_model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
        bert_model.eval()

        assembled_samples = []
        skipped_count = 0
        nodes_dim = DEFAULT_NODES_DIM

        start_vec = time.time()
        with torch.no_grad():
            for g_idx, j_path in enumerate(json_files):
                try:
                    with open(j_path, "r", encoding="utf-8") as jf:
                        g_data = json.load(jf)
                except Exception:
                    skipped_count += 1
                    continue

                nodes_raw = g_data.get("nodes", [])
                edges_raw = g_data.get("edges", [])
                target = int(g_data.get("label", 0))

                fname = j_path.stem
                try:
                    row_id = int(fname.split("_")[0])
                    raw_func_code = idx_to_func.get(row_id, "")
                except Exception:
                    raw_func_code = ""

                filtered_nodes = []
                for n in nodes_raw:
                    n_code = (n.get("code") or "").strip()
                    n_line = n.get("line", -1)
                    n_label = n.get("node_type", "")
                    if n_code and n_line > 0 and n_label not in ["Comment", "Unknown", "COMMENT", "UNKNOWN"]:
                        filtered_nodes.append(n)

                if not filtered_nodes or len(filtered_nodes) > 500:
                    skipped_count += 1
                    continue

                nodes_by_col = sorted(filtered_nodes, key=lambda n: n.get("col", 0) if n.get("col") is not None else 0)
                nodes_by_line = sorted(nodes_by_col, key=lambda n: n.get("line", 0) if n.get("line") is not None else 0)

                ordered_nodes = nodes_by_line[:nodes_dim]

                id_to_order = {n.get("id"): order_idx for order_idx, n in enumerate(ordered_nodes)}

                node_features = []
                all_tokens_batch = []
                tokens_per_node = []

                for node in ordered_nodes:
                    mfn = node.get("mfn", "")
                    label = node.get("node_type", "")
                    if "<operator>" in mfn or "<operators>" in mfn:
                        label = mfn.split(".")[-1]

                    type_id = float(NODE_TYPE_MAP.get(label, UNKNOWN_TYPE_ID))

                    code_str = node.get("code", "")
                    tfn = node.get("tfn", "")
                    if tfn and tfn != "ANY" and tfn != "<empty>" and tfn not in code_str:
                        code_str = f"{tfn} {code_str}"

                    toks = vul_lmgnn_tokenizer(code_str, flag=True)
                    if not toks:
                        toks = [t for t in code_str.split() if t]
                    if not toks:
                        toks = ["<empty>"]

                    tokens_per_node.append((type_id, len(toks)))
                    all_tokens_batch.extend(toks)

                if all_tokens_batch:
                    batch_enc = tokenizer(
                        all_tokens_batch,
                        max_length=64,
                        truncation=True,
                        padding=True,
                        return_tensors="pt"
                    ).to(device)
                    all_cls_feats = bert_model(**batch_enc)[0][:, 0].cpu().numpy()
                else:
                    all_cls_feats = np.zeros((0, 768), dtype=np.float32)

                cursor = 0
                for type_id, count in tokens_per_node:
                    if count > 0 and cursor + count <= len(all_cls_feats):
                        node_tok_feats = all_cls_feats[cursor : cursor + count]
                        mean_emb = np.mean(node_tok_feats, axis=0)
                    else:
                        mean_emb = np.zeros(768, dtype=np.float32)
                    cursor += count

                    full_node_emb = np.concatenate(([type_id], mean_emb), axis=0)
                    node_features.append(full_node_emb)

                x_arr = np.zeros((nodes_dim, DEFAULT_EMB_SIZE), dtype=np.float32)
                if node_features:
                    actual_n = len(node_features)
                    x_arr[:actual_n, :] = np.array(node_features, dtype=np.float32)

                x_tensor = torch.from_numpy(x_arr).float()

                coo_edges = []
                for edge in edges_raw:
                    if len(edge) >= 3:
                        src = edge[0]
                        dst = edge[2]
                        if src in id_to_order and dst in id_to_order:
                            src_ord = id_to_order[src]
                            dst_ord = id_to_order[dst]
                            if src_ord != dst_ord:
                                coo_edges.append([src_ord, dst_ord])

                if not coo_edges:
                    coo_edges = [[0, 0]]

                edge_index = torch.tensor(coo_edges, dtype=torch.long).t().contiguous()

                graph_data = Data(
                    x=x_tensor,
                    edge_index=edge_index,
                    y=torch.tensor([target]).float()
                )
                graph_data.func = raw_func_code

                assembled_samples.append({
                    "input": graph_data,
                    "target": target,
                    "func": raw_func_code
                })

                if (g_idx + 1) % 100 == 0 or (g_idx + 1) == len(json_files):
                    sys.stdout.write(f"\r[Vul-LMGNN] Processed {g_idx + 1:,} / {len(json_files):,} graphs...")
                    sys.stdout.flush()

        print()
        UI.success(
            f"Extraction complete: {len(assembled_samples):,} valid 1:1 graphs built "
            f"in {_fmt_duration(time.time() - start_vec)} (skipped {skipped_count:,})."
        )

        UI.step("Step 6/6: Packaging and serializing into .pkl chunks...")
        samples_df = pd.DataFrame(assembled_samples)
        v_cnt = int((samples_df["target"] == 1).sum())
        c_cnt = int((samples_df["target"] == 0).sum())
        UI.info(f"Assembled dataset: {v_cnt:,} Vulnerable (1) vs {c_cnt:,} Clean (0).")

        dataset_slug = Path(dataset_path).stem.replace("dataset_", "")
        out_chunks_dir = Path(PREPROC_DATASETS_DIR) / "vul_lmgnn" / dataset_slug
        out_chunks_dir.mkdir(parents=True, exist_ok=True)

        num_chunks = math.ceil(len(samples_df) / CHUNK_SIZE)
        for chunk_i in range(num_chunks):
            chunk_slice = samples_df.iloc[chunk_i * CHUNK_SIZE : (chunk_i + 1) * CHUNK_SIZE].copy()
            chunk_filename = f"{chunk_i}_cpg_input.pkl"
            chunk_path = out_chunks_dir / chunk_filename
            chunk_slice.to_pickle(str(chunk_path))

        UI.success(f"Serialized {len(samples_df):,} graphs into {num_chunks} chunks at:\n  {out_chunks_dir}")
        UI.header("Vul-LMGNN 1:1 Pipeline Finished Successfully!")
        UI.success(f"All {num_chunks} chunks saved and ready for training!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LiVDB Vul-LMGNN Preprocessing Pipeline")
    parser.add_argument(
        "--dataset", "-d",
        default=None,
        help="Path to dataset CSV file (default: datasets/linux_kernel_2015-01-01_2026-09-10_ast_curated_balanced1to1.csv)"
    )
    args = parser.parse_args()

    default_csv = os.path.join(_ROOT, "datasets", "linux_kernel_2015-01-01_2026-09-10_ast_curated_balanced1to1.csv")
    ds_path = args.dataset or default_csv

    plugin = VulLMGNNPlugin()
    plugin.run(ds_path)
