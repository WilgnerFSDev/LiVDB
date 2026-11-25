import concurrent.futures
import functools
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import orjson

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from plugins.base_plugin import BasePlugin
from plugins.shared.export_to_raw import export_to_raw
from plugins.shared.joern_runner import RAW_CODE_DIR, generate_cpg
from scripts.ui import UI
from scripts.utils import PREPROC_DATASETS_DIR

REVEAL_EDGE_MAP: Dict[str, int] = {
    "AST": 1,
    "CFG": 2,
    "CDG": 3,
    "REACHING_DEF": 4,
    "DOMINATE": 5,
    "POST_DOMINATE": 6,
    "ARGUMENT": 7,
    "CALL": 8,
    "RETURN": 9,
    "PARAMETER_LINK": 10,
    "CONTAINS": 11,
    "EVAL_TYPE": 12,
}

EDGE_SYNONYMS: Dict[str, str] = {
    "IS_AST_PARENT": "AST",
    "FLOWS_TO": "CFG",
    "CONTROLS": "CDG",
    "REACHES": "REACHING_DEF",
    "DEF": "REACHING_DEF",
    "USE": "REACHING_DEF",
    "DDG": "REACHING_DEF",
    "DOM": "DOMINATE",
    "POST_DOM": "POST_DOMINATE",
}

REVEAL_TYPE_MAP: Dict[str, int] = {
    'AndExpression': 1, 'Sizeof': 2, 'Identifier': 3, 'ForInit': 4, 'ReturnStatement': 5, 'SizeofOperand': 6,
    'InclusiveOrExpression': 7, 'PtrMemberAccess': 8, 'AssignmentExpression': 9, 'ParameterList': 10,
    'IdentifierDeclType': 11, 'SizeofExpression': 12, 'SwitchStatement': 13, 'IncDec': 14, 'Function': 15,
    'BitAndExpression': 16, 'UnaryExpression': 17, 'DoStatement': 18, 'GotoStatement': 19, 'Callee': 20,
    'OrExpression': 21, 'ShiftExpression': 22, 'Decl': 23, 'CFGErrorNode': 24, 'WhileStatement': 25,
    'InfiniteForNode': 26, 'RelationalExpression': 27, 'CFGExitNode': 28, 'Condition': 29, 'BreakStatement': 30,
    'CompoundStatement': 31, 'UnaryOperator': 32, 'CallExpression': 33, 'CastExpression': 34,
    'ConditionalExpression': 35, 'ArrayIndexing': 36, 'PostIncDecOperationExpression': 37, 'Label': 38,
    'ArgumentList': 39, 'EqualityExpression': 40, 'ReturnType': 41, 'Parameter': 42, 'Argument': 43, 'Symbol': 44,
    'ParameterType': 45, 'Statement': 46, 'AdditiveExpression': 47, 'PrimaryExpression': 48, 'DeclStmt': 49,
    'CastTarget': 50, 'IdentifierDeclStatement': 51, 'IdentifierDecl': 52, 'CFGEntryNode': 53, 'TryStatement': 54,
    'Expression': 55, 'ExclusiveOrExpression': 56, 'ClassDef': 57, 'File': 58, 'UnaryOperationExpression': 59,
    'ClassDefStatement': 60, 'FunctionDef': 61, 'IfStatement': 62, 'MultiplicativeExpression': 63,
    'ContinueStatement': 64, 'MemberAccess': 65, 'ExpressionStatement': 66, 'ForStatement': 67, 'InitializerList': 68,
    'ElseStatement': 69
}

JOERN_TO_REVEAL_TYPE: Dict[str, str] = {
    "CALL": "CallExpression",
    "IDENTIFIER": "Identifier",
    "LITERAL": "PrimaryExpression",
    "METHOD": "FunctionDef",
    "METHOD_RETURN": "ReturnType",
    "METHOD_PARAMETER_IN": "Parameter",
    "PARAM": "Parameter",
    "LOCAL": "IdentifierDecl",
    "BLOCK": "CompoundStatement",
    "RETURN": "ReturnStatement",
    "CONTROL_STRUCTURE": "Condition",
    "TYPE_REF": "IdentifierDeclType",
    "UNKNOWN": "Expression",
    "JUMP_TARGET": "Label",
    "FIELD_IDENTIFIER": "MemberAccess",
    "MODIFIER": "Symbol",
}

AST_TYPE_DIM = 69
W2V_DIM = 100
FEATURE_DIM = 169
W2V_WINDOW = 10
MAX_NODES_LIMIT = 500

JOERN_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "joern_reveal_extract.sc"
)

_TOKEN_RE = re.compile(r'"[^"]*"|[A-Za-z_]\w*|[0-9]+(?:\.[0-9]+)?|\S')


@functools.lru_cache(maxsize=16384)
def concrete_tokenize(code: str) -> str:
    if not code:
        return ""
    code = re.sub(r"//.*?\n|/\*.*?\*/", " ", code, flags=re.S)
    tokens = _TOKEN_RE.findall(code)
    return " ".join([t.strip() for t in tokens if t.strip()])


def _find_jdk11_and_joern() -> Tuple[Optional[Path], Path]:
    system_joern = shutil.which("joern")
    if system_joern:
        return None, Path(system_joern)

    candidate_jdks = [
        Path(_ROOT) / "tools" / "jdk-11",
        Path(_ROOT) / "plugins" / "plugin_cfvd" / "jdk-11",
    ]
    bundled_jdk = None
    for p in candidate_jdks:
        if (p / "bin" / "java").exists():
            bundled_jdk = p
            break

    candidate_joerns = [
        Path(_ROOT) / "tools" / "joern-cli" / "joern",
        Path(_ROOT) / "plugins" / "plugin_cfvd" / "joern-cli" / "joern",
    ]
    joern_bin = None
    for p in candidate_joerns:
        if p.exists() and os.access(str(p), os.X_OK):
            joern_bin = p
            break

    if not joern_bin:
        joern_bin = candidate_joerns[0]

    if bundled_jdk is None or not joern_bin.exists():
        setup_script = Path(_ROOT) / "scripts" / "setup_devign_tools.py"
        if setup_script.exists():
            UI.warning("JDK 11 or Joern 1.1.260 missing. Running automated setup...")
            try:
                subprocess.run([sys.executable, str(setup_script)], check=True)
                for p in candidate_jdks:
                    if (p / "bin" / "java").exists():
                        bundled_jdk = p
                        break
                for p in candidate_joerns:
                    if p.exists():
                        joern_bin = p
                        break
            except Exception as e:
                UI.error(f"Auto-setup failed: {e}. Please run 'python scripts/setup_devign_tools.py' manually.")

    return bundled_jdk, joern_bin


def _run_joern_script(
    cpg_path: str,
    output_dir: str,
    workspace_dir: Optional[str] = None,
    joern_heap_mb: int = 1536,
    min_nodes: int = 1,
    max_nodes: int = MAX_NODES_LIMIT,
    min_edges: int = 1,
) -> Tuple[bool, float, int, int]:
    start_t = time.time()
    cpg_name = os.path.basename(cpg_path)
    cpg_abs = os.path.abspath(cpg_path)
    output_abs = os.path.abspath(output_dir)

    if not os.path.exists(JOERN_SCRIPT_PATH):
        UI.error(f"Joern script not found at {JOERN_SCRIPT_PATH}")
        return False, 0.0, 0, 0

    os.makedirs(output_abs, exist_ok=True)
    bundled_jdk, joern_bin = _find_jdk11_and_joern()

    base_ws = workspace_dir or os.path.dirname(output_abs)
    temp_workspace = os.path.join(base_ws, f"tmp_ws_{cpg_name}")
    amm_home = os.path.join(temp_workspace, ".ammonite_home")
    os.makedirs(amm_home, exist_ok=True)

    env = os.environ.copy()
    if bundled_jdk:
        env["JAVA_HOME"] = str(bundled_jdk)
        env["PATH"] = f"{bundled_jdk / 'bin'}:{env.get('PATH', '')}"

    env["HOME"] = amm_home
    xms = max(256, joern_heap_mb // 4)
    env["JAVA_OPTS"] = f"-Duser.home={amm_home} -Xms{xms}m -Xmx{joern_heap_mb}m"
    env["JOERN_JAVA_OPTS"] = f"-Xms{xms}m -Xmx{joern_heap_mb}m -XX:+UseG1GC --enable-native-access=ALL-UNNAMED"

    is_modern = False
    try:
        check = subprocess.run(
            [str(joern_bin), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        help_out = (check.stdout or "") + (check.stderr or "")
        if "--param <value>" in help_out or "--param " in help_out:
            is_modern = True
    except Exception:
        pass

    cmd = [
        str(joern_bin),
        f"-J-Duser.home={amm_home}",
        f"-J-Xmx{joern_heap_mb}m",
        "--script",
        JOERN_SCRIPT_PATH,
    ]

    if is_modern:
        cmd.extend([
            "--param", f"cpgPath={cpg_abs}",
            "--param", f"outputDir={output_abs}",
            "--param", f"minNodes={min_nodes}",
            "--param", f"maxNodes={max_nodes}",
            "--param", f"minEdges={min_edges}",
        ])
    else:
        cmd.extend([
            "--params",
            f"cpgPath={cpg_abs},outputDir={output_abs},minNodes={min_nodes},maxNodes={max_nodes},minEdges={min_edges}",
        ])

    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=temp_workspace,
        )

        duration = time.time() - start_t
        written = 0
        skipped = 0

        combined = (process.stdout or "") + "\n" + (process.stderr or "")
        for line in combined.splitlines():
            if "[joern_reveal]" in line and "Written=" in line:
                try:
                    w_part = line.split("Written=")[1].split()[0]
                    s_part = line.split("Skipped(filters)=")[1].split()[0]
                    written = int(w_part)
                    skipped = int(s_part)
                except (IndexError, ValueError):
                    pass

        if process.returncode != 0:
            err_msg = (process.stderr or "").strip() or (process.stdout or "").strip()
            err_tail = "\n".join(err_msg.splitlines()[-8:]) if err_msg else "Unknown error"
            UI.error(f"Joern extract failed for {cpg_name}:\n{err_tail}")
            return False, duration, written, skipped

        return True, duration, written, skipped
    except Exception as exc:
        UI.error(f"Execution error on {cpg_name}: {exc}")
        return False, time.time() - start_t, 0, 0
    finally:
        shutil.rmtree(temp_workspace, ignore_errors=True)


class RevealPlugin(BasePlugin):

    def get_name(self) -> str:
        return "ReVeal: Comprehensive Representation Learning (Chakraborty et al. 2020)"

    def get_description(self) -> str:
        return (
            "ReVeal pipeline: CPG extraction with 12 multi-relational edges, "
            "169d node features (69 AST + 100 W2V, window=10), 80/10/10 split, GGNN + Triplet Loss."
        )

    def run(self, dataset_path: str) -> None:
        UI.header("ReVeal Canonical Pipeline (IEEE TSE 2020)")
        UI.info(f"Target Dataset : {dataset_path}")
        UI.info("Specifications : 12 edge types, max_nodes=500, d=169 (Tv=69 + Cv=100), split 80/10/10, GGNN + Triplet Loss")

        try:
            import pandas as pd
            df_labels = pd.read_csv(dataset_path, usecols=["label"])
            n_vuln = int((df_labels["label"] == 1).sum())
            n_clean = int((df_labels["label"] == 0).sum())
            total_funcs = n_vuln + n_clean
            if total_funcs > 0:
                pct_vuln = (n_vuln / total_funcs) * 100
                pct_clean = (n_clean / total_funcs) * 100
                UI.info(
                    f"Class distribution : [bold red]{n_vuln:,} vulnerable ({pct_vuln:.1f}%)[/bold red] vs "
                    f"[bold green]{n_clean:,} clean ({pct_clean:.1f}%)[/bold green]"
                )
                UI.info(
                    "ReVeal Canonical : Trains on natural imbalanced distribution in Phase I, then balances latent space via SMOTE + Triplet Loss in Phase II."
                )
        except Exception:
            pass

        graphs_dir = os.path.join(self.workspace_dir, "graphs")
        joern_output_dir = os.path.join(self.workspace_dir, "joern_output")
        dataset_name = Path(dataset_path).stem
        output_ggnn_dir = os.path.join(PREPROC_DATASETS_DIR, f"{dataset_name}_reveal_canonical")

        os.makedirs(graphs_dir, exist_ok=True)
        os.makedirs(joern_output_dir, exist_ok=True)
        os.makedirs(output_ggnn_dir, exist_ok=True)

        has_existing = (
            any(Path(graphs_dir).glob("*.json"))
            or any(Path(joern_output_dir).glob("*.bin"))
            or any(Path(output_ggnn_dir).glob("*.jsonl"))
        )
        if has_existing:
            UI.warning("Previous generated artifacts detected in the plugin workspace.")
            clear_ans = UI.ask_string(
                "Delete existing generated artifacts and regenerate CPGs from scratch? [y/N]",
                default="N",
            ).strip().lower()

            if clear_ans in ("y", "yes", "s", "sim"):
                UI.step("Cleaning plugin workspace and previous artifacts...")
                shutil.rmtree(graphs_dir, ignore_errors=True)
                shutil.rmtree(joern_output_dir, ignore_errors=True)
                shutil.rmtree(output_ggnn_dir, ignore_errors=True)
                shutil.rmtree(RAW_CODE_DIR, ignore_errors=True)
                os.makedirs(graphs_dir, exist_ok=True)
                os.makedirs(joern_output_dir, exist_ok=True)
                os.makedirs(output_ggnn_dir, exist_ok=True)
                UI.success("Workspace cleaned successfully! Processing will start from scratch.")
            else:
                UI.info("Keeping existing artifacts; already processed steps will be reused.")

        seed_raw = UI.ask_string(
            "Seed for dataset split (80/10/10) and Word2Vec [default: 1000, enter 0 for random/auto]:",
            default="1000",
        ).strip()
        try:
            seed_val = int(seed_raw)
        except ValueError:
            seed_val = 1000

        if seed_val == 0:
            effective_seed = None
            seed_label = "Auto (Random)"
        else:
            effective_seed = seed_val
            seed_label = f"{effective_seed}"

        UI.info(f"Seed configuration: [bold cyan]{seed_label}[/bold cyan]")

        UI.step("Step 1/5: Exporting dataset functions to individual .c files...")
        start_t = time.time()
        export_to_raw(output_dir=RAW_CODE_DIR, dataset_path=dataset_path)
        c_files = [f for f in os.listdir(RAW_CODE_DIR) if f.endswith(".c")]
        UI.success(f"Exported {len(c_files)} .c files in {time.time() - start_t:.1f}s.")

        UI.step("Step 2/5: Building Code Property Graphs (CPGs) via Joern...")
        bundled_jdk, joern_bin = _find_jdk11_and_joern()
        joern_source = "System PATH (Modern Joern)" if shutil.which("joern") else "Bundled / Portable"
        UI.info(f"Joern engine: [bold green]{joern_bin}[/bold green] ({joern_source})")

        import plugins.shared.joern_runner as jr
        jr.CURRENT_OUTPUT_DIR = joern_output_dir
        cpg_ok = generate_cpg(input_dir=RAW_CODE_DIR)
        if not cpg_ok:
            UI.error("Joern CPG generation failed. Aborting pipeline.")
            return

        cpg_files = sorted(
            [
                os.path.join(joern_output_dir, f)
                for f in os.listdir(joern_output_dir)
                if f.endswith(".bin")
            ]
        )
        UI.success(f"Found {len(cpg_files)} CPG batch file(s).")

        UI.step("Step 3/5: Extracting ReVeal canonical multi-relational graphs...")
        existing_graphs = [
            os.path.join(graphs_dir, f)
            for f in os.listdir(graphs_dir)
            if f.endswith(".json")
        ]

        run_extract = True
        if existing_graphs:
            UI.info(f"Found {len(existing_graphs)} existing extracted JSON graph(s).")
            if not UI.ask_boolean("Re-extract graphs from CPGs?", default=False):
                run_extract = False
                UI.success("Using cached graph extractions.")

        if run_extract:
            try:
                import psutil
                free_ram_mb = int(psutil.virtual_memory().available / 1024 / 1024)
            except Exception:
                free_ram_mb = 4096

            joern_heap_mb = 1536
            safe_usable_ram = int(free_ram_mb * 0.60)
            auto_workers = max(1, safe_usable_ram // joern_heap_mb)
            workers = max(1, min(auto_workers, 2))

            total_written, total_skipped = 0, 0
            extract_start = time.time()

            first_cpg = cpg_files[0]
            UI.info(f"Pre-compiling Joern script & processing initial batch ({os.path.basename(first_cpg)})...")
            first_ok, first_dur, first_w, first_s = _run_joern_script(
                first_cpg,
                graphs_dir,
                workspace_dir=self.workspace_dir,
                joern_heap_mb=joern_heap_mb,
                min_nodes=1,
                max_nodes=MAX_NODES_LIMIT,
                min_edges=1,
            )
            if not first_ok:
                UI.error(f"Initial warm-up batch failed for {os.path.basename(first_cpg)}. Aborting.")
                return

            total_written += first_w
            total_skipped += first_s
            UI.success(f"Batch 1/{len(cpg_files)} processed in {first_dur:.1f}s ({first_w} graphs).")

            remaining_cpgs = cpg_files[1:]
            if remaining_cpgs:
                UI.info(f"Extracting remaining {len(remaining_cpgs)} batch(es) with {workers} parallel worker(s)...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            _run_joern_script,
                            cpath,
                            graphs_dir,
                            self.workspace_dir,
                            joern_heap_mb,
                            1,
                            MAX_NODES_LIMIT,
                            1,
                        ): cpath
                        for cpath in remaining_cpgs
                    }

                    for fut in concurrent.futures.as_completed(futures):
                        ok, dur, w, s = fut.result()
                        if ok:
                            total_written += w
                            total_skipped += s
                        else:
                            cpath = futures[fut]
                            UI.error(f"Batch {os.path.basename(cpath)} failed extraction.")

            UI.success(
                f"Extraction complete in {time.time() - extract_start:.1f}s: "
                f"{total_written} graphs exported, {total_skipped} skipped (e.g. >500 nodes)."
            )

        graph_files = sorted(
            [
                os.path.join(graphs_dir, f)
                for f in os.listdir(graphs_dir)
                if f.endswith(".json")
            ]
        )
        if not graph_files:
            UI.error("No graph JSON files found. Aborting.")
            return

        UI.info(f"Total valid function graphs available: {len(graph_files):,}")

        UI.step(f"Step 4/5: Computing dataset split (80% Train, 10% Valid, 10% Test) [Seed: {seed_label}]...")
        if effective_seed is not None:
            random.seed(effective_seed)
        shuffled = list(graph_files)
        random.shuffle(shuffled)

        n_total = len(shuffled)
        train_end = int(n_total * 0.80)
        valid_end = int(n_total * 0.90)

        train_files = set(shuffled[:train_end])
        valid_files = set(shuffled[train_end:valid_end])
        test_files = set(shuffled[valid_end:])

        UI.info(
            f"Split distribution: {len(train_files):,} Train (80%), "
            f"{len(valid_files):,} Validation (10%), {len(test_files):,} Test (10%)"
        )

        UI.step("Step 5/5: Training Word2Vec (window=10) and assembling 169d ReVeal GGNN input records...")

        train_sentences = []
        for gp in train_files:
            try:
                with open(gp, "rb") as f:
                    data = orjson.loads(f.read())
                for node in data.get("nodes", []):
                    toks = concrete_tokenize(node.get("code", "")).split()
                    if toks:
                        train_sentences.append(toks)
            except Exception:
                pass

        UI.info(f"Training Word2Vec on {len(train_sentences):,} node code token sequences (window={W2V_WINDOW})...")
        from gensim.models.callbacks import CallbackAny2Vec
        from gensim.models.word2vec import Word2Vec

        class W2VEpochTracker(CallbackAny2Vec):
            def __init__(self, total_epochs: int):
                self.epoch = 0
                self.total_epochs = total_epochs
                self.start_time = time.time()
                self.epoch_start = time.time()

            def on_epoch_begin(self, model):
                self.epoch += 1
                self.epoch_start = time.time()

            def on_epoch_end(self, model):
                ep_dur = time.time() - self.epoch_start
                tot_elapsed = time.time() - self.start_time
                avg_ep = tot_elapsed / self.epoch
                rem_epochs = self.total_epochs - self.epoch
                eta_s = avg_ep * rem_epochs
                eta_str = f"{int(eta_s // 60)}m {int(eta_s % 60):02d}s" if eta_s >= 60 else f"{eta_s:.1f}s"
                el_str = f"{int(tot_elapsed // 60)}m {int(tot_elapsed % 60):02d}s" if tot_elapsed >= 60 else f"{tot_elapsed:.1f}s"
                UI.info(
                    f"  Word2Vec Epoch [bold cyan]{self.epoch}/{self.total_epochs}[/bold cyan] ✓ "
                    f"[{ep_dur:.1f}s] (Elapsed: {el_str} | ETA: {eta_str})"
                )

        w2v_epochs = 5
        tracker = W2VEpochTracker(total_epochs=w2v_epochs)
        w2v_workers = 1 if effective_seed is not None else (os.cpu_count() or 4)
        w2v_start = time.time()
        w2v = Word2Vec(
            sentences=train_sentences,
            vector_size=W2V_DIM,
            window=W2V_WINDOW,
            min_count=1,
            sg=1,
            negative=5,
            workers=w2v_workers,
            seed=effective_seed if effective_seed is not None else 42,
            epochs=w2v_epochs,
            callbacks=[tracker],
        )
        w2v_path = os.path.join(output_ggnn_dir, "w2v_reveal_100d.model")
        w2v.save(w2v_path)
        UI.success(f"Word2Vec model saved to {w2v_path} (trained in {time.time() - w2v_start:.1f}s).")

        split_manifest_path = os.path.join(output_ggnn_dir, "split_manifest.json")
        with open(split_manifest_path, "w", encoding="utf-8") as smf:
            json.dump({
                "seed": effective_seed if effective_seed is not None else "auto",
                "split_ratio": "80% Train, 10% Validation, 10% Test",
                "train_files": sorted(os.path.basename(f) for f in train_files),
                "valid_files": sorted(os.path.basename(f) for f in valid_files),
                "test_files": sorted(os.path.basename(f) for f in test_files),
            }, smf, indent=2)
        UI.info(f"Split manifest saved to {split_manifest_path}.")

        train_out = os.path.join(output_ggnn_dir, "train_GGNNinput.jsonl")
        valid_out = os.path.join(output_ggnn_dir, "valid_GGNNinput.jsonl")
        test_out  = os.path.join(output_ggnn_dir, "test_GGNNinput.jsonl")

        edge_counter: Dict[str, int] = {k: 0 for k in REVEAL_EDGE_MAP}
        unmapped_edges: Dict[str, int] = {}
        written_counts = {"train": 0, "valid": 0, "test": 0}

        def _get_node_feature(code_str: str, node_type_str: str) -> List[float]:
            type_one_hot = [0.0] * AST_TYPE_DIM
            mapped_type = node_type_str
            if mapped_type not in REVEAL_TYPE_MAP:
                mapped_type = JOERN_TO_REVEAL_TYPE.get(node_type_str.upper(), "Expression")
            type_idx = REVEAL_TYPE_MAP.get(mapped_type, 55) - 1
            if 0 <= type_idx < AST_TYPE_DIM:
                type_one_hot[type_idx] = 1.0

            toks = concrete_tokenize(code_str).split()
            if toks:
                vecs = [w2v.wv[t] for t in toks if t in w2v.wv.key_to_index]
                if vecs:
                    w2v_vec = np.mean(vecs, axis=0).astype(float).tolist()
                else:
                    w2v_vec = [0.0] * W2V_DIM
            else:
                w2v_vec = [0.0] * W2V_DIM

            return type_one_hot + w2v_vec

        total_graphs = len(graph_files)
        UI.info(f"Assembling ReVeal GGNN records (d={FEATURE_DIM}) for {total_graphs:,} graphs...")
        assemble_start = time.time()
        last_log = assemble_start

        with open(train_out, "wb") as f_train, open(valid_out, "wb") as f_valid, open(test_out, "wb") as f_test:
            for idx, gp in enumerate(graph_files, start=1):
                if gp in train_files:
                    target_f = f_train
                    split_key = "train"
                elif gp in valid_files:
                    target_f = f_valid
                    split_key = "valid"
                else:
                    target_f = f_test
                    split_key = "test"

                try:
                    with open(gp, "rb") as f:
                        rec = orjson.loads(f.read())
                except Exception:
                    continue

                nodes = rec.get("nodes", [])
                edges = rec.get("edges", [])
                label = int(rec.get("label", 0))

                node_features = [
                    _get_node_feature(n.get("code", ""), n.get("node_type", ""))
                    for n in nodes
                ]

                graph_edges = []
                for e in edges:
                    if len(e) == 3:
                        src, etype, dst = e
                        etype_str = str(etype).strip().upper()
                        if etype_str in EDGE_SYNONYMS:
                            etype_str = EDGE_SYNONYMS[etype_str]
                        if etype_str in REVEAL_EDGE_MAP:
                            edge_id = REVEAL_EDGE_MAP[etype_str]
                            graph_edges.append([int(src), edge_id, int(dst)])
                            edge_counter[etype_str] += 1
                        else:
                            unmapped_edges[etype_str] = unmapped_edges.get(etype_str, 0) + 1

                entry = orjson.dumps({
                    "node_features": node_features,
                    "graph": graph_edges,
                    "targets": [[label]],
                }) + b"\n"

                target_f.write(entry)
                written_counts[split_key] += 1

                now = time.time()
                if idx % 2500 == 0 or idx == total_graphs or (now - last_log >= 6.0):
                    elapsed = now - assemble_start
                    rate = idx / elapsed if elapsed > 0 else 0
                    rem = total_graphs - idx
                    eta_s = rem / rate if rate > 0 else 0
                    eta_str = f"{int(eta_s // 60)}m {int(eta_s % 60):02d}s" if eta_s >= 60 else f"{eta_s:.0f}s"
                    el_str = f"{int(elapsed // 60)}m {int(elapsed % 60):02d}s" if elapsed >= 60 else f"{elapsed:.0f}s"
                    pct = (idx / total_graphs) * 100
                    UI.info(
                        f"  Assembling records: [bold green]{idx:,}/{total_graphs:,}[/bold green] "
                        f"({pct:5.1f}%) - Elapsed: {el_str} - ETA: {eta_str}"
                    )
                    last_log = now

        UI.success(f"ReVeal GGNN records assembled in {time.time() - assemble_start:.1f}s.")

        meta_path = os.path.join(output_ggnn_dir, "reveal_canonical_metadata.json")
        metadata = {
            "model": "ReVeal (Chakraborty et al., IEEE TSE 2020)",
            "canonical_edges": REVEAL_EDGE_MAP,
            "edge_counts_total": edge_counter,
            "unmapped_edges": unmapped_edges,
            "timesteps_T": 8,
            "hidden_size_z": 200,
            "feature_size_d": FEATURE_DIM,
            "ast_type_dim_Tv": AST_TYPE_DIM,
            "w2v_dim_Cv": W2V_DIM,
            "w2v_window": W2V_WINDOW,
            "node_limit": MAX_NODES_LIMIT,
            "train_samples": written_counts["train"],
            "valid_samples": written_counts["valid"],
            "test_samples": written_counts["test"],
            "split_ratio": "80% Train, 10% Validation, 10% Test",
            "seed": effective_seed if effective_seed is not None else "auto",
            "phase_1_backbone": {
                "model": "GGNNSum",
                "layers": 8,
                "activation": "tanh",
                "pooling": "sum",
                "dropout": 0.2,
                "learning_rate": 0.0001,
                "batch_size": 128,
                "early_stopping_patience": 50,
                "max_iterations": 500,
            },
            "phase_2_representation_learning": {
                "model": "ReVealMLP (256 -> 128 -> 256)",
                "balancing": "SMOTE (k=5)",
                "loss": "Triplet Loss (alpha=0.5, beta=0.001, gamma=0.5)",
                "learning_rate": 0.001,
                "epochs": 100,
                "early_stopping_patience": 5,
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        UI.header("ReVeal Canonical Dataset Created Successfully!")
        UI.success(f"Output directory : {output_ggnn_dir}")
        UI.info(f"Feature size (d) : {FEATURE_DIM} (Tv={AST_TYPE_DIM} AST one-hot + Cv={W2V_DIM} Word2Vec)")
        UI.info(f"Train records    : {written_counts['train']:,} functions (80%)")
        UI.info(f"Valid records    : {written_counts['valid']:,} functions (10%)")
        UI.info(f"Test records     : {written_counts['test']:,} functions (10% held-out)")
        UI.header("Multi-relational Edge Statistics:")
        for etype, count in edge_counter.items():
            UI.info(f"  • {etype:<15} (ID {REVEAL_EDGE_MAP[etype]:2d}): {count:,} edges")

        if unmapped_edges:
            UI.warning(f"Unmapped edge labels observed: {unmapped_edges}")

        UI.header("How to Train ReVeal in training/:")
        UI.info("Step 1 - Train GGNN Backbone (Phase I):")
        UI.info(
            f"  cd training\n"
            f"  python main.py \\\n"
            f"    --config reveal \\\n"
            f"    --model_type ggnn \\\n"
            f"    --dataset {dataset_name} \\\n"
            f"    --input_dir ../preproc_datasets/{dataset_name}_reveal_canonical \\\n"
            f"    --feature_size {FEATURE_DIM} \\\n"
            f"    --graph_embed_size 200 \\\n"
            f"    --batch_size 128"
        )
        UI.info("\nStep 2 - Train Representation Learning MLP with SMOTE + Triplet Loss (Phase II):")
        UI.info(
            f"  python train_reveal.py \\\n"
            f"    --config reveal \\\n"
            f"    --model_type ggnn \\\n"
            f"    --dataset {dataset_name} \\\n"
            f"    --input_dir ../preproc_datasets/{dataset_name}_reveal_canonical \\\n"
            f"    --ggnn_model models/{dataset_name}/GGNNSum-model.bin \\\n"
            f"    --feature_size {FEATURE_DIM} \\\n"
            f"    --graph_embed_size 200 \\\n"
            f"    --batch_size 128"
        )
