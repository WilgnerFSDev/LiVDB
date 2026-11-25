# Plugin System - Developer Guide

## Overview

The `plugins/` directory contains self-contained pipeline experiments that operate on annotated dataset CSV files located in `datasets/`. The main CLI discovers plugins automatically and presents them in the **"Run Plugins"** submenu.

---

## Data Contract

Every plugin receives a single argument: `dataset_path` - the absolute path to the selected dataset CSV file (from `datasets/`).

### `dataset.csv` Schema

| Column      | Type   | Description                                              |
|-------------|--------|----------------------------------------------------------|
| `cve_id`    | string | CVE identifier, e.g. `CVE-2024-26600`                   |
| `filename`  | string | Source filename inside the kernel repository             |
| `hash`      | string | SHA-256 of the normalized function body                  |
| `lines`     | int    | Line count of the function body                          |
| `label`     | int    | `1` = vulnerable, `0` = patched/clean                    |
| `function`  | string | Full C function source code (may contain newlines)       |

---

## Plugin Interface

Every plugin must subclass `BasePlugin` from `plugins/base_plugin.py` and implement three methods:

```python
from plugins.base_plugin import BasePlugin

class MyPlugin(BasePlugin):
    def get_name(self) -> str:
        return "My Plugin Name"

    def get_description(self) -> str:
        return "One-line description shown in the submenu."

    def run(self, dataset_path: str) -> None:
        ...
```

**Discovery convention:** the file must be named `plugin_<something>.py` and the `BasePlugin` subclass must be the only such class defined directly in the module.

---

## UI Module Reference

Plugins have full access to `scripts/ui.py`. Import it as:

```python
from scripts.ui import UI
```

| Method | Purpose |
|--------|---------|
| `UI.header(text)` | Bold magenta section title |
| `UI.step(text)` | Cyan bullet for a pipeline step |
| `UI.success(text)` | Green checkmark |
| `UI.info(text)` | Dimmed informational message |
| `UI.warning(text)` | Yellow warning |
| `UI.error(text)` | Red error |
| `UI.ask_string(prompt, default)` | Text input prompt |
| `UI.ask_boolean(prompt, default)` | Yes/No prompt |
| `UI.ask_choice(prompt, choices)` | Interactive numbered menu |
| `UI.wait_for_user()` | Pause and wait for Enter |
| `UI.clear()` | Clear the terminal |
| `UI.print_banner()` | Render the LiVDB banner |

### Rich Progress Bars

```python
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
) as progress:
    task = progress.add_task("[cyan]Processing...", total=len(items))
    for item in items:
        # ... do work ...
        progress.advance(task)
```

---

## Discovery Convention

Each plugin lives in its own **subdirectory** inside `plugins/`. The directory name must start with `plugin_` and contain a file called `main_plugin.py`. That file must define exactly one subclass of `BasePlugin`.

```
plugins/
├── base_plugin.py
├── shared/                   # Logic shared across multiple plugins
│   ├── export_to_raw.py
│   └── joern_runner.py
├── plugin_devign/            # Canonical Devign 6-edge CPG + Word2Vec
│   ├── __init__.py
│   ├── main_plugin.py
│   └── joern_devign_extract.sc
├── plugin_reveal/            # Canonical ReVeal 12-edge CPG + Word2Vec
├── plugin_cfvd/              # CFVD / ISSTA'24 PDG + GraphCodeBERT
└── plugin_vul_lmgnn/         # Vul-LMGNN CPG + CodeBERT
```

The CLI discovers plugins automatically at startup - no registration step required.
 
---
 
## Performance: Lazy Loading Imports
 
Because the CLI dynamically probes every `main_plugin.py` to build the menu, any heavy global imports (like `pandas`, `gensim`, `torch`, or `tensorflow`) will severely degrade the CLI startup time.
 
**Rule:** Always load heavy Data Science/ML libraries inside the `run()` method or specific helper functions.
 
### BAD (Top-level import)
```python
from gensim.models import Word2Vec  # Loads on discovery

class MyPlugin(BasePlugin):
    def run(self, dataset_path: str):
        model = Word2Vec(...)
```
 
### GOOD (Lazy loading)
```python
class MyPlugin(BasePlugin):
    def run(self, dataset_path: str):
        from gensim.models import Word2Vec  # Only loads when executed
        model = Word2Vec(...)
```
 
---

## Tutorial: Creating Your Own Plugin

1. Create a new directory: `plugins/plugin_my_research/`
2. Add `__init__.py` (empty).
3. Create `main_plugin.py` using the template below.

```python
"""One-sentence description of what this plugin does."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from plugins.base_plugin import BasePlugin
from scripts.ui import UI


class MyResearchPlugin(BasePlugin):
    def get_name(self) -> str:
        return "My Research Plugin (Experiment X)"

    def get_description(self) -> str:
        return "Brief description visible in the plugin submenu."

    def run(self, dataset_path: str) -> None:
        UI.header("Experiment X Pipeline")

        UI.step("Step 1/N: Loading dataset...")
        import csv, sys as _sys
        try:
            _sys.setrecursionlimit(10000)
            csv.field_size_limit(2147483647)
        except OverflowError:
            pass

        rows = []
        with open(dataset_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        UI.info(f"Loaded {len(rows)} rows from dataset.")

        UI.step("Step 2/N: Your logic here...")

        UI.success("Experiment X complete!")
```

To use shared utilities already available in `plugins/shared/`:

```python
from plugins.shared.export_to_raw import export_to_raw
from plugins.shared.joern_runner import generate_cpg, RAW_CODE_DIR
```

---

## Existing Plugins

| Directory | Description |
|-----------|-------------|
| `plugin_devign/` | Canonical Devign 6-edge CPG (`AST`, `CFG`, `NCS`, `DFG_R`, `DFG_W`, `DFG_C`) + Word2Vec (dim=100). |
| `plugin_reveal/` | Canonical ReVeal 12-edge CPG + 169d node features (AST one-hot + Word2Vec). |
| `plugin_cfvd/` | CFVD / ISSTA'24 line-level PDG + GraphCodeBERT embeddings (768d). |
| `plugin_vul_lmgnn/` | Vul-LMGNN CPG extraction + CodeBERT embeddings (769d). |
