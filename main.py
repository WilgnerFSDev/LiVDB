import importlib
import inspect
import os
from datetime import datetime
from typing import List, Optional

from plugins.base_plugin import BasePlugin
from scripts.cve_fetcher import (
    fetch_all_cves,
    fetch_cves_by_date,
    process_cached_cves,
)
from scripts.curator import curate_dataset
from scripts.dataset_builder import process_cve_dataset
from scripts.tangled_analyzer import run_tangled_analysis
from scripts.ui import UI
from scripts.utils import (
    CACHE_DIR,
    DATASETS_DIR,
    RAW_DATA_DIR,
    clean_workspace,
    get_source_code_path,
    run_git_command,
    get_default_cpe,
)


def _discover_plugins() -> List[BasePlugin]:
    plugins_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins")
    instances: List[BasePlugin] = []

    if not os.path.isdir(plugins_dir):
        return instances

    for entry in sorted(os.listdir(plugins_dir)):
        entry_path = os.path.join(plugins_dir, entry)
        if not (os.path.isdir(entry_path) and entry.startswith("plugin_")):
            continue
        main_plugin_path = os.path.join(entry_path, "main_plugin.py")
        if not os.path.isfile(main_plugin_path):
            UI.warning(f"Plugin directory '{entry}' has no main_plugin.py - skipping.")
            continue
        module_name = f"plugins.{entry}.main_plugin"
        try:
            module = importlib.import_module(module_name)
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BasePlugin) and obj is not BasePlugin:
                    instances.append(obj())
        except Exception as exc:
            UI.warning(f"Could not load plugin '{entry}': {exc}")

    return instances


def _run_plugins_submenu() -> None:
    while True:
        UI.clear()
        UI.print_banner()

        plugins = _discover_plugins()

        if not plugins:
            UI.warning("No plugins found in the 'plugins/' directory.")
            UI.info(
                "Create a subdirectory named 'plugin_*/' with a 'main_plugin.py' that subclasses BasePlugin."
            )
            UI.wait_for_user()
            return

        choices = []
        for idx, plugin in enumerate(plugins, start=1):
            choices.append((str(idx), plugin.get_name(), plugin.get_description()))
        choices.append(("0", "Back", "Return to main menu."))

        try:
            choice = UI.ask_choice("Run Plugins", choices)
        except KeyboardInterrupt:
            return

        if choice == "0":
            return

        try:
            selected = plugins[int(choice) - 1]
        except (IndexError, ValueError):
            UI.error("Invalid selection.")
            continue

        if not os.path.exists(DATASETS_DIR):
            os.makedirs(DATASETS_DIR, exist_ok=True)
        csv_files = sorted([f for f in os.listdir(DATASETS_DIR) if f.endswith(".csv")])

        if not csv_files:
            UI.warning(f"No dataset files found in '{DATASETS_DIR}/' directory.")
            UI.info("Please build a dataset first.")
            UI.wait_for_user()
            continue

        if len(csv_files) == 1:
            dataset_path = os.path.join(DATASETS_DIR, csv_files[0])
            UI.info(f"Using dataset: {csv_files[0]}")
        else:
            csv_choices = [(str(i + 1), f, "") for i, f in enumerate(csv_files)]
            ds_choice = UI.ask_choice("Select the dataset to process:", csv_choices)
            if not ds_choice:
                continue
            dataset_path = os.path.join(DATASETS_DIR, csv_files[int(ds_choice) - 1])

        UI.clear()
        UI.print_banner()
        try:
            selected.run(dataset_path=dataset_path)
        except KeyboardInterrupt:
            UI.warning("\nPlugin execution cancelled by user.")
        UI.wait_for_user()


def configure_nvd_token() -> None:
    UI.header("NVD API Token Configuration")

    current_token = os.environ.get("NVD_API_KEY")
    if current_token:
        masked = f"{current_token[:8]}****..." if len(current_token) > 8 else "****..."
        UI.info(
            f"An NVD API Token is already configured: [bold green]{masked}[/bold green]"
        )
        if not UI.ask_boolean(
            "Do you want to change the existing token?", default=False
        ):
            UI.info("Keeping existing token.")
            return

    UI.info(
        "Obtaining an NVD API Token is strongly recommended to avoid rate limits (from 5 req/min to 50 req/30sec)."
    )
    UI.info(
        "1. Go to: [underline blue]https://nvd.nist.gov/developers/api-key-requested[/underline blue]"
    )
    UI.info("2. Fill out the form and submit.")
    UI.info("3. Check your email for the API token.")
    print()

    UI.info(
        "Tips: To paste, use [bold]Right-Click[/bold] or [bold]CTRL+SHIFT+V[/bold]. Press [bold]CTRL+C[/bold] to cancel."
    )
    token = UI.ask_string("Paste your NVD API Token", default="skip")

    if token and token.lower() != "skip":
        env_path = ".env"
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        with open(env_path, "w", encoding="utf-8") as f:
            updated = False
            for line in lines:
                if line.startswith("NVD_API_KEY="):
                    f.write(f"NVD_API_KEY={token}\n")
                    updated = True
                else:
                    f.write(line)
            if not updated:
                f.write(f"NVD_API_KEY={token}\n")

        os.environ["NVD_API_KEY"] = token
        UI.success("Token successfully saved to .env file!")
    else:
        UI.warning("Skipped token configuration.")


def _save_source_code_path(new_path: str) -> None:
    env_path = ".env"
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    with open(env_path, "w", encoding="utf-8") as f:
        updated = False
        for line in lines:
            if line.startswith("SOURCE_CODE_PATH="):
                f.write(f"SOURCE_CODE_PATH={new_path}\n")
                updated = True
            else:
                f.write(line)
        if not updated:
            f.write(f"SOURCE_CODE_PATH={new_path}\n")

    os.environ["SOURCE_CODE_PATH"] = new_path
    from scripts.utils import get_source_code_path
    get_source_code_path.cache_clear()


def configure_source_path() -> None:
    UI.header("Source Code Repository Configuration")

    current_path = get_source_code_path()
    if current_path and os.path.exists(current_path):
        UI.info(
            f"Source code path is already configured: [bold green]{current_path}[/bold green]"
        )
        if not UI.ask_boolean(
            "Do you want to change the existing path?", default=False
        ):
            UI.info("Keeping existing path.")
            return

    UI.info("Set the path to your local clone of the Source code repository.")
    UI.info(
        "Tips: To paste, use [bold]Right-Click[/bold] or [bold]CTRL+SHIFT+V[/bold]. Press [bold]CTRL+C[/bold] to cancel."
    )

    new_path = UI.ask_string("Enter path to your local Linux repository", default=current_path)

    if new_path and new_path.strip():
        resolved_path = os.path.abspath(os.path.expanduser(new_path.strip()))
        if not os.path.exists(resolved_path):
            UI.warning(f"Warning: Directory '{resolved_path}' does not exist on disk.")
            if not UI.ask_boolean("Do you still want to save this path?", default=False):
                UI.warning("Path configuration cancelled.")
                return
        _save_source_code_path(resolved_path)
        UI.success(f"Path successfully saved to .env: {resolved_path}")
    else:
        UI.warning("Skipped path configuration.")


def configure_default_cpe() -> None:
    from dotenv import set_key
    from scripts.utils import get_default_cpe, get_cpe_search_string

    UI.header("Default CPE (Product) Configuration")

    current_cpe = get_default_cpe()
    UI.info(f"Current Default CPE: [bold green]{current_cpe}[/bold green]")
    if not UI.ask_boolean("Do you want to change the default CPE?", default=False):
        UI.info("Keeping existing CPE.")
        return

    new_cpe = UI.ask_string(
        "Enter new DEFAULT_CPE (e.g., cpe:2.3:o:linux:linux_kernel)", default=current_cpe
    )

    if new_cpe and new_cpe != current_cpe:
        env_path = ".env"
        if not os.path.exists(env_path):
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("")

        set_key(env_path, "DEFAULT_CPE", new_cpe)
        os.environ["DEFAULT_CPE"] = new_cpe

        get_default_cpe.cache_clear()
        get_cpe_search_string.cache_clear()

        UI.success(f"Default CPE successfully updated to: {new_cpe}")
    else:
        UI.warning("Skipped CPE configuration.")


def validate_source_repo() -> bool:
    path = get_source_code_path()

    if not path or not os.path.exists(path):
        if not path:
            UI.warning("Source code repository path is not configured.")
        else:
            UI.error(f"Source code repository path does not exist: {path}")

        configure_source_path()
        path = get_source_code_path()
        if not path or not os.path.exists(path):
            return False

    output = run_git_command(
        ["git", "rev-parse", "--is-inside-work-tree"], path, silent=True
    )
    if output is None:
        UI.error(f"Path is not a valid git repository: {path}")
        configure_source_path()
        path = get_source_code_path()
        if (
            not path
            or run_git_command(
                ["git", "rev-parse", "--is-inside-work-tree"], path, silent=True
            )
            is None
        ):
            return False

    return True


def _has_cached_cves() -> bool:
    if not os.path.exists(CACHE_DIR):
        return False
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
    return len(cache_files) > 0


def _perform_fetch(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> bool:
    if not validate_source_repo():
        return False

    UI.clear()
    UI.print_banner()
    fetch_choice = UI.ask_choice(
        "Choose the fetch strategy:",
        [
            (
                "1",
                "Fetch ALL CVEs (Enterprise Scale - NIST Best Practice)",
                "Synchronize your entire local cache with the latest NVD data (run only on first setup).",
            ),
            (
                "2",
                "Fetch by Date Range",
                "Download only CVEs published within a specific period.",
            ),
            (
                "3",
                "Incremental Update (Recommended for updates)",
                "Fetch only CVEs modified since last sync (run after first setup to get modified CVEs only).",
            ),
            ("0", "Back", "Return to main menu."),
        ],
    )

    if fetch_choice == "0":
        return False

    if fetch_choice == "1":
        should_pull_git = UI.ask_boolean(
            "Run 'git pull' on Source code repository first?", default=True
        )
        fetch_all_cves(should_pull_git)
    elif fetch_choice == "2":
        if not start_date or not end_date:
            current_date = datetime.now()
            default_start = f"{current_date.year}-01-01"
            default_end = current_date.strftime("%Y-%m-%d")

            start_date = UI.ask_string(
                "Enter Start Date (YYYY-MM-DD)", default=default_start
            )
            end_date = UI.ask_string("Enter End Date (YYYY-MM-DD)", default=default_end)

        head_date_str = run_git_command(
            ["git", "log", "-1", "--format=%cd", "--date=short"],
            get_source_code_path(),
            silent=True,
        )
        default_pull = False

        if head_date_str:
            head_date_str = head_date_str.strip()
            try:
                head_date = datetime.strptime(head_date_str, "%Y-%m-%d")
                current_date = datetime.now()
                days_diff = (current_date - head_date).days

                if days_diff > 30:
                    UI.warning(
                        f"Source code repository is {days_diff} days old. Consider running 'git pull'."
                    )
                    default_pull = True
            except ValueError:
                pass

        should_pull_git = UI.ask_boolean(
            "Run 'git pull' on Source code repository first?",
            default=default_pull,
        )
        fetch_cves_by_date(start_date, end_date, should_pull_git)
    elif fetch_choice == "3":
        from scripts.cve_fetcher import fetch_cves_incremental

        should_pull_git = UI.ask_boolean(
            "Run 'git pull' on Source code repository first?", default=True
        )
        fetch_cves_incremental(should_pull_git)

    return True


def run_build_action(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    paired_only: bool = False,
) -> None:
    UI.header("Building Dataset" + (" (Paired Only: 1:1)" if paired_only else ""))

    if not os.path.exists(RAW_DATA_DIR) or not os.listdir(RAW_DATA_DIR):
        UI.warning(f"No extracted CVEs found in '{os.path.basename(RAW_DATA_DIR)}'")

        if _has_cached_cves():
            UI.info("Found cached CVEs available for processing.")
            should_process_cache = UI.ask_boolean(
                "Would you like to process the cached CVEs and extract files?",
                default=True,
            )

            if should_process_cache:
                if not validate_source_repo():
                    return

                should_pull_git = UI.ask_boolean(
                    "Run 'git pull' on Source code repository first?", default=True
                )
                process_cached_cves(should_pull_git, start_date, end_date)

                if not os.path.exists(RAW_DATA_DIR) or not os.listdir(RAW_DATA_DIR):
                    UI.warning(
                        "Cache processing completed, but no files were extracted."
                    )
                    UI.info(
                        "Check if the cached CVEs have valid project references or try fetching new CVEs."
                    )
                    return

        if not os.path.exists(RAW_DATA_DIR) or not os.listdir(RAW_DATA_DIR):
            UI.info("You can fetch new CVEs from the NVD API.")
            should_fetch = UI.ask_boolean(
                "Would you like to fetch CVEs now?", default=True
            )

            if not should_fetch:
                UI.info("Dataset build cancelled.")
                return

            if not _perform_fetch(start_date, end_date):
                UI.info("Fetch was cancelled. Dataset build aborted.")
                return

            if not os.path.exists(RAW_DATA_DIR) or not os.listdir(RAW_DATA_DIR):
                UI.warning("Fetch completed, but no CVEs were retrieved.")
                UI.info("Check your date range or API configuration and try again.")
                return

    cpe = get_default_cpe()
    project_name = "dataset"
    if cpe:
        parts = [p for p in cpe.strip().split(":") if p]
        if parts:
            project_name = parts[-1]

    should_apply_filters = UI.ask_boolean(
        "Apply additional vulnerability filters (CVSS Base Score, CWE, Changed Functions)?",
        default=False,
    )
    min_score = None
    max_score = None
    cwes = None
    single_function_only = False
    max_functions_changed = None

    if should_apply_filters:
        score_choice = UI.ask_choice(
            "Filter by CVSS Base Score:",
            [
                ("0", "Skip score filtering (Default)", "Do not filter by CVSS base score."),
                ("1", "Score above (>= X)", "Include vulnerabilities with CVSS base score above or equal to X."),
                ("2", "Score below (<= X)", "Include vulnerabilities with CVSS base score below or equal to X."),
            ],
        )
        if score_choice == "1":
            val = UI.ask_string(
                "Enter minimum CVSS score (e.g., 7.0, or press Enter to skip):", default=""
            )
            try:
                min_score = float(val) if val else None
            except ValueError:
                UI.warning("Invalid score format. Skipping score filter.")
        elif score_choice == "2":
            val = UI.ask_string(
                "Enter maximum CVSS score (e.g., 5.0, or press Enter to skip):", default=""
            )
            try:
                max_score = float(val) if val else None
            except ValueError:
                UI.warning("Invalid score format. Skipping score filter.")

        cwe_input = UI.ask_string(
            "Enter CWE IDs to filter by (e.g., 615, 123, 354 or CWE-476, or press Enter to skip):",
            default="",
        )
        if cwe_input and cwe_input.strip():
            cwes = [c.strip() for c in cwe_input.replace(";", ",").split(",") if c.strip()]

        entangle_choice = UI.ask_choice(
            "Filter by Function Changes (Tangled Commits):",
            [
                (
                    "0",
                    "All patches (Default - 1 or more functions changed)",
                    "Include all patches regardless of number of functions modified.",
                ),
                (
                    "1",
                    "Non-tangled only (Strictly 1 function changed - ReVeal methodology)",
                    "Only include patches modifying exactly 1 function to eliminate tangled commits.",
                ),
            ],
        )
        if entangle_choice == "1":
            single_function_only = True

    if start_date and end_date:
        default_name = f"{project_name}_{start_date}_{end_date}"
    else:
        default_name = project_name

    if min_score is not None and max_score is not None:
        default_name += f"_cvss_{min_score}_{max_score}"
    elif min_score is not None:
        default_name += f"_cvss_gte_{min_score}"
    elif max_score is not None:
        default_name += f"_cvss_lte_{max_score}"

    if cwes:
        cwe_tags = []
        for c in cwes[:3]:
            tag = c.upper().strip()
            if not tag.startswith("CWE-") and tag.isdigit():
                tag = f"CWE{tag}"
            else:
                tag = tag.replace("-", "")
            cwe_tags.append(tag.lower())
        default_name += f"_{'_'.join(cwe_tags)}"

    if single_function_only:
        default_name += "_non_tangled"

    if paired_only:
        default_name += "_paired"

    hash_choice = UI.ask_choice(
        "Select Function Hash Strategy:",
        [
            (
                "0",
                "Text-based Hash (Default)",
                "Calculates hash using normalized source code with comment stripping.",
            ),
            (
                "1",
                "AST-based Hash (Tree-sitter)",
                "Calculates hash using syntactic AST leaf tokens (ignores comments and extras).",
            ),
        ],
    )
    hash_strategy = "ast" if hash_choice == "1" else "text"

    if hash_strategy == "ast":
        default_name += "_ast"

    dataset_name = UI.ask_string(
        "Enter the name for the output dataset (without .csv)", default=default_name
    )
    if not dataset_name:
        dataset_name = default_name
    if not dataset_name.endswith(".csv"):
        dataset_name += ".csv"

    os.makedirs(DATASETS_DIR, exist_ok=True)
    custom_output_path = os.path.join(DATASETS_DIR, dataset_name)

    UI.clear()
    UI.print_banner()
    UI.step(f"Processing CVE dataset into {dataset_name}...")
    process_cve_dataset(
        RAW_DATA_DIR,
        custom_output_path,
        start_date=start_date,
        end_date=end_date,
        min_score=min_score,
        max_score=max_score,
        cwes=cwes,
        single_function_only=single_function_only,
        hash_strategy=hash_strategy,
        paired_only=paired_only,
    )


def _run_settings_submenu() -> None:
    while True:
        UI.clear()
        UI.print_banner()

        choices = [
            (
                "1",
                "Clean workspace options... (Submenu)",
                "Open cleanup menu to delete temporary files, cache, or datasets.",
            ),
            (
                "2",
                "Configure NVD API Key",
                "Set or update your NIST NVD API Key for faster fetching.",
            ),
            (
                "3",
                "Configure Source Code Path",
                "Set the local path to your Source Code git repository.",
            ),
            (
                "4",
                "Configure Preprocessed Datasets Output Dir",
                "Set the folder where plugins save the preprocessed dataset.",
            ),
            (
                "5",
                "Configure Default CPE (Product)",
                "Set or update the target product CPE for vulnerability matching.",
            ),
            ("0", "Back to Main Menu", "Return to the previous screen."),
        ]

        try:
            choice = UI.ask_choice("Settings and Workspace Management", choices)
        except KeyboardInterrupt:
            break

        if choice == "0":
            break
        elif choice == "1":
            UI.clear()
            UI.print_banner()
            clean_choice = UI.ask_choice(
                "What would you like to clean?",
                [
                    (
                        "1",
                        "Cache only (cache/*.json)",
                        "Wipe the NVD CVE JSON files stored in the local cache.",
                    ),
                    (
                        "2",
                        "Raw Data only (raw_data/*)",
                        "Remove intermediate folders containing extracted vulnerable and patched code.",
                    ),
                    (
                        "3",
                        "Raw Code only (raw_code/*)",
                        "Remove the folder where source code files are downloaded.",
                    ),
                    (
                        "4",
                        "Datasets only (datasets/*.csv)",
                        "Delete all generated dataset files.",
                    ),
                    (
                        "5",
                        "All (Cache, Raw Data, Raw Code, and Datasets)",
                        "Reset everything: purge cache, intermediate data, raw code, and datasets.",
                    ),
                    ("0", "Back", "Return to settings menu."),
                ],
            )

            if clean_choice == "0":
                continue
            elif clean_choice == "1":
                clean_workspace(
                    clean_cache=True,
                    clean_raw=False,
                    clean_datasets=False,
                    clean_raw_code=False,
                )
            elif clean_choice == "2":
                clean_workspace(
                    clean_cache=False,
                    clean_raw=True,
                    clean_datasets=False,
                    clean_raw_code=False,
                )
            elif clean_choice == "3":
                clean_workspace(
                    clean_cache=False,
                    clean_raw=False,
                    clean_datasets=False,
                    clean_raw_code=True,
                )
            elif clean_choice == "4":
                clean_workspace(
                    clean_cache=False,
                    clean_raw=False,
                    clean_datasets=True,
                    clean_raw_code=False,
                )
            elif clean_choice == "5":
                clean_workspace(
                    clean_cache=True,
                    clean_raw=True,
                    clean_datasets=True,
                    clean_raw_code=True,
                )
            UI.success("Workspace cleaned!")
            UI.wait_for_user()

        elif choice == "2":
            configure_nvd_token()
            UI.wait_for_user()

        elif choice == "4":
            import os

            from dotenv import set_key

            from scripts.utils import PREPROC_DATASETS_DIR

            UI.clear()
            UI.print_banner()
            UI.header("Configure Preprocessed Datasets Output Dir")
            UI.info(f"Current Value: {PREPROC_DATASETS_DIR}")

            new_dir = UI.ask_string(
                "Enter new directory path", default=PREPROC_DATASETS_DIR
            )
            if new_dir:
                env_path = ".env"
                if not os.path.exists(env_path):
                    with open(env_path, "w") as f:
                        f.write("")
                set_key(env_path, "PREPROC_DATASETS_DIR", new_dir)
                UI.success(f"PREPROC_DATASETS_DIR updated to: {new_dir}")
                UI.info("Please restart the application for changes to take effect.")
            UI.wait_for_user()

        elif choice == "3":
            configure_source_path()
            UI.wait_for_user()

        elif choice == "5":
            configure_default_cpe()
            UI.wait_for_user()


def check_and_setup_requirements() -> bool:
    import shutil
    import subprocess

    git_available = shutil.which("git") is not None
    joern_available = shutil.which("joern") is not None
    joern_parse_available = shutil.which("joern-parse") is not None

    if not git_available or not joern_available or not joern_parse_available:
        UI.clear()
        UI.print_banner()
        UI.header("Error: System requirements not met")
        UI.error("The following mandatory requirements were not found in PATH:")
        if not git_available:
            UI.error("  - Git is not installed or was not found.")
            UI.info("    To download and install Git, visit: https://git-scm.com/install")
        if not joern_available or not joern_parse_available:
            UI.error("  - Joern (joern and/or joern-parse) is not installed or was not found.")
            UI.info("    To download and install Joern, visit: https://docs.joern.io/installation/")
            UI.info("    Please also ensure you have JDK installed:")
            UI.info("    (JDK 19 (get here for example: https://openjdk.java.net/install/ ); newer JDK versions might work, but have not been properly tested)")
        print()
        UI.wait_for_user()
        return False

    current_path = get_source_code_path()
    if not current_path or not current_path.strip() or not os.path.exists(current_path):
        UI.clear()
        UI.print_banner()
        UI.header("Linux Kernel Repository Configuration")
        if not current_path or not current_path.strip():
            UI.warning("The source code path (SOURCE_CODE_PATH) is not configured.")
        else:
            UI.warning(f"The path configured in SOURCE_CODE_PATH does not exist: {current_path}")

        UI.info("Note: You can skip this now and configure or change it at any time later in Settings (Option 4).")

        if UI.ask_boolean("Would you like to clone the official Linux kernel repository (https://github.com/torvalds/linux.git) now?", default=False):
            default_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "linux"))
            destination_dir = UI.ask_string("Enter destination directory for clone", default=default_dir)
            if not destination_dir:
                destination_dir = default_dir

            destination_dir = os.path.abspath(os.path.expanduser(destination_dir))

            if os.path.isdir(os.path.join(destination_dir, ".git")):
                UI.info(f"A Git repository already exists at '{destination_dir}'.")
                _save_source_code_path(destination_dir)
                UI.success(f"SOURCE_CODE_PATH updated to: {destination_dir}")
                UI.wait_for_user()
                return True

            os.makedirs(destination_dir, exist_ok=True)

            UI.step(f"Cloning Linux Kernel into '{destination_dir}'...")
            UI.info("Note: This process may take a significant amount of time depending on your network connection.")

            try:
                result = subprocess.run(
                    ["git", "clone", "https://github.com/torvalds/linux.git", destination_dir],
                    check=True
                )
                if result.returncode == 0:
                    _save_source_code_path(destination_dir)
                    UI.success(f"Repository successfully cloned to: {destination_dir}")
                    UI.success("SOURCE_CODE_PATH updated.")
                else:
                    UI.error("Failed to clone the repository.")
            except Exception as e:
                UI.error(f"Error during cloning: {e}")

            UI.wait_for_user()
        else:
            if UI.ask_boolean(
                "Do you already have a local clone and want to specify its path now?",
                default=True,
            ):
                configure_source_path()
                UI.wait_for_user()

    current_nvd_key = os.environ.get("NVD_API_KEY")
    if not current_nvd_key or not current_nvd_key.strip():
        UI.clear()
        UI.print_banner()
        UI.header("NVD API Key Configuration")
        UI.warning("NVD API Key is not configured.")
        UI.info("An API key increases NVD request limits (50 req/30s vs 5 req/30s without key).")
        UI.info("Note: You can skip this now and configure or change it at any time later in Settings (Option 4).")
        if UI.ask_boolean("Would you like to configure your NVD API Key now?", default=False):
            configure_nvd_token()
            UI.wait_for_user()

    return True


def main() -> None:
    if not check_and_setup_requirements():
        return

    while True:
        UI.clear()
        UI.print_banner()
        api_key = os.environ.get("NVD_API_KEY")
        if api_key:
            masked = f"{api_key[:8]}****..." if len(api_key) > 8 else "****..."
            UI.info(f"NVD_API_KEY: [bold green]{masked}[/bold green]")
        else:
            UI.warning("NVD_API_KEY: Not configured")

        choices = [
            (
                "1",
                "Fetch CVEs (Enterprise Sync or Date Range)",
                "Retrieve Project vulnerabilities from NVD API using full sync or specific date ranges.",
            ),
            (
                "2",
                "Build dataset.csv (Entire or Date Range)",
                "Extract vulnerable and patched functions from the Source code repository to create the final CSV dataset.",
            ),
            (
                "3",
                "Run pre-processing plugins...",
                "Execute user's plugins for raw dataset pre-processing",
            ),
            (
                "4",
                "Settings and Workspace Management",
                "Clean workspace, configure NVD API token, set Source code path, or preprocessed datasets output.",
            ),
            (
                "5",
                "Analyze Tangled Commits Distribution",
                "Select a dataset to plot distribution of functions changed per CVE in the terminal and export PNG histogram.",
            ),
            ("0", "Exit", "Close the program."),
        ]

        try:
            action_input = UI.ask_choice("Choose action:", choices)
        except KeyboardInterrupt:
            UI.info("\nExiting program (cancelled by user).")
            break

        try:
            if action_input == "1":
                if _perform_fetch():
                    UI.wait_for_user()

            elif action_input == "2":
                UI.clear()
                UI.print_banner()
                build_choice = UI.ask_choice(
                    "Dataset Build Strategy:",
                    [
                        (
                            "1",
                            "Build Entire Database (Full Context - All Functions)",
                            "Process every CVE in local cache (Modified pairs + Unchanged context functions).",
                        ),
                        (
                            "2",
                            "Build Entire Database (Paired Only - 1:1 Vulnerable vs Patched)",
                            "Extract strictly modified function pairs (1:1 Vulnerable vs Patched), excluding unchanged functions.",
                        ),
                        (
                            "3",
                            "Build by Date Range (Full Context)",
                            "Analyze only the CVEs published in a specific time frame (including context functions).",
                        ),
                        (
                            "4",
                            "Build by Date Range (Paired Only - 1:1 Vulnerable vs Patched)",
                            "Analyze only the CVEs published in a specific time frame (strictly 1:1 pairs).",
                        ),
                        (
                            "5",
                            "Curate Existing Dataset (Deduplication & Undersampling)",
                            "Select an existing .csv from datasets/ to remove redundant functions and apply optional undersampling.",
                        ),
                        ("0", "Back", "Return to main menu."),
                    ],
                )
                if build_choice == "0":
                    continue
                elif build_choice == "1":
                    run_build_action(paired_only=False)
                elif build_choice == "2":
                    run_build_action(paired_only=True)
                elif build_choice in ("3", "4"):
                    is_paired = (build_choice == "4")
                    current_date = datetime.now()
                    default_start = f"{current_date.year}-01-01"
                    default_end = current_date.strftime("%Y-%m-%d")

                    start_date = UI.ask_string(
                        "Enter Start Date (YYYY-MM-DD)", default=default_start
                    )
                    end_date = UI.ask_string(
                        "Enter End Date (YYYY-MM-DD)", default=default_end
                    )
                    run_build_action(
                        start_date=start_date,
                        end_date=end_date,
                        paired_only=is_paired,
                    )
                elif build_choice == "5":
                    if not os.path.exists(DATASETS_DIR):
                        os.makedirs(DATASETS_DIR, exist_ok=True)
                    csv_files = sorted(
                        [f for f in os.listdir(DATASETS_DIR) if f.endswith(".csv")]
                    )

                    if not csv_files:
                        UI.warning(
                            f"No dataset files found in '{DATASETS_DIR}/' directory."
                        )
                        UI.wait_for_user()
                        continue

                    csv_choices = [(str(i + 1), f, "") for i, f in enumerate(csv_files)]
                    ds_choice = UI.ask_choice(
                        "Select the dataset to curate:", csv_choices
                    )
                    if ds_choice:
                        selected_path = os.path.join(
                            DATASETS_DIR, csv_files[int(ds_choice) - 1]
                        )
                        curate_dataset(selected_path)

                UI.wait_for_user()

            elif action_input == "3":
                _run_plugins_submenu()

            elif action_input == "4":
                _run_settings_submenu()

            elif action_input == "5":
                if not os.path.exists(DATASETS_DIR):
                    os.makedirs(DATASETS_DIR, exist_ok=True)
                csv_files = sorted(
                    [f for f in os.listdir(DATASETS_DIR) if f.endswith(".csv")]
                )

                if not csv_files:
                    UI.warning(
                        f"No dataset files found in '{DATASETS_DIR}/' directory."
                    )
                    UI.wait_for_user()
                    continue

                csv_choices = [(str(i + 1), f, "") for i, f in enumerate(csv_files)]
                csv_choices.append(("0", "Back", "Return to main menu."))
                ds_choice = UI.ask_choice(
                    "Select the dataset to analyze:", csv_choices
                )
                if ds_choice and ds_choice != "0":
                    selected_path = os.path.join(
                        DATASETS_DIR, csv_files[int(ds_choice) - 1]
                    )
                    run_tangled_analysis(selected_path)
                    UI.wait_for_user()

            elif action_input == "0":
                UI.info("Exiting.")
                break

            else:
                UI.error("Invalid action selected.")

        except KeyboardInterrupt:
            UI.warning("\nAction cancelled by user. Returning to menu...")

        print()


if __name__ == "__main__":
    main()
