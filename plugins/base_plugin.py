import inspect
from abc import ABC, abstractmethod
from pathlib import Path

from scripts.utils import WORKSPACE_DIR_NAME


class BasePlugin(ABC):
    def __init__(self) -> None:
        plugin_file = inspect.getfile(self.__class__)
        self.plugin_dir = Path(plugin_file).parent
        self.workspace_dir = self.plugin_dir / WORKSPACE_DIR_NAME

        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        gitignore_path = self.workspace_dir / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, "w", encoding="utf-8") as f:
                f.write("*\n!.gitignore\n")

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass

    @abstractmethod
    def run(self, dataset_path: str) -> None:
        pass
