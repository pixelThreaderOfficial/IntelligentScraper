from pathlib import Path
from typing import Callable
import logging

BASE_DIR = Path(__file__).parent.parent.parent

LOG_FILE_PTH = BASE_DIR / "data" / "system_logs.log"


class EventLog:
    def __init__(self):

        LOG_FILE_PTH.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("EventLogger: main")

        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:

            file_handler = logging.FileHandler(LOG_FILE_PTH, encoding="utf-8")

            formatter = logging.Formatter(
                fmt=(
                    "%(log_id)-12s"
                    "%(asctime)-25s"
                    "%(loc)-30s"
                    "%(levelname)-12s"
                    "%(message)s"
                ),
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

        print("Logger started")

    def stop(self):
        print("Logger stopped")

    def _log(self, urgency: str, log_id: str, loc: str, message: str):

        extra_data = {"log_id": log_id, "loc": loc}

        if urgency == "INFO":
            self.logger.info(message, extra=extra_data)

        elif urgency == "WARNING":
            self.logger.warning(message, extra=extra_data)

        elif urgency == "ERROR":
            self.logger.error(message, extra=extra_data)

    def info(self, log_id: str, loc: str, message: str):
        self._log(urgency="INFO", log_id=log_id, loc=loc, message=message)

    def warning(self, log_id: str, loc: str, message: str):
        self._log(urgency="WARNING", log_id=log_id, loc=loc, message=message)

    def error(self, log_id: str, loc: str, message: str):
        self._log(urgency="ERROR", log_id=log_id, loc=loc, message=message)


class ManageFiles:
    def __init__(self, usage: str = "general"):
        self.usage = usage
        self.file = BASE_DIR / "data" / f"{usage}.md"

    def _file_exists(self):
        return self.file.exists()

    def read(self):
        if not self._file_exists():
            return f"Error: File '{self.usage}' does not exist."

        with open(self.file, "r", encoding="utf-8") as f:
            return f.read()

    def write(self, content: str):
        with open(self.file, "w", encoding="utf-8") as f:
            f.write(content)


# ---------------------------------------------------------------------------
# SessionFileManager – generic path-based file operations
# ---------------------------------------------------------------------------
class SessionFileManager:
    """
    Centralises all filesystem manipulation for the scraping pipeline.

    All methods are sync (the orchestrator calls them in its sync context).
    Parent directories are created automatically before writes.
    Errors are raised explicitly – callers decide how to handle them.
    """

    @staticmethod
    def ensure_dir(path: Path) -> None:
        """Create *path* (and parents) if it does not already exist."""
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
        """Write *content* to *path*, creating parent directories first."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)

    @staticmethod
    def read_text(path: Path, encoding: str = "utf-8") -> str:
        """Read and return the full text content of *path*."""
        return path.read_text(encoding=encoding)

    @staticmethod
    def write_scraped_markdown(
        session_folder: Path,
        title_or_host: str,
        source_url: str,
        content: str,
        timestamp: str,
        slugify_fn: Callable[[str], str],
    ) -> Path:
        """
        Build a timestamped markdown file in *session_folder* and write it.

        Returns the Path of the written file.

        Filename pattern: ``<slug>_<timestamp>.md``

        Markdown shape::

            # <title_or_host>

            **Source:** <source_url>

            ---

            <content>
        """
        slug = slugify_fn(title_or_host)
        filename = f"{slug}_{timestamp}.md"
        file_path = session_folder / filename

        md_content = (
            f"# {title_or_host}\n\n"
            f"**Source:** {source_url}\n\n"
            f"---\n\n{content}"
        )

        # Ensure session folder exists (idempotent)
        session_folder.mkdir(parents=True, exist_ok=True)
        file_path.write_text(md_content, encoding="utf-8")
        return file_path