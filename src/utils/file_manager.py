from pathlib import Path
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
            return "Error: File '{self.usage}' does not exist."

        with open(self.file, "r", encoding="utf-8") as f:
            return f.read()

    def write(self, content: str):
        with open(self.file, "w", encoding="utf-8") as f:
            f.write(content)
