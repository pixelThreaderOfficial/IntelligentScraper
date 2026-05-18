import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path


def get_id():
    return uuid.uuid4()


def get_time():
    return datetime.now(ZoneInfo("Asia/Kolkata"))


def get_file_size_mb(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist")

    size_in_bytes = file_path.stat().st_size

    size_in_mb = size_in_bytes / (1024 * 1024)

    return round(size_in_mb, 2)
