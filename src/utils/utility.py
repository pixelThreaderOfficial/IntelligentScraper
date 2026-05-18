import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

def get_id():
    return uuid.uuid4()

def get_time():
    return datetime.now(ZoneInfo("Asia/Kolkata"))

