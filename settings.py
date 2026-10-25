from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
load_dotenv(ENV_PATH, override=False)


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Обязательная настройка {name} отсутствует в {ENV_PATH}")
    return value


def _integer(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else PROJECT_DIR / value


TELEGRAM_TOKEN = _required("TELEGRAM_TOKEN")
GEMINI_API_KEY = _required("GEMINI_API_KEY")
os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)

GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "imagen-3.0-generate-001")
GEMINI_TEST_MODEL = os.getenv("GEMINI_TEST_MODEL", GEMINI_CHAT_MODEL)

MAX_FILE_SIZE_MB = _integer("MAX_FILE_SIZE_MB", 20)
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_503_RETRIES = _integer("MAX_503_RETRIES", 3)
RETRY_DELAY_SECONDS = _float("RETRY_DELAY_SECONDS", 4.0)
TELEGRAM_POLLING_TIMEOUT = _integer("TELEGRAM_POLLING_TIMEOUT", 20)
TELEGRAM_LONG_POLLING_TIMEOUT = _integer("TELEGRAM_LONG_POLLING_TIMEOUT", 10)

MARKS_DB_PATH = _path("MARKS_DB_PATH", "data/marks.db")
MARKS_STORAGE_DIR = _path("MARKS_STORAGE_DIR", "data/marks_pdfs")
MARKS_SCAN_DPI = _integer("MARKS_SCAN_DPI", 200)
MARKS_GRID_COLUMNS = _integer("MARKS_GRID_COLUMNS", 8)
MARKS_GRID_ROWS = _integer("MARKS_GRID_ROWS", 8)
MARKS_CELL_WIDTH_PT = _float("MARKS_CELL_WIDTH_PT", 136.063)
MARKS_CELL_HEIGHT_PT = _float("MARKS_CELL_HEIGHT_PT", 93.5433)
MARKS_HIGHLIGHT_OPACITY = _float("MARKS_HIGHLIGHT_OPACITY", 0.28)
MARKS_MAX_QUANTITY = _integer("MARKS_MAX_QUANTITY", 100)
