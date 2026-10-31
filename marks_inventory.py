from __future__ import annotations

import csv
import hashlib
import io
import re
import secrets
import sqlite3
import statistics
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps
import zxingcpp


@dataclass(frozen=True)
class ImportResult:
    document_id: int
    imported: int
    total: int
    available: int
    duplicate: bool


@dataclass(frozen=True)
class SearchIndexImportResult:
    document_id: int
    imported: int
    unmatched: int
    total_rows: int


@dataclass(frozen=True)
class InventoryStats:
    total: int
    available: int
    selected: int
    documents: int


@dataclass(frozen=True)
class ClaimLocation:
    article: str
    page_number: int
    position_on_page: int
    display_name: str = ""
    document_name: str = ""


@dataclass(frozen=True)
class ClaimResult:
    document_id: int | None
    matched: int
    total: int
    available: int
    pdf_bytes: bytes
    output_name: str | None
    locations: tuple[ClaimLocation, ...] = ()
    claim_token: str = ""


@dataclass(frozen=True)
class AvailableMarkInfo:
    id: int
    article: str
    page_number: int
    position_on_page: int
    display_name: str
    document_id: int


@dataclass(frozen=True)
class SelectedMarkInfo:
    article: str
    page_number: int
    position_on_page: int
    selected_at: str


class ArticleOcrError(RuntimeError):
    pass


def normalize_code(value: str) -> str:
    normalized = re.sub(r"[\s\x1d()]", "", value).casefold()
    return re.sub(r"^\][a-z]\d", "", normalized)


def escape_like(value: str) -> str:
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


class MarksInventory:
    def __init__(
        self,
        db_path: str | Path,
        storage_dir: str | Path,
        scan_dpi: int = 200,
        grid_columns: int = 8,
        grid_rows: int = 8,
        cell_width_pt: float = 136.063,
        cell_height_pt: float = 93.5433,
        highlight_opacity: float = 0.28,
        max_quantity: int = 100,
        ocr_dpi: int = 300,
        tesseract_command: str = "tesseract",
        ocr_timeout_seconds: float = 120.0,
        ocr_max_pages: int = 50,
        ocr_total_timeout_seconds: float = 300.0,
        max_concurrent_ocr: int = 2,
    ) -> None:
        self.db_path = Path(db_path)
        self.storage_dir = Path(storage_dir)
        self.scan_dpi = scan_dpi
        self.grid_columns = grid_columns
        self.grid_rows = grid_rows
        self.cell_width_pt = cell_width_pt
        self.cell_height_pt = cell_height_pt
        self.highlight_opacity = highlight_opacity
        self.max_quantity = max_quantity
        self.ocr_dpi = ocr_dpi
        self.tesseract_command = tesseract_command
        self.ocr_timeout_seconds = ocr_timeout_seconds
        self.ocr_max_pages = max(1, int(ocr_max_pages))
        self.ocr_total_timeout_seconds = max(1.0, float(ocr_total_timeout_seconds))
        self._ocr_semaphore = threading.BoundedSemaphore(
            max(1, int(max_concurrent_ocr))
        )
        self._article_index_locks: dict[int, Any] = {}
        self._article_index_locks_guard = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.chmod(0o700)
        self.storage_dir.chmod(0o700)
        self._initialize_database()
        self.db_path.chmod(0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _article_index_lock(self, chat_id: int) -> Any:
        with self._article_index_locks_guard:
            return self._article_index_locks.setdefault(chat_id, threading.RLock())

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    chat_id INTEGER NOT NULL,
                    file_name TEXT NOT NULL,
                    product_name TEXT NOT NULL,
                    search_name TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    pdf_path TEXT NOT NULL,
                    articles_indexed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(chat_id, sha256)
                );

                CREATE TABLE IF NOT EXISTS marks (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    code TEXT NOT NULL,
                    normalized_code TEXT NOT NULL,
                    page_index INTEGER NOT NULL,
                    x0 REAL NOT NULL,
                    y0 REAL NOT NULL,
                    x1 REAL NOT NULL,
                    y1 REAL NOT NULL,
                    article TEXT NOT NULL DEFAULT '',
                    normalized_article TEXT NOT NULL DEFAULT '',
                    position_on_page INTEGER NOT NULL DEFAULT 0,
                    claim_token TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'available',
                    selected_at TEXT,
                    UNIQUE(document_id, normalized_code)
                );

                CREATE TABLE IF NOT EXISTS article_index_pages (
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page_index INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(document_id, page_index)
                );

                CREATE TABLE IF NOT EXISTS chat_state (
                    chat_id INTEGER PRIMARY KEY,
                    current_document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS mark_text_index (
                    mark_id INTEGER PRIMARY KEY REFERENCES marks(id) ON DELETE CASCADE,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    source_file_name TEXT NOT NULL,
                    global_number INTEGER NOT NULL,
                    set_name TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    row_number INTEGER NOT NULL,
                    column_number INTEGER NOT NULL,
                    display_name TEXT NOT NULL,
                    normalized_display_name TEXT NOT NULL,
                    short_code TEXT NOT NULL,
                    normalized_short_code TEXT NOT NULL,
                    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(document_id, page_number, row_number, column_number)
                );
                """
            )
            document_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(documents)")
            }
            if "articles_indexed" not in document_columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN articles_indexed INTEGER NOT NULL DEFAULT 0"
                )

            mark_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(marks)")
            }
            if "article" not in mark_columns:
                connection.execute(
                    "ALTER TABLE marks ADD COLUMN article TEXT NOT NULL DEFAULT ''"
                )
            if "normalized_article" not in mark_columns:
                connection.execute(
                    "ALTER TABLE marks ADD COLUMN normalized_article TEXT NOT NULL DEFAULT ''"
                )
            if "position_on_page" not in mark_columns:
                connection.execute(
                    "ALTER TABLE marks ADD COLUMN position_on_page INTEGER NOT NULL DEFAULT 0"
                )
            if "claim_token" not in mark_columns:
                connection.execute(
                    "ALTER TABLE marks ADD COLUMN claim_token TEXT NOT NULL DEFAULT ''"
                )

    def _recover_unfinished_claims(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE marks
                SET status = 'available', selected_at = NULL, claim_token = ''
                WHERE claim_token <> ''
                """
            )

    @staticmethod
    def _set_current_document(
        connection: sqlite3.Connection,
        chat_id: int,
        document_id: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO chat_state(chat_id, current_document_id, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chat_id) DO UPDATE SET
                current_document_id = excluded.current_document_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (chat_id, document_id),
        )

    def current_document_id(self, chat_id: int) -> int | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT s.current_document_id
                FROM chat_state s
                JOIN documents d ON d.id = s.current_document_id
                WHERE s.chat_id = ? AND d.chat_id = s.chat_id
                """,
                (chat_id,),
            ).fetchone()
        return int(row["current_document_id"]) if row else None

    def import_search_index_csv(
        self,
        chat_id: int,
        file_name: str,
        csv_bytes: bytes,
    ) -> SearchIndexImportResult:
        required_columns = {"Лист", "Ряд", "Колонка", "Название", "Код"}
        optional_columns = {"Номер", "Набор"}
        try:
            text = csv_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("CSV должен быть сохранён в UTF-8") from error

        extra_fields_key = "__extra_fields__"
        reader = csv.DictReader(
            io.StringIO(text),
            delimiter=";",
            restkey=extra_fields_key,
            restval=None,
        )
        fieldnames = set(reader.fieldnames or ())
        if (
            not required_columns.issubset(fieldnames)
            or fieldnames - required_columns - optional_columns
        ):
            raise ValueError("CSV имеет неверный набор столбцов")

        parsed_rows: list[dict[str, Any]] = []
        seen_locations: set[tuple[int, int, int]] = set()
        for source_index, source_row in enumerate(reader, start=1):
            if source_row.get(extra_fields_key) is not None:
                raise ValueError("CSV содержит строку с лишними полями")
            if any(source_row.get(column) is None for column in required_columns):
                raise ValueError("CSV содержит строку с пропущенными полями")
            try:
                global_number = (
                    int((source_row["Номер"] or "").strip())
                    if "Номер" in fieldnames
                    else source_index
                )
                page_number = int((source_row["Лист"] or "").strip())
                row_number = int((source_row["Ряд"] or "").strip())
                column_number = int((source_row["Колонка"] or "").strip())
            except (TypeError, ValueError) as error:
                raise ValueError("CSV содержит некорректные числовые координаты") from error
            location = (page_number, row_number, column_number)
            if (
                global_number < 1
                or page_number < 1
                or not 1 <= row_number <= self.grid_rows
                or not 1 <= column_number <= self.grid_columns
            ):
                raise ValueError("CSV содержит координаты вне сетки документа")
            if location in seen_locations:
                raise ValueError("CSV содержит повторяющиеся координаты ячеек")
            seen_locations.add(location)
            display_name = (source_row["Название"] or "").strip()
            short_code = (source_row["Код"] or "").strip()
            if not display_name or not short_code:
                raise ValueError("CSV содержит пустое название или код")
            parsed_rows.append(
                {
                    "global_number": global_number,
                    "set_name": (source_row.get("Набор") or "").strip(),
                    "page_number": page_number,
                    "row_number": row_number,
                    "column_number": column_number,
                    "display_name": display_name,
                    "short_code": short_code,
                }
            )

        with self._article_index_lock(chat_id):
            with self._connection() as connection:
                current_row = connection.execute(
                    """
                    SELECT s.current_document_id
                    FROM chat_state s
                    JOIN documents d ON d.id = s.current_document_id
                    WHERE s.chat_id = ? AND d.chat_id = s.chat_id
                    """,
                    (chat_id,),
                ).fetchone()
                document_id = (
                    int(current_row["current_document_id"])
                    if current_row is not None
                    else None
                )
                adopt_current_document = document_id is None
                if adopt_current_document:
                    document_rows = connection.execute(
                        "SELECT id FROM documents WHERE chat_id = ? ORDER BY id",
                        (chat_id,),
                    ).fetchall()
                    if len(document_rows) != 1:
                        raise ValueError(
                            "Не удалось однозначно определить текущий PDF с марками"
                        )
                    document_id = int(document_rows[0]["id"])
                document_row = connection.execute(
                    "SELECT pdf_path FROM documents WHERE id = ? AND chat_id = ?",
                    (document_id, chat_id),
                ).fetchone()
                if document_row is None:
                    raise ValueError("Текущий PDF не найден в базе")
                marks = connection.execute(
                    """
                    SELECT m.id, m.page_index, m.x0, m.y0, m.x1, m.y1
                    FROM marks m
                    JOIN documents d ON d.id = m.document_id
                    WHERE m.document_id = ? AND d.chat_id = ?
                    ORDER BY m.page_index, m.y0, m.x0
                    """,
                    (document_id, chat_id),
                ).fetchall()

                marks_by_page: dict[int, list[dict[str, Any]]] = {}
                for mark in marks:
                    marks_by_page.setdefault(int(mark["page_index"]), []).append(
                        dict(mark)
                    )

                mark_by_location: dict[tuple[int, int, int], dict[str, Any]] = {}
                source_document = pymupdf.open(str(document_row["pdf_path"]))
                try:
                    for page_index, page_marks in marks_by_page.items():
                        if not 0 <= page_index < len(source_document):
                            raise ValueError("В базе указан несуществующий лист PDF")
                        self._expand_to_cell_rectangles(
                            source_document[page_index].rect,
                            page_marks,
                        )
                        for mark in page_marks:
                            position = int(mark["position_on_page"])
                            row_number = (position - 1) // self.grid_columns + 1
                            column_number = (position - 1) % self.grid_columns + 1
                            location = (
                                page_index + 1,
                                row_number,
                                column_number,
                            )
                            if location in mark_by_location:
                                raise ValueError(
                                    "Несколько марок попали в одну ячейку сетки PDF"
                                )
                            mark_by_location[location] = mark
                finally:
                    source_document.close()

                imported = 0
                for row in parsed_rows:
                    location = (
                        int(row["page_number"]),
                        int(row["row_number"]),
                        int(row["column_number"]),
                    )
                    mark = mark_by_location.get(location)
                    if mark is None:
                        continue
                    display_name = str(row["display_name"])
                    short_code = str(row["short_code"])
                    article = self._extract_article(display_name.split())
                    position_on_page = (
                        (int(row["row_number"]) - 1) * self.grid_columns
                        + int(row["column_number"])
                    )
                    connection.execute(
                        """
                        INSERT INTO mark_text_index(
                            mark_id, document_id, source_file_name, global_number,
                            set_name, page_number, row_number, column_number,
                            display_name, normalized_display_name,
                            short_code, normalized_short_code, imported_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(mark_id) DO UPDATE SET
                            source_file_name = excluded.source_file_name,
                            global_number = excluded.global_number,
                            set_name = excluded.set_name,
                            page_number = excluded.page_number,
                            row_number = excluded.row_number,
                            column_number = excluded.column_number,
                            display_name = excluded.display_name,
                            normalized_display_name = excluded.normalized_display_name,
                            short_code = excluded.short_code,
                            normalized_short_code = excluded.normalized_short_code,
                            imported_at = CURRENT_TIMESTAMP
                        """,
                        (
                            int(mark["id"]),
                            document_id,
                            Path(file_name).name,
                            int(row["global_number"]),
                            str(row["set_name"]),
                            int(row["page_number"]),
                            int(row["row_number"]),
                            int(row["column_number"]),
                            display_name,
                            normalize_code(display_name),
                            short_code,
                            normalize_code(short_code),
                        ),
                    )
                    connection.execute(
                        """
                        UPDATE marks
                        SET article = CASE WHEN ? <> '' THEN ? ELSE article END,
                            normalized_article = CASE WHEN ? <> '' THEN ? ELSE normalized_article END,
                            position_on_page = ?
                        WHERE id = ?
                        """,
                        (
                            article,
                            article,
                            article,
                            normalize_code(article),
                            position_on_page,
                            int(mark["id"]),
                        ),
                    )
                    imported += 1

                if adopt_current_document:
                    self._set_current_document(connection, chat_id, document_id)

        return SearchIndexImportResult(
            document_id=document_id,
            imported=imported,
            unmatched=len(parsed_rows) - imported,
            total_rows=len(parsed_rows),
        )

    def looks_like_marks_pdf(self, pdf_bytes: bytes) -> bool:
        try:
            document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            if len(document) == 0:
                document.close()
                return False
            page = document[0]
            pixmap = page.get_pixmap(dpi=self.scan_dpi, alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            result = bool(
                zxingcpp.read_barcodes(
                    image,
                    formats=zxingcpp.BarcodeFormat.DataMatrix,
                    try_rotate=True,
                    try_downscale=True,
                    try_invert=True,
                )
            )
            document.close()
            return result
        except (pymupdf.FileDataError, ValueError):
            return False

    def decode_image(self, image_bytes: bytes) -> list[str]:
        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        except (OSError, ValueError, Image.DecompressionBombError):
            return []

        barcodes = zxingcpp.read_barcodes(
            image,
            formats=zxingcpp.BarcodeFormat.DataMatrix,
            try_rotate=True,
            try_downscale=True,
            try_invert=True,
        )
        codes: list[str] = []
        seen: set[str] = set()
        for barcode in barcodes:
            normalized = normalize_code(barcode.text)
            if normalized and normalized not in seen:
                seen.add(normalized)
                codes.append(barcode.text)
        return codes

    def _ocr_page_words(
        self,
        page: pymupdf.Page,
        timeout_seconds: float | None = None,
    ) -> list[tuple[Any, ...]]:
        with self._ocr_semaphore:
            return self._ocr_page_words_unlocked(page, timeout_seconds)

    def _ocr_page_words_unlocked(
        self,
        page: pymupdf.Page,
        timeout_seconds: float | None = None,
    ) -> list[tuple[Any, ...]]:
        pixmap = page.get_pixmap(dpi=self.ocr_dpi, alpha=False)
        effective_timeout = self.ocr_timeout_seconds
        if timeout_seconds is not None:
            effective_timeout = max(0.1, min(effective_timeout, timeout_seconds))
        try:
            completed = subprocess.run(
                [
                    self.tesseract_command,
                    "stdin",
                    "stdout",
                    "-l",
                    "eng",
                    "--psm",
                    "11",
                    "tsv",
                ],
                input=pixmap.tobytes("png"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=effective_timeout,
            )
        except FileNotFoundError as error:
            raise ArticleOcrError("Tesseract не найден") from error
        except subprocess.TimeoutExpired as error:
            raise ArticleOcrError("Превышено время распознавания артикула") from error
        except (subprocess.SubprocessError, OSError) as error:
            raise ArticleOcrError("Не удалось распознать артикул") from error

        scale_x = page.rect.width / pixmap.width
        scale_y = page.rect.height / pixmap.height
        words: list[tuple[Any, ...]] = []
        lines = completed.stdout.decode("utf-8", errors="replace").splitlines()
        if not lines:
            return words

        headers = lines[0].split("\t")
        for line in lines[1:]:
            values = line.split("\t")
            if len(values) != len(headers):
                continue
            row = dict(zip(headers, values))
            text = row.get("text", "").strip()
            if row.get("level") != "5" or not text:
                continue
            try:
                x0 = int(row["left"]) * scale_x
                y0 = int(row["top"]) * scale_y
                x1 = (int(row["left"]) + int(row["width"])) * scale_x
                y1 = (int(row["top"]) + int(row["height"])) * scale_y
            except (KeyError, ValueError):
                continue
            words.append((x0, y0, x1, y1, text))
        return words

    def _decode_pdf(
        self,
        pdf_bytes: bytes,
        *,
        perform_ocr: bool = True,
        stop_article_query: str = "",
        stop_article_quantity: int = 1,
        page_indexes: set[int] | None = None,
        ocr_time_budget_seconds: float | None = None,
    ) -> list[dict[str, object]]:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        decoded: list[dict[str, object]] = []
        normalized_stop_query = normalize_code(stop_article_query)
        started_at = time.monotonic()
        try:
            for page_index, page in enumerate(document):
                if page_indexes is not None and page_index not in page_indexes:
                    continue
                pixmap = page.get_pixmap(dpi=self.scan_dpi, alpha=False)
                image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                scale_x = page.rect.width / image.width
                scale_y = page.rect.height / image.height
                barcodes = zxingcpp.read_barcodes(
                    image,
                    formats=zxingcpp.BarcodeFormat.DataMatrix,
                    try_rotate=True,
                    try_downscale=True,
                    try_invert=True,
                )
                page_marks: list[dict[str, object]] = []
                for barcode in barcodes:
                    position = barcode.position
                    xs = [
                        position.top_left.x,
                        position.top_right.x,
                        position.bottom_right.x,
                        position.bottom_left.x,
                    ]
                    ys = [
                        position.top_left.y,
                        position.top_right.y,
                        position.bottom_right.y,
                        position.bottom_left.y,
                    ]
                    page_marks.append(
                        {
                            "code": barcode.text,
                            "normalized_code": normalize_code(barcode.text),
                            "page_index": page_index,
                            "x0": min(xs) * scale_x,
                            "y0": min(ys) * scale_y,
                            "x1": max(xs) * scale_x,
                            "y1": max(ys) * scale_y,
                        }
                    )
                self._expand_to_cell_rectangles(page.rect, page_marks)
                page_words = page.get_text("words")
                self._assign_article_metadata(page_marks, page_words)
                if (
                    perform_ocr
                    and page_marks
                    and any(not mark["article"] for mark in page_marks)
                ):
                    remaining_timeout = None
                    if ocr_time_budget_seconds is not None:
                        remaining_timeout = ocr_time_budget_seconds - (
                            time.monotonic() - started_at
                        )
                        if remaining_timeout <= 0:
                            raise ArticleOcrError("Исчерпан общий лимит времени OCR")
                    ocr_words = self._ocr_page_words(page, remaining_timeout)
                    self._assign_article_metadata(page_marks, [*page_words, *ocr_words])
                decoded.extend(page_marks)
                if normalized_stop_query and sum(
                    normalized_stop_query in str(mark["normalized_article"])
                    for mark in decoded
                ) >= max(1, int(stop_article_quantity)):
                    break
        finally:
            document.close()
        return decoded

    @staticmethod
    def _cluster_centers(values: list[float], tolerance: float = 2.0) -> list[float]:
        clusters: list[list[float]] = []
        for value in sorted(values):
            if not clusters or value - statistics.fmean(clusters[-1]) > tolerance:
                clusters.append([value])
            else:
                clusters[-1].append(value)
        return [statistics.fmean(cluster) for cluster in clusters]

    def _expand_to_cell_rectangles(
        self,
        page_rect: pymupdf.Rect,
        marks: list[dict[str, object]],
    ) -> None:
        if not marks:
            return

        centers_x = [(float(mark["x0"]) + float(mark["x1"])) / 2 for mark in marks]
        centers_y = [(float(mark["y0"]) + float(mark["y1"])) / 2 for mark in marks]
        columns = self._cluster_centers(centers_x)
        rows = self._cluster_centers(centers_y)

        x_diffs = [right - left for left, right in zip(columns, columns[1:])]
        y_diffs = [bottom - top for top, bottom in zip(rows, rows[1:])]

        def infer_step(differences: list[float], fallback: float) -> float:
            normalized = [
                difference / max(1, round(difference / fallback))
                for difference in differences
                if difference > 0
            ]
            if not normalized:
                return fallback
            step = statistics.median(normalized)
            return step if fallback * 0.75 <= step <= fallback * 1.35 else fallback

        step_x = infer_step(x_diffs, self.cell_width_pt)
        step_y = infer_step(y_diffs, self.cell_height_pt)
        grid_width = self.cell_width_pt + max(0, self.grid_columns - 1) * step_x
        grid_height = self.cell_height_pt + max(0, self.grid_rows - 1) * step_y
        origin_x = page_rect.x0 + (page_rect.width - grid_width) / 2
        origin_y = page_rect.y0 + (page_rect.height - grid_height) / 2

        for mark, center_x, center_y in zip(marks, centers_x, centers_y):
            column = min(
                range(self.grid_columns),
                key=lambda index: abs(
                    origin_x + index * step_x + self.cell_width_pt / 2 - center_x
                ),
            )
            row = min(
                range(self.grid_rows),
                key=lambda index: abs(
                    origin_y + index * step_y + self.cell_height_pt / 2 - center_y
                ),
            )
            raw_x0 = origin_x + column * step_x
            raw_y0 = origin_y + row * step_y
            mark["x0"] = max(page_rect.x0, raw_x0)
            mark["y0"] = max(page_rect.y0, raw_y0)
            mark["x1"] = min(page_rect.x1, raw_x0 + self.cell_width_pt)
            mark["y1"] = min(page_rect.y1, raw_y0 + self.cell_height_pt)
            mark["position_on_page"] = row * self.grid_columns + column + 1

    @staticmethod
    def _extract_article(words: list[str]) -> str:
        tokens = [word.strip("'‘’\".,;:()[]{}") for word in words]
        tokens = [token for token in tokens if token]
        upper_tokens = [token.upper() for token in tokens]
        model_pattern = re.compile(r"[A-ZА-ЯЁ0-9$]{2,}(?:-[A-ZА-ЯЁ0-9]{2,})+")
        numeric_pattern = re.compile(r"(?<![0-9A-ZА-ЯЁ-])[0-9]+-[0-9]+[A-ZА-ЯЁ](?![0-9A-ZА-ЯЁ-])")
        candidates: list[str] = []
        for index, token in enumerate(upper_tokens):
            combined = token
            if token.endswith("-") and index + 1 < len(upper_tokens):
                combined += upper_tokens[index + 1]
            for candidate in model_pattern.findall(combined):
                if any(character.isalpha() for character in candidate) and any(
                    character.isdigit() for character in candidate
                ):
                    candidates.append(candidate.strip("-"))
        if not candidates:
            for token in upper_tokens:
                candidates.extend(numeric_pattern.findall(token))
        if not candidates:
            return ""

        model = max(candidates, key=lambda value: (value.count("-"), len(value)))
        brand = "COVANI" if "COVANI" in upper_tokens else ""
        return f"{brand} {model}".strip()

    @staticmethod
    def _article_matches(candidate: str, query: str) -> bool:
        if not candidate or not query:
            return False
        variants = {query}
        if query.startswith("covani"):
            variants.add(query.removeprefix("covani"))
        else:
            variants.add(f"covani{query}")
        return any(
            candidate == variant or candidate.startswith(f"{variant}-")
            for variant in variants
            if variant
        )

    def _assign_article_metadata(
        self,
        marks: list[dict[str, object]],
        page_words: list[tuple[Any, ...]],
    ) -> None:
        ordered_marks = sorted(
            marks,
            key=lambda mark: (float(mark["y0"]), float(mark["x0"])),
        )
        for position, mark in enumerate(ordered_marks, start=1):
            words: list[tuple[float, float, str]] = []
            for word in page_words:
                x0, y0, x1, y1 = map(float, word[:4])
                center_x = (x0 + x1) / 2
                center_y = (y0 + y1) / 2
                if (
                    float(mark["x0"]) <= center_x <= float(mark["x1"])
                    and float(mark["y0"]) <= center_y <= float(mark["y1"])
                ):
                    words.append((y0, x0, str(word[4])))
            article = self._extract_article([word for _, _, word in sorted(words)])
            mark["article"] = article
            mark["normalized_article"] = normalize_code(article)
            mark.setdefault("position_on_page", position)

    def import_pdf(
        self,
        chat_id: int,
        file_name: str,
        pdf_bytes: bytes,
        product_name: str | None = None,
    ) -> ImportResult:
        with self._article_index_lock(chat_id):
            return self._import_pdf_locked(
                chat_id,
                file_name,
                pdf_bytes,
                product_name,
            )

    def _import_pdf_locked(
        self,
        chat_id: int,
        file_name: str,
        pdf_bytes: bytes,
        product_name: str | None = None,
    ) -> ImportResult:
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        display_name = (product_name or Path(file_name).stem).strip()

        with self._connection() as connection:
            existing = connection.execute(
                "SELECT id FROM documents WHERE chat_id = ? AND sha256 = ?",
                (chat_id, digest),
            ).fetchone()
            if existing:
                document_id = int(existing["id"])
                self._set_current_document(connection, chat_id, document_id)
                counts = connection.execute(
                    """
                    SELECT COUNT(*) AS total,
                           SUM(status = 'available') AS available
                    FROM marks WHERE document_id = ?
                    """,
                    (document_id,),
                ).fetchone()
                return ImportResult(
                    document_id=document_id,
                    imported=0,
                    total=int(counts["total"] or 0),
                    available=int(counts["available"] or 0),
                    duplicate=True,
                )

        marks = self._decode_pdf(pdf_bytes, perform_ocr=False)
        pdf_path = self.storage_dir / f"{digest}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        pdf_path.chmod(0o600)

        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO documents(chat_id, file_name, product_name, search_name, sha256, pdf_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    file_name,
                    display_name,
                    f"{display_name} {file_name}".casefold(),
                    digest,
                    str(pdf_path),
                ),
            )
            document_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT OR IGNORE INTO marks(
                    document_id, code, normalized_code, page_index, x0, y0, x1, y1,
                    article, normalized_article, position_on_page
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        document_id,
                        mark["code"],
                        mark["normalized_code"],
                        mark["page_index"],
                        mark["x0"],
                        mark["y0"],
                        mark["x1"],
                        mark["y1"],
                        mark["article"],
                        mark["normalized_article"],
                        mark["position_on_page"],
                    )
                    for mark in marks
                ],
            )
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM marks WHERE document_id = ?",
                    (document_id,),
                ).fetchone()[0]
            )
            self._set_current_document(connection, chat_id, document_id)

        return ImportResult(
            document_id=document_id,
            imported=total,
            total=total,
            available=total,
            duplicate=False,
        )

    def reindex_articles(
        self,
        chat_id: int,
        stop_article_query: str = "",
        stop_article_quantity: int = 1,
        document_id: int | None = None,
    ) -> int:
        indexed_marks = 0
        normalized_stop_query = normalize_code(stop_article_query)
        required_quantity = max(1, int(stop_article_quantity))
        with self._article_index_lock(chat_id):
            if (
                normalized_stop_query
                and self._available_article_match_count(
                    chat_id,
                    normalized_stop_query,
                    document_id=document_id,
                ) >= required_quantity
            ):
                return 0
            started_at = time.monotonic()
            processed_pages = 0
            incomplete_pages = False
            with self._connection() as connection:
                if document_id is None:
                    documents = connection.execute(
                        """
                        SELECT id, pdf_path
                        FROM documents
                        WHERE chat_id = ?
                        ORDER BY id DESC
                        """,
                        (chat_id,),
                    ).fetchall()
                else:
                    documents = connection.execute(
                        """
                        SELECT id, pdf_path
                        FROM documents
                        WHERE chat_id = ? AND id = ?
                        """,
                        (chat_id, document_id),
                    ).fetchall()

            for document_row in documents:
                try:
                    pdf_bytes = Path(str(document_row["pdf_path"])).read_bytes()
                    source_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                    try:
                        page_count = len(source_document)
                    finally:
                        source_document.close()
                except (OSError, pymupdf.FileDataError, ValueError):
                    continue

                document_id = int(document_row["id"])
                with self._connection() as connection:
                    indexed_pages = {
                        int(row["page_index"])
                        for row in connection.execute(
                            "SELECT page_index FROM article_index_pages WHERE document_id = ?",
                            (document_id,),
                        )
                    }

                for page_index in range(page_count):
                    if page_index in indexed_pages:
                        continue
                    if processed_pages >= self.ocr_max_pages:
                        raise ArticleOcrError(
                            "Достигнут лимит страниц OCR; повтори запрос для продолжения"
                        )
                    remaining_timeout = self.ocr_total_timeout_seconds - (
                        time.monotonic() - started_at
                    )
                    if remaining_timeout <= 0:
                        raise ArticleOcrError(
                            "Достигнут общий лимит времени OCR; повтори запрос для продолжения"
                        )
                    with self._connection() as connection:
                        expected_codes = {
                            str(row["normalized_code"])
                            for row in connection.execute(
                                """
                                SELECT normalized_code FROM marks
                                WHERE document_id = ? AND page_index = ?
                                """,
                                (document_id, page_index),
                            )
                        }
                    try:
                        marks = self._decode_pdf(
                            pdf_bytes,
                            page_indexes={page_index},
                            ocr_time_budget_seconds=remaining_timeout,
                        )
                    except (OSError, pymupdf.FileDataError, ValueError):
                        break
                    processed_pages += 1

                    with self._connection() as connection:
                        connection.executemany(
                            """
                            UPDATE marks
                            SET position_on_page = ?
                            WHERE document_id = ? AND normalized_code = ?
                            """,
                            [
                                (
                                    mark["position_on_page"],
                                    document_id,
                                    mark["normalized_code"],
                                )
                                for mark in marks
                            ],
                        )
                        connection.executemany(
                            """
                            UPDATE marks
                            SET article = ?, normalized_article = ?
                            WHERE document_id = ? AND normalized_code = ?
                            """,
                            [
                                (
                                    mark["article"],
                                    mark["normalized_article"],
                                    document_id,
                                    mark["normalized_code"],
                                )
                                for mark in marks
                                if mark["article"]
                            ],
                        )
                        stored_articles = {
                            str(row["normalized_code"]): str(row["article"])
                            for row in connection.execute(
                                """
                                SELECT normalized_code, article FROM marks
                                WHERE document_id = ? AND page_index = ?
                                """,
                                (document_id, page_index),
                            )
                        }
                        page_fully_indexed = not expected_codes or all(
                            stored_articles.get(code) for code in expected_codes
                        )
                        incomplete_pages = incomplete_pages or not page_fully_indexed
                        if page_fully_indexed:
                            connection.execute(
                                """
                                INSERT OR IGNORE INTO article_index_pages(document_id, page_index)
                                VALUES (?, ?)
                                """,
                                (document_id, page_index),
                            )
                        indexed_page_count = int(
                            connection.execute(
                                """
                                SELECT COUNT(*) FROM article_index_pages
                                WHERE document_id = ?
                                """,
                                (document_id,),
                            ).fetchone()[0]
                        )
                        if indexed_page_count >= page_count:
                            connection.execute(
                                "UPDATE documents SET articles_indexed = 1 WHERE id = ?",
                                (document_id,),
                            )
                        available_articles = [
                            str(row["normalized_article"])
                            for row in connection.execute(
                                """
                                SELECT normalized_article FROM marks
                                WHERE document_id = ? AND status = 'available'
                                """,
                                (document_id,),
                            )
                        ]

                    if page_fully_indexed:
                        indexed_pages.add(page_index)
                    indexed_marks += sum(bool(mark["article"]) for mark in marks)
                    if normalized_stop_query and sum(
                        self._article_matches(article, normalized_stop_query)
                        for article in available_articles
                    ) >= required_quantity:
                        return indexed_marks
            if incomplete_pages:
                raise ArticleOcrError(
                    "Артикулы распознаны не полностью; повтори запрос для продолжения"
                )
        return indexed_marks

    def _available_article_match_count(
        self,
        chat_id: int,
        query: str,
        document_id: int | None = None,
    ) -> int:
        normalized_query = normalize_code(query)
        if not normalized_query:
            return 0
        with self._connection() as connection:
            articles = connection.execute(
                """
                SELECT m.normalized_article
                FROM marks m
                JOIN documents d ON d.id = m.document_id
                WHERE d.chat_id = ? AND m.status = 'available'
                  AND (? IS NULL OR d.id = ?)
                """,
                (chat_id, document_id, document_id),
            ).fetchall()
        return sum(
            self._article_matches(str(row["normalized_article"]), normalized_query)
            for row in articles
        )

    def _has_complete_text_index(
        self,
        chat_id: int,
        document_id: int | None = None,
    ) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(m.id) AS total,
                       COUNT(i.mark_id) AS indexed
                FROM documents d
                JOIN marks m ON m.document_id = d.id
                LEFT JOIN mark_text_index i ON i.mark_id = m.id
                WHERE d.chat_id = ? AND (? IS NULL OR d.id = ?)
                """,
                (chat_id, document_id, document_id),
            ).fetchone()
        total = int(row["total"] or 0)
        return total > 0 and int(row["indexed"] or 0) == total

    def claim(
        self,
        chat_id: int,
        query: str,
        quantity: int = 1,
        *,
        match_document_name: bool = True,
        article_only: bool = False,
        document_id: int | None = None,
    ) -> ClaimResult:
        normalized_query = normalize_code(query)
        name_query = query.strip().casefold()
        code_pattern = f"%{escape_like(normalized_query)}%"
        name_pattern = f"%{escape_like(name_query)}%"
        if not name_query or not normalized_query:
            stats = self.stats(chat_id)
            return ClaimResult(
                document_id=None,
                matched=0,
                total=stats.total,
                available=stats.available,
                pdf_bytes=b"",
                output_name=None,
            )
        quantity = max(1, min(int(quantity), self.max_quantity))
        article_prefix = f"{escape_like(normalized_query)}-%"
        branded_query = f"covani{normalized_query}"
        branded_prefix = f"{escape_like(branded_query)}-%"
        match_values: tuple[object, ...] = (
            int(article_only),
            normalized_query,
            article_prefix,
            branded_query,
            branded_prefix,
            int(article_only),
            normalized_query,
            code_pattern,
            code_pattern,
            code_pattern,
            code_pattern,
            int(match_document_name),
            name_pattern,
        )

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if document_id is None:
                document_row = connection.execute(
                    """
                    SELECT d.*
                    FROM documents d
                    WHERE d.chat_id = ?
                      AND EXISTS (
                          SELECT 1 FROM marks m
                          LEFT JOIN mark_text_index i ON i.mark_id = m.id
                          WHERE m.document_id = d.id
                            AND m.status = 'available'
                            AND (
                                (? = 1 AND (
                                    m.normalized_article = ?
                                    OR m.normalized_article LIKE ? ESCAPE '!'
                                    OR m.normalized_article = ?
                                    OR m.normalized_article LIKE ? ESCAPE '!'
                                ))
                                OR (? = 0 AND (
                                    m.normalized_code = ?
                                    OR m.normalized_code LIKE ? ESCAPE '!'
                                    OR m.normalized_article LIKE ? ESCAPE '!'
                                    OR i.normalized_display_name LIKE ? ESCAPE '!'
                                    OR i.normalized_short_code LIKE ? ESCAPE '!'
                                    OR (? = 1 AND d.search_name LIKE ? ESCAPE '!')
                                ))
                          )
                      )
                    ORDER BY d.id DESC
                    LIMIT 1
                    """,
                    (chat_id, *match_values),
                ).fetchone()
            else:
                document_row = connection.execute(
                    """
                    SELECT d.*
                    FROM documents d
                    WHERE d.chat_id = ? AND d.id = ?
                      AND EXISTS (
                          SELECT 1 FROM marks m
                          LEFT JOIN mark_text_index i ON i.mark_id = m.id
                          WHERE m.document_id = d.id
                            AND m.status = 'available'
                            AND (
                                (? = 1 AND (
                                    m.normalized_article = ?
                                    OR m.normalized_article LIKE ? ESCAPE '!'
                                    OR m.normalized_article = ?
                                    OR m.normalized_article LIKE ? ESCAPE '!'
                                ))
                                OR (? = 0 AND (
                                    m.normalized_code = ?
                                    OR m.normalized_code LIKE ? ESCAPE '!'
                                    OR m.normalized_article LIKE ? ESCAPE '!'
                                    OR i.normalized_display_name LIKE ? ESCAPE '!'
                                    OR i.normalized_short_code LIKE ? ESCAPE '!'
                                    OR (? = 1 AND d.search_name LIKE ? ESCAPE '!')
                                ))
                          )
                      )
                    """,
                    (chat_id, document_id, *match_values),
                ).fetchone()

            if document_row is None:
                stats = self.stats(chat_id)
                connection.rollback()
                return ClaimResult(
                    document_id=None,
                    matched=0,
                    total=stats.total,
                    available=stats.available,
                    pdf_bytes=b"",
                    output_name=None,
                )

            document_id = int(document_row["id"])
            rows = connection.execute(
                """
                SELECT m.*, i.display_name
                FROM marks m
                JOIN documents d ON d.id = m.document_id
                LEFT JOIN mark_text_index i ON i.mark_id = m.id
                WHERE m.document_id = ?
                  AND m.status = 'available'
                  AND (
                      (? = 1 AND (
                          m.normalized_article = ?
                          OR m.normalized_article LIKE ? ESCAPE '!'
                          OR m.normalized_article = ?
                          OR m.normalized_article LIKE ? ESCAPE '!'
                      ))
                      OR (? = 0 AND (
                          m.normalized_code = ?
                          OR m.normalized_code LIKE ? ESCAPE '!'
                          OR m.normalized_article LIKE ? ESCAPE '!'
                          OR i.normalized_display_name LIKE ? ESCAPE '!'
                          OR i.normalized_short_code LIKE ? ESCAPE '!'
                          OR (? = 1 AND d.search_name LIKE ? ESCAPE '!')
                      ))
                  )
                ORDER BY m.page_index, m.y0, m.x0
                LIMIT ?
                """,
                (document_id, *match_values, quantity),
            ).fetchall()

            source = Path(document_row["pdf_path"]).read_bytes()
            document = pymupdf.open(stream=source, filetype="pdf")
            try:
                for row in rows:
                    rectangle = pymupdf.Rect(row["x0"], row["y0"], row["x1"], row["y1"])
                    document[int(row["page_index"])].draw_rect(
                        rectangle,
                        color=(0.0, 0.65, 0.0),
                        fill=(0.35, 1.0, 0.35),
                        width=2.5,
                        fill_opacity=self.highlight_opacity,
                        stroke_opacity=0.95,
                        overlay=True,
                    )
                output_pdf = document.tobytes(deflate=True, garbage=3)
            finally:
                document.close()

            mark_ids = [int(row["id"]) for row in rows]
            claim_token = secrets.token_urlsafe(24) if mark_ids else ""
            connection.executemany(
                """
                UPDATE marks
                SET status = 'selected', selected_at = CURRENT_TIMESTAMP, claim_token = ?
                WHERE id = ?
                """,
                [(claim_token, mark_id) for mark_id in mark_ids],
            )
            counts = connection.execute(
                """
                SELECT COUNT(m.id) AS total,
                       SUM(m.status = 'available') AS available
                FROM marks m
                JOIN documents d ON d.id = m.document_id
                WHERE d.chat_id = ?
                """,
                (chat_id,),
            ).fetchone()
            connection.commit()

        source_name = Path(str(document_row["file_name"])).stem
        product_name = str(document_row["product_name"] or "")
        return ClaimResult(
            document_id=document_id,
            matched=len(rows),
            total=int(counts["total"] or 0),
            available=int(counts["available"] or 0),
            pdf_bytes=output_pdf,
            output_name=f"{source_name}_marked.pdf",
            locations=tuple(
                ClaimLocation(
                    article=str(row["article"]),
                    page_number=int(row["page_index"]) + 1,
                    position_on_page=int(row["position_on_page"]),
                    display_name=str(row["display_name"] or ""),
                    document_name=product_name or source_name,
                )
                for row in rows
            ),
            claim_token=claim_token,
        )

    def release_claim(self, chat_id: int, claim_token: str) -> int:
        if not claim_token:
            return 0
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE marks
                SET status = 'available', selected_at = NULL, claim_token = ''
                WHERE status = 'selected' AND claim_token = ?
                  AND document_id IN (
                      SELECT id FROM documents WHERE chat_id = ?
                  )
                """,
                (claim_token, chat_id),
            )
        return max(0, int(cursor.rowcount))

    def finalize_claim(self, chat_id: int, claim_token: str) -> int:
        if not claim_token:
            return 0
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE marks
                SET claim_token = ''
                WHERE status = 'selected' AND claim_token = ?
                  AND document_id IN (
                      SELECT id FROM documents WHERE chat_id = ?
                  )
                """,
                (claim_token, chat_id),
            )
        return max(0, int(cursor.rowcount))

    def find_selected_by_article(
        self,
        chat_id: int,
        article: str,
    ) -> list[AvailableMarkInfo]:
        """Находит выбранные марки по артикулу/названию (для интерактивной отмены)."""
        normalized = normalize_code(article)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT m.id, m.article, m.page_index, m.position_on_page,
                       i.display_name, m.document_id
                FROM marks m
                JOIN documents d ON d.id = m.document_id
                LEFT JOIN mark_text_index i ON i.mark_id = m.id
                WHERE d.chat_id = ?
                  AND m.status = 'selected'
                  AND (
                      m.normalized_article = ?
                      OR m.normalized_article LIKE ? ESCAPE '!'
                      OR i.normalized_display_name LIKE ? ESCAPE '!'
                  )
                ORDER BY m.selected_at DESC, m.page_index, m.position_on_page
                """,
                (
                    chat_id,
                    normalized,
                    f"{escape_like(normalized)}-%",
                    f"%{escape_like(normalize_code(article))}%",
                ),
            ).fetchall()
        return [
            AvailableMarkInfo(
                id=int(row["id"]),
                article=str(row["article"] or ""),
                page_number=int(row["page_index"]) + 1,
                position_on_page=int(row["position_on_page"]),
                display_name=str(row["display_name"] or ""),
                document_id=int(row["document_id"]),
            )
            for row in rows
        ]

    def unselect_specific_marks(self, chat_id: int, mark_ids: list[int]) -> int:
        """Возвращает конкретные марки по ID в available. Атомарно."""
        if not mark_ids:
            return 0
        placeholders = ",".join("?" for _ in mark_ids)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                f"""
                UPDATE marks
                SET status = 'available', selected_at = NULL, claim_token = ''
                WHERE id IN ({placeholders})
                  AND status = 'selected'
                  AND document_id IN (SELECT id FROM documents WHERE chat_id = ?)
                """,
                (*mark_ids, chat_id),
            )
            released = cursor.rowcount
            connection.commit()
        return max(0, released)

    def unselect_marks(
        self,
        chat_id: int,
        query: str,
        quantity: int = 1,
    ) -> int:
        normalized_query = normalize_code(query)
        name_query = query.strip().casefold()
        if not name_query or not normalized_query:
            return 0
        quantity = max(1, min(int(quantity), self.max_quantity))
        code_pattern = f"%{escape_like(normalized_query)}%"
        name_pattern = f"%{escape_like(name_query)}%"
        article_prefix = f"{escape_like(normalized_query)}-%"
        branded_query = f"covani{normalized_query}"
        branded_prefix = f"{escape_like(branded_query)}-%"

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT m.id
                FROM marks m
                JOIN documents d ON d.id = m.document_id
                LEFT JOIN mark_text_index i ON i.mark_id = m.id
                WHERE d.chat_id = ?
                  AND m.status = 'selected'
                  AND (
                      m.normalized_code = ?
                      OR m.normalized_code LIKE ? ESCAPE '!'
                      OR m.normalized_article = ?
                      OR m.normalized_article LIKE ? ESCAPE '!'
                      OR m.normalized_article = ?
                      OR m.normalized_article LIKE ? ESCAPE '!'
                      OR i.normalized_display_name LIKE ? ESCAPE '!'
                      OR i.normalized_short_code LIKE ? ESCAPE '!'
                      OR d.search_name LIKE ? ESCAPE '!'
                  )
                ORDER BY m.selected_at DESC, m.id DESC
                LIMIT ?
                """,
                (
                    chat_id,
                    normalized_query,
                    code_pattern,
                    normalized_query,
                    article_prefix,
                    branded_query,
                    branded_prefix,
                    name_pattern,
                    code_pattern,
                    name_pattern,
                    quantity,
                ),
            ).fetchall()
            mark_ids = [int(row["id"]) for row in rows]
            if not mark_ids:
                connection.rollback()
                return 0
            connection.executemany(
                """
                UPDATE marks
                SET status = 'available', selected_at = NULL, claim_token = ''
                WHERE id = ? AND status = 'selected'
                """,
                [(mark_id,) for mark_id in mark_ids],
            )
            released = connection.total_changes
        return released

    def export_selected_marks(self, chat_id: int, delimiter: str = ";") -> bytes | None:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT m.article, m.page_index, m.position_on_page, m.selected_at,
                       i.display_name, i.short_code, d.product_name, d.file_name
                FROM marks m
                JOIN documents d ON d.id = m.document_id
                LEFT JOIN mark_text_index i ON i.mark_id = m.id
                WHERE d.chat_id = ? AND m.status = 'selected'
                ORDER BY m.selected_at, m.page_index, m.position_on_page, m.id
                """,
                (chat_id,),
            ).fetchall()
        if not rows:
            return None
        output = io.StringIO()
        writer = csv.writer(output, delimiter=delimiter, lineterminator="\n")
        writer.writerow(
            [
                "Артикул",
                "Название",
                "Документ",
                "Лист",
                "Позиция на листе",
                "Выбрана",
                "Короткий код",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    str(row["article"] or ""),
                    str(row["display_name"] or ""),
                    str(row["product_name"] or row["file_name"] or ""),
                    int(row["page_index"]) + 1,
                    int(row["position_on_page"]),
                    str(row["selected_at"] or ""),
                    str(row["short_code"] or ""),
                ]
            )
        return output.getvalue().encode("utf-8-sig")

    def suggest_similar(
        self,
        chat_id: int,
        query: str,
        limit: int = 5,
    ) -> list[str]:
        """Возвращает похожие названия свободных марок без выбора."""
        name_query = query.strip().casefold()
        if not name_query:
            return []
        words = [w for w in name_query.split() if len(w) > 1]
        if not words:
            return []
        like_conditions = " OR ".join(
            ["i.normalized_display_name LIKE ? ESCAPE '!'"] * len(words)
        )
        params: list[object] = [chat_id]
        params.extend(f"%{escape_like(word)}%" for word in words)
        params.append(int(limit))
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT DISTINCT i.display_name
                FROM mark_text_index i
                JOIN marks m ON m.id = i.mark_id
                JOIN documents d ON d.id = m.document_id
                WHERE d.chat_id = ?
                  AND m.status = 'available'
                  AND ({like_conditions})
                ORDER BY i.display_name
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [str(row["display_name"]) for row in rows]

    def reset_inventory(self, chat_id: int) -> int:
        """Возвращает все выбранные марки чата в available. Возвращает число сброшенных."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE marks
                SET status = 'available', selected_at = NULL, claim_token = ''
                WHERE status = 'selected'
                  AND document_id IN (
                      SELECT id FROM documents WHERE chat_id = ?
                  )
                """,
                (chat_id,),
            )
        return max(0, int(cursor.rowcount))

    def find_available_by_article(
        self,
        chat_id: int,
        article: str,
    ) -> list[AvailableMarkInfo]:
        """Находит все свободные марки по артикулу (точное совпадение)."""
        normalized = normalize_code(article)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT m.id, m.article, m.page_index, m.position_on_page,
                       i.display_name, m.document_id
                FROM marks m
                JOIN documents d ON d.id = m.document_id
                LEFT JOIN mark_text_index i ON i.mark_id = m.id
                WHERE d.chat_id = ?
                  AND m.status = 'available'
                  AND (
                      m.normalized_article = ?
                      OR i.normalized_display_name LIKE ? ESCAPE '!'
                  )
                ORDER BY m.page_index, m.position_on_page
                """,
                (
                    chat_id,
                    normalized,
                    f"%{escape_like(normalized)}%",
                ),
            ).fetchall()
        return [
            AvailableMarkInfo(
                id=int(row["id"]),
                article=str(row["article"] or ""),
                page_number=int(row["page_index"]) + 1,
                position_on_page=int(row["position_on_page"]),
                display_name=str(row["display_name"] or ""),
                document_id=int(row["document_id"]),
            )
            for row in rows
        ]

    def claim_specific_marks(
        self,
        chat_id: int,
        mark_ids: list[int],
    ) -> ClaimResult:
        """Выбирает конкретные марки по ID."""
        if not mark_ids:
            stats = self.stats(chat_id)
            return ClaimResult(
                document_id=None,
                matched=0,
                total=stats.total,
                available=stats.available,
                pdf_bytes=b"",
                output_name=None,
            )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in mark_ids)
            rows = connection.execute(
                f"""
                SELECT m.*, d.file_name, d.pdf_path, d.product_name, i.display_name
                FROM marks m
                JOIN documents d ON d.id = m.document_id
                LEFT JOIN mark_text_index i ON i.mark_id = m.id
                WHERE d.chat_id = ?
                  AND m.id IN ({placeholders})
                  AND m.status = 'available'
                """,
                (chat_id, *mark_ids),
            ).fetchall()
            if not rows:
                connection.rollback()
                stats = self.stats(chat_id)
                return ClaimResult(
                    document_id=None,
                    matched=0,
                    total=stats.total,
                    available=stats.available,
                    pdf_bytes=b"",
                    output_name=None,
                )

            document_id = int(rows[0]["document_id"])
            source = Path(str(rows[0]["pdf_path"])).read_bytes()
            document = pymupdf.open(stream=source, filetype="pdf")
            try:
                for row in rows:
                    rectangle = pymupdf.Rect(row["x0"], row["y0"], row["x1"], row["y1"])
                    document[int(row["page_index"])].draw_rect(
                        rectangle,
                        color=(0.0, 0.65, 0.0),
                        fill=(0.35, 1.0, 0.35),
                        width=2.5,
                        fill_opacity=self.highlight_opacity,
                        stroke_opacity=0.95,
                        overlay=True,
                    )
                output_pdf = document.tobytes(deflate=True, garbage=3)
            finally:
                document.close()

            claim_token = secrets.token_urlsafe(24)
            connection.executemany(
                """
                UPDATE marks
                SET status = 'selected', selected_at = CURRENT_TIMESTAMP, claim_token = ?
                WHERE id = ? AND status = 'available'
                """,
                [(claim_token, int(row["id"])) for row in rows],
            )
            counts = connection.execute(
                """
                SELECT COUNT(m.id) AS total,
                       SUM(m.status = 'available') AS available
                FROM marks m
                JOIN documents d ON d.id = m.document_id
                WHERE d.chat_id = ?
                """,
                (chat_id,),
            ).fetchone()
            connection.commit()

        source_name = Path(str(rows[0]["file_name"])).stem
        return ClaimResult(
            document_id=document_id,
            matched=len(rows),
            total=int(counts["total"] or 0),
            available=int(counts["available"] or 0),
            pdf_bytes=output_pdf,
            output_name=f"{source_name}_marked.pdf",
            locations=tuple(
                ClaimLocation(
                    article=str(row["article"]),
                    page_number=int(row["page_index"]) + 1,
                    position_on_page=int(row["position_on_page"]),
                    display_name=str(row["display_name"] or ""),
                    document_name=str(
                        row["product_name"] if "product_name" in row.keys() else ""
                    ) or source_name,
                )
                for row in rows
            ),
            claim_token=claim_token,
        )

    def selected_marks_by_query(
        self,
        chat_id: int,
        query: str,
    ) -> list[SelectedMarkInfo]:
        """Находит уже выбранные марки чата по коду, названию или артикулу."""
        normalized_query = normalize_code(query)
        name_query = query.strip().casefold()
        if not name_query or not normalized_query:
            return []
        code_pattern = f"%{escape_like(normalized_query)}%"
        name_pattern = f"%{escape_like(name_query)}%"
        article_prefix = f"{escape_like(normalized_query)}-%"
        branded_query = f"covani{normalized_query}"
        branded_prefix = f"{escape_like(branded_query)}-%"
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT m.article, m.page_index, m.position_on_page, m.selected_at
                FROM marks m
                JOIN documents d ON d.id = m.document_id
                LEFT JOIN mark_text_index i ON i.mark_id = m.id
                WHERE d.chat_id = ?
                  AND m.status = 'selected'
                  AND (
                      m.normalized_code = ?
                      OR m.normalized_code LIKE ? ESCAPE '!'
                      OR m.normalized_article = ?
                      OR m.normalized_article LIKE ? ESCAPE '!'
                      OR m.normalized_article = ?
                      OR m.normalized_article LIKE ? ESCAPE '!'
                      OR i.normalized_display_name LIKE ? ESCAPE '!'
                      OR i.normalized_short_code LIKE ? ESCAPE '!'
                      OR d.search_name LIKE ? ESCAPE '!'
                  )
                ORDER BY m.selected_at DESC
                LIMIT 10
                """,
                (
                    chat_id,
                    normalized_query,
                    code_pattern,
                    normalized_query,
                    article_prefix,
                    branded_query,
                    branded_prefix,
                    name_pattern,
                    code_pattern,
                    name_pattern,
                ),
            ).fetchall()
        return [
            SelectedMarkInfo(
                article=str(row["article"] or ""),
                page_number=int(row["page_index"]) + 1,
                position_on_page=int(row["position_on_page"]),
                selected_at=str(row["selected_at"] or ""),
            )
            for row in rows
        ]

    def claim_current_by_article(
        self,
        chat_id: int,
        article: str,
        quantity: int = 1,
    ) -> ClaimResult:
        with self._article_index_lock(chat_id):
            document_id = self.current_document_id(chat_id)
            if document_id is None:
                stats = self.stats(chat_id)
                return ClaimResult(
                    document_id=None,
                    matched=0,
                    total=stats.total,
                    available=stats.available,
                    pdf_bytes=b"",
                    output_name=None,
                )
            return self.claim_by_article(
                chat_id,
                article,
                quantity,
                document_id=document_id,
            )

    def claim_by_article(
        self,
        chat_id: int,
        article: str,
        quantity: int = 1,
        *,
        document_id: int | None = None,
    ) -> ClaimResult:
        with self._article_index_lock(chat_id):
            return self._claim_by_article_locked(
                chat_id,
                article,
                quantity,
                document_id=document_id,
            )

    def _claim_by_article_locked(
        self,
        chat_id: int,
        article: str,
        quantity: int = 1,
        *,
        document_id: int | None = None,
    ) -> ClaimResult:
        quantity = max(1, min(int(quantity), self.max_quantity))
        model_only = re.sub(r"\bCOVANI\b", "", article, flags=re.IGNORECASE).strip()
        if article.strip() and not self._has_complete_text_index(chat_id, document_id):
            self.reindex_articles(
                chat_id,
                model_only or article,
                stop_article_quantity=quantity,
                document_id=document_id,
            )

        normalized = normalize_code(article)
        with self._connection() as connection:
            distinct = connection.execute(
                """
                SELECT COUNT(DISTINCT m.normalized_article) AS n
                FROM marks m
                JOIN documents d ON d.id = m.document_id
                WHERE d.chat_id = ?
                  AND (? IS NULL OR d.id = ?)
                  AND m.status = 'available'
                  AND (
                      m.normalized_article = ?
                      OR m.normalized_article LIKE ? ESCAPE '!'
                  )
                """,
                (
                    chat_id,
                    document_id,
                    document_id,
                    normalized,
                    f"{escape_like(normalized)}-%",
                ),
            ).fetchone()
        if int(distinct["n"]) > 1:
            stats = self.stats(chat_id)
            return ClaimResult(
                document_id=None,
                matched=0,
                total=stats.total,
                available=stats.available,
                pdf_bytes=b"",
                output_name=None,
            )

        result = self.claim(
            chat_id,
            article,
            quantity,
            match_document_name=False,
            article_only=True,
            document_id=document_id,
        )
        if result.matched:
            return result
        if model_only and model_only.casefold() != article.strip().casefold():
            return self.claim(
                chat_id,
                model_only,
                quantity,
                match_document_name=False,
                article_only=True,
                document_id=document_id,
            )
        return result

    def list_documents(self, chat_id: int) -> list[dict[str, Any]]:
        """Документы чата со счётчиками марок и флагом текущего."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT d.id, d.file_name, d.product_name, d.created_at,
                       COUNT(m.id) AS total,
                       SUM(m.status = 'available') AS available,
                       SUM(m.status = 'selected') AS selected,
                       (s.current_document_id = d.id) AS is_current
                FROM documents d
                LEFT JOIN marks m ON m.document_id = d.id
                LEFT JOIN chat_state s ON s.chat_id = d.chat_id
                WHERE d.chat_id = ?
                GROUP BY d.id
                ORDER BY d.id DESC
                """,
                (chat_id,),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "file_name": str(row["file_name"]),
                "product_name": str(row["product_name"]),
                "created_at": str(row["created_at"] or ""),
                "total": int(row["total"] or 0),
                "available": int(row["available"] or 0),
                "selected": int(row["selected"] or 0),
                "is_current": bool(row["is_current"]),
            }
            for row in rows
        ]

    def set_current_document(self, chat_id: int, document_id: int) -> bool:
        """Делает документ текущим для чата. False, если документ чужой/нет."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id FROM documents WHERE id = ? AND chat_id = ?",
                (document_id, chat_id),
            ).fetchone()
            if row is None:
                return False
            self._set_current_document(connection, chat_id, document_id)
            connection.commit()
        return True

    def stats(self, chat_id: int) -> InventoryStats:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(m.id) AS total,
                       SUM(m.status = 'available') AS available,
                       SUM(m.status = 'selected') AS selected,
                       COUNT(DISTINCT d.id) AS documents
                FROM documents d
                LEFT JOIN marks m ON m.document_id = d.id
                WHERE d.chat_id = ?
                """,
                (chat_id,),
            ).fetchone()
        return InventoryStats(
            total=int(row["total"] or 0),
            available=int(row["available"] or 0),
            selected=int(row["selected"] or 0),
            documents=int(row["documents"] or 0),
        )
