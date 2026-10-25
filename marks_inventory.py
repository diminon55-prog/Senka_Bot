from __future__ import annotations

import hashlib
import io
import re
import sqlite3
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
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
class InventoryStats:
    total: int
    available: int
    selected: int
    documents: int


@dataclass(frozen=True)
class ClaimResult:
    document_id: int | None
    matched: int
    total: int
    available: int
    pdf_bytes: bytes
    output_name: str | None


def normalize_code(value: str) -> str:
    normalized = re.sub(r"[\s\x1d()]", "", value).casefold()
    return re.sub(r"^\][a-z]\d", "", normalized)


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
                    status TEXT NOT NULL DEFAULT 'available',
                    selected_at TEXT,
                    UNIQUE(document_id, normalized_code)
                );
                """
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

    def _decode_pdf(self, pdf_bytes: bytes) -> list[dict[str, object]]:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        decoded: list[dict[str, object]] = []
        try:
            for page_index, page in enumerate(document):
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
                decoded.extend(page_marks)
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
        step_x = statistics.median(x_diffs) if x_diffs else self.cell_width_pt
        step_y = statistics.median(y_diffs) if y_diffs else self.cell_height_pt
        grid_width = self.cell_width_pt + max(0, self.grid_columns - 1) * step_x
        grid_height = self.cell_height_pt + max(0, self.grid_rows - 1) * step_y
        origin_x = page_rect.x0 + (page_rect.width - grid_width) / 2
        origin_y = page_rect.y0 + (page_rect.height - grid_height) / 2

        for mark, center_x, center_y in zip(marks, centers_x, centers_y):
            column = min(range(len(columns)), key=lambda index: abs(columns[index] - center_x))
            row = min(range(len(rows)), key=lambda index: abs(rows[index] - center_y))
            mark["x0"] = origin_x + column * step_x
            mark["y0"] = origin_y + row * step_y
            mark["x1"] = float(mark["x0"]) + self.cell_width_pt
            mark["y1"] = float(mark["y0"]) + self.cell_height_pt

    def import_pdf(
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

        marks = self._decode_pdf(pdf_bytes)
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
                    document_id, code, normalized_code, page_index, x0, y0, x1, y1
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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

        return ImportResult(
            document_id=document_id,
            imported=total,
            total=total,
            available=total,
            duplicate=False,
        )

    def claim(self, chat_id: int, query: str, quantity: int = 1) -> ClaimResult:
        normalized_query = normalize_code(query)
        name_query = query.strip().casefold()
        if not name_query:
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

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            document_row = connection.execute(
                """
                SELECT d.*
                FROM documents d
                WHERE d.chat_id = ?
                  AND EXISTS (
                      SELECT 1 FROM marks m
                      WHERE m.document_id = d.id
                        AND m.status = 'available'
                        AND (
                            m.normalized_code = ?
                            OR m.normalized_code LIKE ?
                            OR d.search_name LIKE ?
                        )
                  )
                ORDER BY d.id DESC
                LIMIT 1
                """,
                (
                    chat_id,
                    normalized_query,
                    f"%{normalized_query}%",
                    f"%{name_query}%",
                ),
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
                SELECT m.*
                FROM marks m
                JOIN documents d ON d.id = m.document_id
                WHERE m.document_id = ?
                  AND m.status = 'available'
                  AND (
                      m.normalized_code = ?
                      OR m.normalized_code LIKE ?
                      OR d.search_name LIKE ?
                  )
                ORDER BY m.page_index, m.y0, m.x0
                LIMIT ?
                """,
                (
                    document_id,
                    normalized_query,
                    f"%{normalized_query}%",
                    f"%{name_query}%",
                    quantity,
                ),
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
            connection.executemany(
                "UPDATE marks SET status = 'selected', selected_at = CURRENT_TIMESTAMP WHERE id = ?",
                [(mark_id,) for mark_id in mark_ids],
            )
            counts = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(status = 'available') AS available
                FROM marks WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
            connection.commit()

        source_name = Path(str(document_row["file_name"])).stem
        return ClaimResult(
            document_id=document_id,
            matched=len(rows),
            total=int(counts["total"] or 0),
            available=int(counts["available"] or 0),
            pdf_bytes=output_pdf,
            output_name=f"{source_name}_marked.pdf",
        )

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
