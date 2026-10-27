import io
import csv
import gc
import sqlite3
import stat
import tempfile
import threading
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf
from PIL import Image
import zxingcpp

from marks_inventory import ArticleOcrError, MarksInventory, normalize_code


def make_datamatrix_png(value: str) -> bytes:
    barcode = zxingcpp.create_barcode(value, zxingcpp.BarcodeFormat.DataMatrix)
    bitmap = zxingcpp.write_barcode_to_image(barcode, scale=5, add_quiet_zones=True)
    height, width = bitmap.shape
    image = Image.frombytes("L", (width, height), bytes(bitmap))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def make_marks_pdf(codes: list[str], articles: list[str] | None = None) -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=400, height=200)
    for index, code in enumerate(codes):
        x0 = 25 + index * 200
        page.insert_image(
            pymupdf.Rect(x0, 25, x0 + 110, 135),
            stream=make_datamatrix_png(code),
        )
        if articles:
            page.insert_text(
                (x0, 160),
                articles[index],
                fontsize=8,
            )
    data = document.tobytes(deflate=True)
    document.close()
    return data


def make_search_index_csv(
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames
        or ["Номер", "Набор", "Лист", "Ряд", "Колонка", "Название", "Код"],
        delimiter=";",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


class MarksInventoryTests(unittest.TestCase):
    def test_csv_search_index_augments_marks_without_replacing_original_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=2,
                grid_rows=1,
                cell_width_pt=180,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                42,
                "marks.pdf",
                make_marks_pdf(["FULL-DATAMATRIX-001", "FULL-DATAMATRIX-002"]),
                "Марки",
            )
            with closing(sqlite3.connect(root / "marks.db")) as connection:
                original_codes = connection.execute(
                    "SELECT code FROM marks ORDER BY x0"
                ).fetchall()
                connection.execute("UPDATE marks SET position_on_page = 0")
                connection.commit()

            import_result = inventory.import_search_index_csv(
                42,
                "marks_index.csv",
                make_search_index_csv(
                    [
                        {
                            "Номер": 1,
                            "Набор": "1–2",
                            "Лист": 1,
                            "Ряд": 1,
                            "Колонка": 1,
                            "Название": "Сабо COVANI OTHER-001 чёрные (36)",
                            "Код": "SHORT-CODE-001",
                        },
                        {
                            "Номер": 2,
                            "Набор": "1–2",
                            "Лист": 1,
                            "Ряд": 1,
                            "Колонка": 2,
                            "Название": "Сабо COVANI SMS25-BHLM3-015H серые (37)",
                            "Код": "SHORT-CODE-002",
                        },
                    ]
                ),
            )

            self.assertEqual(import_result.imported, 2)
            self.assertEqual(import_result.unmatched, 0)
            result = inventory.claim_current_by_article(
                42,
                "COVANI SMS25-BHLM3-015H",
            )
            self.assertEqual(result.matched, 1)
            self.assertEqual(result.locations[0].position_on_page, 2)
            with closing(sqlite3.connect(root / "marks.db")) as connection:
                stored_codes = connection.execute(
                    "SELECT code FROM marks ORDER BY x0"
                ).fetchall()
                text_rows = connection.execute(
                    """
                    SELECT global_number, page_number, row_number, column_number,
                           display_name, short_code
                    FROM mark_text_index ORDER BY global_number
                    """
                ).fetchall()
            self.assertEqual(stored_codes, original_codes)
            self.assertEqual(len(text_rows), 2)
            self.assertEqual(text_rows[1][1:4], (1, 1, 2))
            self.assertIn("SMS25-BHLM3-015H", text_rows[1][4])

    def test_csv_search_index_accepts_the_five_documented_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                42,
                "marks.pdf",
                make_marks_pdf(["FULL-DATAMATRIX-001"]),
                "Марки",
            )

            result = inventory.import_search_index_csv(
                42,
                "minimal.csv",
                make_search_index_csv(
                    [
                        {
                            "Лист": 1,
                            "Ряд": 1,
                            "Колонка": 1,
                            "Название": "Сабо COVANI SMS25-BHLM3-015H серые (37)",
                            "Код": "SHORT-CODE-001",
                        }
                    ],
                    fieldnames=["Лист", "Ряд", "Колонка", "Название", "Код"],
                ),
            )

            self.assertEqual((result.imported, result.unmatched), (1, 0))
            with closing(sqlite3.connect(root / "marks.db")) as connection:
                metadata = connection.execute(
                    "SELECT global_number, set_name FROM mark_text_index"
                ).fetchone()
            self.assertEqual(metadata, (1, ""))

    def test_csv_search_index_preserves_sparse_physical_columns(self):
        document = pymupdf.open()
        page = document.new_page(width=450, height=200)
        for x0, code in (
            (30, "FULL-DATAMATRIX-LEFT"),
            (310, "FULL-DATAMATRIX-RIGHT"),
        ):
            page.insert_image(
                pymupdf.Rect(x0, 25, x0 + 100, 125),
                stream=make_datamatrix_png(code),
            )
        pdf_bytes = document.tobytes(deflate=True)
        document.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=3,
                grid_rows=1,
                cell_width_pt=130,
                cell_height_pt=180,
            )
            inventory.import_pdf(42, "sparse.pdf", pdf_bytes, "Марки")

            result = inventory.import_search_index_csv(
                42,
                "sparse_index.csv",
                make_search_index_csv(
                    [
                        {
                            "Номер": 1,
                            "Набор": "1–2",
                            "Лист": 1,
                            "Ряд": 1,
                            "Колонка": 1,
                            "Название": "COVANI LEFT-001",
                            "Код": "SHORT-LEFT-001",
                        },
                        {
                            "Номер": 2,
                            "Набор": "1–2",
                            "Лист": 1,
                            "Ряд": 1,
                            "Колонка": 3,
                            "Название": "COVANI RIGHT-003",
                            "Код": "SHORT-RIGHT-003",
                        },
                    ]
                ),
            )

            self.assertEqual((result.imported, result.unmatched), (2, 0))
            with closing(sqlite3.connect(root / "marks.db")) as connection:
                linked_columns = connection.execute(
                    """
                    SELECT i.column_number, m.position_on_page
                    FROM mark_text_index i
                    JOIN marks m ON m.id = i.mark_id
                    ORDER BY m.x0
                    """
                ).fetchall()
            self.assertEqual(linked_columns, [(1, 1), (3, 3)])

    def test_csv_search_index_rejects_rows_with_extra_fields_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                42,
                "marks.pdf",
                make_marks_pdf(["FULL-DATAMATRIX-001"]),
                "Марки",
            )
            malformed_csv = (
                "Номер;Набор;Лист;Ряд;Колонка;Название;Код\n"
                "1;1;1;1;1;COVANI MODEL-001;НЕЭКРАНИРОВАННАЯ-ЧАСТЬ;SHORT-CODE-001\n"
            ).encode("utf-8")

            with self.assertRaisesRegex(ValueError, "пол"):
                inventory.import_search_index_csv(
                    42,
                    "malformed.csv",
                    malformed_csv,
                )

            with closing(sqlite3.connect(root / "marks.db")) as connection:
                stored_rows = connection.execute(
                    "SELECT COUNT(*) FROM mark_text_index"
                ).fetchone()[0]
            self.assertEqual(stored_rows, 0)

    def test_csv_search_index_adopts_the_only_legacy_document_as_current(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            imported_pdf = inventory.import_pdf(
                42,
                "legacy.pdf",
                make_marks_pdf(["FULL-DATAMATRIX-001"]),
                "Марки",
            )
            with closing(sqlite3.connect(root / "marks.db")) as connection:
                connection.execute("DELETE FROM chat_state")
                connection.commit()

            result = inventory.import_search_index_csv(
                42,
                "legacy_index.csv",
                make_search_index_csv(
                    [
                        {
                            "Номер": 1,
                            "Набор": "1",
                            "Лист": 1,
                            "Ряд": 1,
                            "Колонка": 1,
                            "Название": "Сабо COVANI SMS25-BHLM3-015H серые (37)",
                            "Код": "SHORT-CODE-001",
                        }
                    ]
                ),
            )

            self.assertEqual(result.document_id, imported_pdf.document_id)
            self.assertEqual(inventory.current_document_id(42), imported_pdf.document_id)

    def test_failed_csv_import_does_not_adopt_a_legacy_current_document(self):
        class FailingMappingInventory(MarksInventory):
            def _expand_to_cell_rectangles(self, *args, **kwargs):
                raise RuntimeError("forced mapping failure")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "marks.db"
            storage_dir = root / "pdfs"
            inventory = MarksInventory(
                database_path,
                storage_dir,
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                42,
                "legacy.pdf",
                make_marks_pdf(["FULL-DATAMATRIX-001"]),
                "Марки",
            )
            with closing(sqlite3.connect(database_path)) as connection:
                connection.execute("DELETE FROM chat_state")
                connection.commit()

            failing = FailingMappingInventory(
                database_path,
                storage_dir,
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            with self.assertRaisesRegex(RuntimeError, "forced mapping failure"):
                failing.import_search_index_csv(
                    42,
                    "legacy_index.csv",
                    make_search_index_csv(
                        [
                            {
                                "Номер": 1,
                                "Набор": "1",
                                "Лист": 1,
                                "Ряд": 1,
                                "Колонка": 1,
                                "Название": "COVANI SMS25-BHLM3-015H",
                                "Код": "SHORT-CODE-001",
                            }
                        ]
                    ),
                )

            self.assertIsNone(failing.current_document_id(42))
            with closing(sqlite3.connect(database_path)) as connection:
                index_rows = connection.execute(
                    "SELECT COUNT(*) FROM mark_text_index"
                ).fetchone()[0]
            self.assertEqual(index_rows, 0)

    def test_regular_findmark_searches_csv_display_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                42,
                "marks.pdf",
                make_marks_pdf(["FULL-DATAMATRIX-001"]),
                "Марки",
            )
            inventory.import_search_index_csv(
                42,
                "marks_index.csv",
                make_search_index_csv(
                    [
                        {
                            "Номер": 1,
                            "Набор": "1",
                            "Лист": 1,
                            "Ряд": 1,
                            "Колонка": 1,
                            "Название": "Сабо COVANI SMS25-BHLM3-015H серый замша (37)",
                            "Код": "SHORT-CODE-001",
                        }
                    ]
                ),
            )

            result = inventory.claim(42, "серый замша")

            self.assertEqual(result.matched, 1)
            self.assertEqual(result.locations[0].position_on_page, 1)

    def test_regular_findmark_searches_csv_short_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                42,
                "marks.pdf",
                make_marks_pdf(["FULL-DATAMATRIX-001"]),
                "Марки",
            )
            short_code = "0100000000000000215AbCdEfGhIjKl"
            inventory.import_search_index_csv(
                42,
                "marks_index.csv",
                make_search_index_csv(
                    [
                        {
                            "Номер": 1,
                            "Набор": "1",
                            "Лист": 1,
                            "Ряд": 1,
                            "Колонка": 1,
                            "Название": "Сабо COVANI SMS25-BHLM3-015H серые (37)",
                            "Код": short_code,
                        }
                    ]
                ),
            )

            result = inventory.claim(42, short_code)

            self.assertEqual(result.matched, 1)

    def test_complete_csv_index_skips_ocr_for_missing_article(self):
        class NoOcrInventory(MarksInventory):
            def _ocr_page_words(self, page, timeout_seconds=None):
                self.fail_if_called = True
                return []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = NoOcrInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.fail_if_called = False
            inventory.import_pdf(
                42,
                "marks.pdf",
                make_marks_pdf(["FULL-DATAMATRIX-001"]),
                "Марки",
            )
            inventory.import_search_index_csv(
                42,
                "marks_index.csv",
                make_search_index_csv(
                    [
                        {
                            "Номер": 1,
                            "Набор": "1",
                            "Лист": 1,
                            "Ряд": 1,
                            "Колонка": 1,
                            "Название": "Сабо COVANI OTHER-001 чёрные (36)",
                            "Код": "SHORT-CODE-001",
                        }
                    ]
                ),
            )

            result = inventory.claim_by_article(42, "COVANI MISSING-999")

            self.assertEqual(result.matched, 0)
            self.assertFalse(inventory.fail_if_called)

    def test_claim_by_article_reports_page_and_position_on_sheet(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=2,
                grid_rows=1,
                cell_width_pt=180,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                chat_id=42,
                file_name="shirts.pdf",
                pdf_bytes=make_marks_pdf(
                    ["MARK-ALPHA-001", "MARK-BETA-002"],
                    ["COVANI OTHER-001", "COVANI SMS25-BHLM3-015H"],
                ),
                product_name="Футболки",
            )

            result = inventory.claim_by_article(42, "COVANI SMS25-BHLM3")

            self.assertEqual(result.matched, 1)
            self.assertEqual(result.available, 1)
            self.assertTrue(result.pdf_bytes.startswith(b"%PDF"))
            self.assertEqual(len(result.locations), 1)
            self.assertEqual(result.locations[0].article, "COVANI SMS25-BHLM3-015H")
            self.assertEqual(result.locations[0].page_number, 1)
            self.assertEqual(result.locations[0].position_on_page, 2)

    def test_position_keeps_physical_slot_when_middle_cell_has_no_decoded_code(self):
        document = pymupdf.open()
        page = document.new_page(width=450, height=200)
        for x0, code, article in (
            (30, "MARK-LEFT-001", "COVANI LEFT-001"),
            (310, "MARK-RIGHT-001", "COVANI RIGHT-001"),
        ):
            page.insert_image(
                pymupdf.Rect(x0, 25, x0 + 100, 125),
                stream=make_datamatrix_png(code),
            )
            page.insert_text((x0, 160), article, fontsize=8)
        pdf_bytes = document.tobytes(deflate=True)
        document.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=3,
                grid_rows=1,
                cell_width_pt=130,
                cell_height_pt=180,
            )
            inventory.import_pdf(42, "gapped.pdf", pdf_bytes, "Футболки")

            result = inventory.claim(42, "MARK-RIGHT-001")

            self.assertEqual(result.matched, 1)
            self.assertEqual(result.locations[0].position_on_page, 3)
            with sqlite3.connect(root / "marks.db") as connection:
                selected_x0 = connection.execute(
                    "SELECT x0 FROM marks WHERE status = 'selected'"
                ).fetchone()[0]
            self.assertAlmostEqual(selected_x0, 300.0, delta=3.0)

    def test_article_search_tolerates_brand_missed_by_ocr(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                chat_id=42,
                file_name="shirts.pdf",
                pdf_bytes=make_marks_pdf(
                    ["MARK-ALPHA-001"],
                    ["SMS25-BHLM3-017A"],
                ),
                product_name="Футболки",
            )

            result = inventory.claim_by_article(42, "COVANI SMS25-BHLM3")

            self.assertEqual(result.matched, 1)
            self.assertEqual(result.locations[0].article, "SMS25-BHLM3-017A")

    def test_article_search_does_not_fall_back_to_document_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                chat_id=42,
                file_name="shirts.pdf",
                pdf_bytes=make_marks_pdf(
                    ["MARK-ALPHA-001"],
                    ["COVANI OTHER-001"],
                ),
                product_name="COVANI SMS25-BHLM3",
            )

            result = inventory.claim_by_article(42, "COVANI SMS25-BHLM3")

            self.assertEqual(result.matched, 0)
            self.assertEqual(result.available, 1)

    def test_article_search_can_be_limited_to_current_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            older = inventory.import_pdf(
                42,
                "older.pdf",
                make_marks_pdf(
                    ["MARK-OLDER-001"],
                    ["COVANI SMS25-BHLM3-015H"],
                ),
                "Старая партия",
            )
            current = inventory.import_pdf(
                42,
                "current.pdf",
                make_marks_pdf(
                    ["MARK-CURRENT-001"],
                    ["COVANI OTHER-001"],
                ),
                "Текущий документ",
            )

            current_result = inventory.claim_by_article(
                42,
                "COVANI SMS25-BHLM3",
                document_id=current.document_id,
            )
            older_result = inventory.claim_by_article(
                42,
                "COVANI SMS25-BHLM3",
                document_id=older.document_id,
            )

            self.assertEqual(current_result.matched, 0)
            self.assertEqual(older_result.matched, 1)

    def test_article_search_treats_like_wildcards_as_literal_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                42,
                "shirts.pdf",
                make_marks_pdf(
                    ["MARK-ALPHA-001"],
                    ["COVANI SMS25-BHLM3-015H"],
                ),
                "Футболки",
            )

            result = inventory.claim_by_article(42, "%")

            self.assertEqual(result.matched, 0)
            self.assertEqual(result.available, 1)

    def test_article_search_respects_model_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                42,
                "shirts.pdf",
                make_marks_pdf(
                    ["MARK-ALPHA-001"],
                    ["COVANI SMS25-BHLM30-015H"],
                ),
                "Футболки",
            )

            result = inventory.claim_by_article(42, "COVANI SMS25-BHLM3")

            self.assertEqual(result.matched, 0)
            self.assertEqual(result.available, 1)

    def test_scanned_pdf_uses_ocr_words_to_index_article(self):
        class OcrInventory(MarksInventory):
            def _ocr_page_words(self, page, timeout_seconds=None):
                return [
                    (225.0, 150.0, 260.0, 165.0, "COVANI"),
                    (265.0, 150.0, 370.0, 165.0, "SMS25-BHLM3-015H"),
                ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = OcrInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=2,
                grid_rows=1,
                cell_width_pt=180,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                chat_id=42,
                file_name="scanned.pdf",
                pdf_bytes=make_marks_pdf(["MARK-ALPHA-001", "MARK-BETA-002"]),
                product_name="Футболки",
            )

            result = inventory.claim_by_article(42, "COVANI SMS25-BHLM3")

            self.assertEqual(result.matched, 1)
            self.assertEqual(result.locations[0].position_on_page, 2)

    def test_incomplete_text_layer_falls_back_to_ocr_for_article(self):
        class CountingOcrInventory(MarksInventory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.ocr_calls = 0

            def _ocr_page_words(self, page, timeout_seconds=None):
                self.ocr_calls += 1
                return [(25.0, 150.0, 250.0, 165.0, "COVANI SMS25-BHLM3-015H")]

        document = pymupdf.open()
        page = document.new_page(width=400, height=200)
        page.insert_image(
            pymupdf.Rect(25, 25, 135, 135),
            stream=make_datamatrix_png("MARK-ALPHA-001"),
        )
        page.insert_text((300, 190), "unrelated footer")
        pdf_bytes = document.tobytes(deflate=True)
        document.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = CountingOcrInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(42, "partial-text.pdf", pdf_bytes, "Футболки")

            result = inventory.claim_by_article(42, "COVANI SMS25-BHLM3")

            self.assertEqual(result.matched, 1)
            self.assertEqual(inventory.ocr_calls, 1)

    def test_temporary_ocr_failure_is_retryable(self):
        class FlakyOcrInventory(MarksInventory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.ocr_calls = 0

            def _ocr_page_words(self, page, timeout_seconds=None):
                self.ocr_calls += 1
                if self.ocr_calls == 1:
                    raise ArticleOcrError("temporary OCR failure")
                return [(25.0, 150.0, 250.0, 165.0, "COVANI SMS25-BHLM3-015H")]

        document = pymupdf.open()
        page = document.new_page(width=400, height=200)
        page.insert_image(
            pymupdf.Rect(25, 25, 135, 135),
            stream=make_datamatrix_png("MARK-ALPHA-001"),
        )
        pdf_bytes = document.tobytes(deflate=True)
        document.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = FlakyOcrInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(42, "scan.pdf", pdf_bytes, "Футболки")

            with self.assertRaises(ArticleOcrError):
                inventory.claim_by_article(42, "COVANI SMS25-BHLM3")
            with sqlite3.connect(root / "marks.db") as connection:
                indexed_pages = connection.execute(
                    "SELECT COUNT(*) FROM article_index_pages"
                ).fetchone()[0]

            result = inventory.claim_by_article(42, "COVANI SMS25-BHLM3")

            self.assertEqual(indexed_pages, 0)
            self.assertEqual(result.matched, 1)
            self.assertEqual(inventory.ocr_calls, 2)

    def test_empty_ocr_result_is_retried_on_next_article_search(self):
        class EmptyThenSuccessfulOcrInventory(MarksInventory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.ocr_calls = 0

            def _ocr_page_words(self, page, timeout_seconds=None):
                self.ocr_calls += 1
                if self.ocr_calls == 1:
                    return []
                return [
                    (25.0, 150.0, 250.0, 165.0, "COVANI SMS25-BHLM3-015H")
                ]

        document = pymupdf.open()
        page = document.new_page(width=400, height=200)
        page.insert_image(
            pymupdf.Rect(25, 25, 135, 135),
            stream=make_datamatrix_png("MARK-ALPHA-001"),
        )
        pdf_bytes = document.tobytes(deflate=True)
        document.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = EmptyThenSuccessfulOcrInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(42, "scan.pdf", pdf_bytes, "Футболки")

            with self.assertRaises(ArticleOcrError):
                inventory.claim_by_article(42, "COVANI SMS25-BHLM3")
            second = inventory.claim_by_article(42, "COVANI SMS25-BHLM3")

            self.assertEqual(second.matched, 1)
            self.assertEqual(inventory.ocr_calls, 2)

    def test_partial_ocr_results_accumulate_without_erasing_articles(self):
        class AlternatingOcrInventory(MarksInventory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.ocr_calls = 0

            def _ocr_page_words(self, page, timeout_seconds=None):
                self.ocr_calls += 1
                if self.ocr_calls == 1:
                    return [
                        (25.0, 150.0, 180.0, 165.0, "COVANI FIRST-001")
                    ]
                return [
                    (225.0, 150.0, 390.0, 165.0, "COVANI SECOND-002")
                ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = AlternatingOcrInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=2,
                grid_rows=1,
                cell_width_pt=180,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                42,
                "partial.pdf",
                make_marks_pdf(["MARK-FIRST-001", "MARK-SECOND-002"]),
                "Футболки",
            )

            with self.assertRaises(ArticleOcrError):
                inventory.claim_by_article(42, "COVANI MISSING-999")
            result = inventory.claim_by_article(42, "COVANI SECOND-002")

            self.assertEqual(result.matched, 1)
            with sqlite3.connect(root / "marks.db") as connection:
                articles = connection.execute(
                    "SELECT article FROM marks ORDER BY x0"
                ).fetchall()
                indexed_pages = connection.execute(
                    "SELECT COUNT(*) FROM article_index_pages"
                ).fetchone()[0]
            self.assertEqual(
                [row[0] for row in articles],
                ["FIRST-001", "SECOND-002"],
            )
            self.assertEqual(indexed_pages, 1)
            self.assertEqual(inventory.ocr_calls, 2)

    def test_ocr_concurrency_limit_includes_page_rasterization(self):
        class FakePixmap:
            width = 100
            height = 100

            def tobytes(self, output_format):
                return b"fake-png"

        state_lock = threading.Lock()
        state = {"active": 0, "maximum": 0}

        class TrackingPage:
            rect = SimpleNamespace(width=100, height=100)

            def get_pixmap(self, dpi, alpha):
                with state_lock:
                    state["active"] += 1
                    state["maximum"] = max(state["maximum"], state["active"])
                time.sleep(0.03)
                with state_lock:
                    state["active"] -= 1
                return FakePixmap()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                root / "marks.db",
                root / "pdfs",
                max_concurrent_ocr=1,
            )
            completed = SimpleNamespace(
                stdout=(
                    b"level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
                    b"left\ttop\twidth\theight\tconf\ttext\n"
                )
            )

            with patch("marks_inventory.subprocess.run", return_value=completed):
                with ThreadPoolExecutor(max_workers=3) as executor:
                    list(executor.map(inventory._ocr_page_words, [TrackingPage()] * 3))

            self.assertEqual(state["maximum"], 1)

    def test_ocr_page_budget_preserves_progress_for_next_request(self):
        class CountingOcrInventory(MarksInventory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.ocr_calls = 0

            def _ocr_page_words(self, page, timeout_seconds=None):
                self.ocr_calls += 1
                article = (
                    "COVANI OTHER-001"
                    if self.ocr_calls == 1
                    else "COVANI SMS25-BHLM3-015H"
                )
                return [(25.0, 150.0, 250.0, 165.0, article)]

        document = pymupdf.open()
        for code in ("MARK-PAGE-ONE", "MARK-PAGE-TWO"):
            page = document.new_page(width=400, height=200)
            page.insert_image(
                pymupdf.Rect(25, 25, 135, 135),
                stream=make_datamatrix_png(code),
            )
        pdf_bytes = document.tobytes(deflate=True)
        document.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = CountingOcrInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
                ocr_max_pages=1,
            )
            inventory.import_pdf(42, "two-pages.pdf", pdf_bytes, "Футболки")

            with self.assertRaises(ArticleOcrError):
                inventory.claim_by_article(42, "COVANI SMS25-BHLM3")
            result = inventory.claim_by_article(42, "COVANI SMS25-BHLM3")

            self.assertEqual(result.matched, 1)
            self.assertEqual(result.locations[0].page_number, 2)
            self.assertEqual(inventory.ocr_calls, 2)

    def test_article_indexing_for_different_chats_does_not_share_one_lock(self):
        class ConcurrentInventory(MarksInventory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.measure_concurrency = False
                self.active_decodes = 0
                self.max_active_decodes = 0
                self.state_lock = threading.Lock()
                self.barrier = threading.Barrier(2)

            def _decode_pdf(self, *args, **kwargs):
                if self.measure_concurrency and kwargs.get("page_indexes"):
                    with self.state_lock:
                        self.active_decodes += 1
                        self.max_active_decodes = max(
                            self.max_active_decodes,
                            self.active_decodes,
                        )
                    try:
                        self.barrier.wait(timeout=0.5)
                    except threading.BrokenBarrierError:
                        pass
                    finally:
                        with self.state_lock:
                            self.active_decodes -= 1
                return super()._decode_pdf(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = ConcurrentInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            pdf_bytes = make_marks_pdf(
                ["MARK-ALPHA-001"],
                ["COVANI SMS25-BHLM3-015H"],
            )
            inventory.import_pdf(1, "first.pdf", pdf_bytes, "Футболки")
            inventory.import_pdf(2, "second.pdf", pdf_bytes, "Футболки")
            inventory.measure_concurrency = True

            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(inventory.reindex_articles, (1, 2)))

            self.assertEqual(inventory.max_active_decodes, 2)

    def test_article_search_ocr_is_lazy_and_stops_after_matching_page(self):
        class CountingOcrInventory(MarksInventory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.ocr_calls = 0

            def _ocr_page_words(self, page, timeout_seconds=None):
                self.ocr_calls += 1
                article = (
                    "COVANI SMS25-BHLM3-015H"
                    if self.ocr_calls == 1
                    else "COVANI OTHER-001"
                )
                return [(25.0, 150.0, 250.0, 165.0, article)]

        document = pymupdf.open()
        for code in ("MARK-PAGE-ONE", "MARK-PAGE-TWO"):
            page = document.new_page(width=400, height=200)
            page.insert_image(
                pymupdf.Rect(25, 25, 135, 135),
                stream=make_datamatrix_png(code),
            )
        pdf_bytes = document.tobytes(deflate=True)
        document.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = CountingOcrInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )

            inventory.import_pdf(42, "two-pages.pdf", pdf_bytes, "Футболки")
            self.assertEqual(inventory.ocr_calls, 0)

            result = inventory.claim_by_article(42, "COVANI SMS25-BHLM3")

            self.assertEqual(result.matched, 1)
            self.assertEqual(result.locations[0].page_number, 1)
            self.assertEqual(inventory.ocr_calls, 1)

    def test_article_search_indexes_enough_pages_for_requested_quantity(self):
        class CountingOcrInventory(MarksInventory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.ocr_calls = 0

            def _ocr_page_words(self, page, timeout_seconds=None):
                self.ocr_calls += 1
                return [(25.0, 150.0, 250.0, 165.0, "COVANI SMS25-BHLM3-015H")]

        document = pymupdf.open()
        for code in ("MARK-PAGE-ONE", "MARK-PAGE-TWO"):
            page = document.new_page(width=400, height=200)
            page.insert_image(
                pymupdf.Rect(25, 25, 135, 135),
                stream=make_datamatrix_png(code),
            )
        pdf_bytes = document.tobytes(deflate=True)
        document.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = CountingOcrInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(42, "two-pages.pdf", pdf_bytes, "Футболки")

            result = inventory.claim_by_article(42, "COVANI SMS25-BHLM3", quantity=2)

            self.assertEqual(result.matched, 2)
            self.assertEqual(inventory.ocr_calls, 2)

    def test_article_search_continues_indexing_when_partial_match_exists(self):
        class CountingOcrInventory(MarksInventory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.ocr_calls = 0

            def _ocr_page_words(self, page, timeout_seconds=None):
                self.ocr_calls += 1
                return [(25.0, 150.0, 250.0, 165.0, "COVANI SMS25-BHLM3-015H")]

        document = pymupdf.open()
        for code in ("MARK-PAGE-ONE", "MARK-PAGE-TWO"):
            page = document.new_page(width=400, height=200)
            page.insert_image(
                pymupdf.Rect(25, 25, 135, 135),
                stream=make_datamatrix_png(code),
            )
        pdf_bytes = document.tobytes(deflate=True)
        document.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = CountingOcrInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(42, "two-pages.pdf", pdf_bytes, "Футболки")
            inventory.reindex_articles(
                42,
                "SMS25-BHLM3",
                stop_article_quantity=1,
            )

            result = inventory.claim_by_article(42, "COVANI SMS25-BHLM3", quantity=2)

            self.assertEqual(result.matched, 2)
            self.assertEqual(inventory.ocr_calls, 2)

    def test_repeated_article_search_continues_past_selected_page(self):
        class CountingOcrInventory(MarksInventory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.ocr_calls = 0

            def _ocr_page_words(self, page, timeout_seconds=None):
                self.ocr_calls += 1
                return [(25.0, 150.0, 250.0, 165.0, "COVANI SMS25-BHLM3-015H")]

        document = pymupdf.open()
        for code in ("MARK-PAGE-ONE", "MARK-PAGE-TWO"):
            page = document.new_page(width=400, height=200)
            page.insert_image(
                pymupdf.Rect(25, 25, 135, 135),
                stream=make_datamatrix_png(code),
            )
        pdf_bytes = document.tobytes(deflate=True)
        document.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = CountingOcrInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(42, "two-pages.pdf", pdf_bytes, "Футболки")

            first = inventory.claim_by_article(42, "COVANI SMS25-BHLM3")
            second = inventory.claim_by_article(42, "COVANI SMS25-BHLM3")

            self.assertEqual(first.locations[0].page_number, 1)
            self.assertEqual(second.matched, 1)
            self.assertEqual(second.locations[0].page_number, 2)
            self.assertEqual(inventory.ocr_calls, 2)

    def test_existing_database_is_migrated_and_reindexed_without_reupload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            storage_dir = root / "pdfs"
            storage_dir.mkdir()
            pdf_bytes = make_marks_pdf(
                ["MARK-ALPHA-001", "MARK-BETA-002"],
                ["COVANI OTHER-001", "COVANI SMS25-BHLM3-015H"],
            )
            pdf_path = storage_dir / "legacy.pdf"
            pdf_path.write_bytes(pdf_bytes)
            database_path = root / "marks.db"
            with sqlite3.connect(database_path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE documents (
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
                    CREATE TABLE marks (
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
                connection.execute(
                    """
                    INSERT INTO documents(
                        id, chat_id, file_name, product_name, search_name, sha256, pdf_path
                    ) VALUES (1, 42, 'legacy.pdf', 'Футболки', 'футболки legacy.pdf', 'legacy', ?)
                    """,
                    (str(pdf_path),),
                )
                connection.executemany(
                    """
                    INSERT INTO marks(
                        document_id, code, normalized_code, page_index, x0, y0, x1, y1
                    ) VALUES (1, ?, ?, 0, ?, 10, ?, 190)
                    """,
                    [
                        ("MARK-ALPHA-001", "mark-alpha-001", 10, 190),
                        ("MARK-BETA-002", "mark-beta-002", 210, 390),
                    ],
                )

            inventory = MarksInventory(
                database_path,
                storage_dir,
                scan_dpi=200,
                grid_columns=2,
                grid_rows=1,
                cell_width_pt=180,
                cell_height_pt=180,
            )

            result = inventory.claim_by_article(42, "COVANI SMS25-BHLM3")

            self.assertEqual(result.matched, 1)
            self.assertEqual(result.locations[0].position_on_page, 2)

    def test_decode_image_reads_datamatrix_from_photo_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(root / "marks.db", root / "pdfs")

            codes = inventory.decode_image(make_datamatrix_png("PHOTO-MARK-001"))

            self.assertEqual(codes, ["PHOTO-MARK-001"])

    def test_normalize_code_accepts_scanner_symbology_prefix_and_gs_separator(self):
        stored = normalize_code("(01)04601234567890(21)ABC123(91)EE(92)SIGN")
        scanned = normalize_code("]d2010460123456789021ABC123\x1d91EE92SIGN")

        self.assertEqual(scanned, stored)

    def test_import_pdf_discovers_marks_and_tracks_available_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=2,
                grid_rows=1,
                cell_width_pt=180,
                cell_height_pt=180,
            )

            result = inventory.import_pdf(
                chat_id=42,
                file_name="shirts.pdf",
                pdf_bytes=make_marks_pdf(["MARK-ALPHA-001", "MARK-BETA-002"]),
                product_name="Футболки",
            )

            self.assertEqual(result.imported, 2)
            self.assertEqual(result.total, 2)
            self.assertEqual(result.available, 2)
            self.assertFalse(result.duplicate)
            self.assertEqual(inventory.stats(42).available, 2)

    def test_claim_by_datamatrix_marks_one_cell_green_and_decrements_stock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=2,
                grid_rows=1,
                cell_width_pt=180,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                chat_id=42,
                file_name="shirts.pdf",
                pdf_bytes=make_marks_pdf(["MARK-ALPHA-001", "MARK-BETA-002"]),
                product_name="Футболки",
            )

            result = inventory.claim(42, "MARK-ALPHA-001", quantity=1)

            self.assertEqual(result.matched, 1)
            self.assertEqual(result.available, 1)
            self.assertTrue(result.pdf_bytes.startswith(b"%PDF"))
            self.assertEqual(inventory.stats(42).selected, 1)

            marked_pdf = pymupdf.open(stream=result.pdf_bytes, filetype="pdf")
            drawings = marked_pdf[0].get_drawings()
            green_fills = [
                drawing
                for drawing in drawings
                if drawing.get("fill")
                and drawing["fill"][1] > drawing["fill"][0]
                and drawing["fill"][1] > drawing["fill"][2]
            ]
            marked_pdf.close()
            self.assertTrue(green_fills)
            self.assertGreater(green_fills[-1]["rect"].width, 150)
            self.assertGreater(green_fills[-1]["rect"].height, 150)

    def test_claim_can_be_released_until_it_is_finalized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                42,
                "shirts.pdf",
                make_marks_pdf(["MARK-ALPHA-001"]),
                "Футболки",
            )

            first = inventory.claim(42, "MARK-ALPHA-001")

            self.assertTrue(first.claim_token)
            self.assertEqual(inventory.release_claim(99, first.claim_token), 0)
            self.assertEqual(inventory.stats(42).selected, 1)
            self.assertEqual(inventory.release_claim(42, first.claim_token), 1)
            self.assertEqual(inventory.stats(42).available, 1)

            second = inventory.claim(42, "MARK-ALPHA-001")

            self.assertEqual(inventory.finalize_claim(42, second.claim_token), 1)
            self.assertEqual(inventory.release_claim(42, second.claim_token), 0)
            self.assertEqual(inventory.stats(42).selected, 1)

    def test_unfinished_claim_is_not_released_without_confirmed_send_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "marks.db"
            storage_dir = root / "pdfs"
            inventory = MarksInventory(
                database_path,
                storage_dir,
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                42,
                "shirts.pdf",
                make_marks_pdf(["MARK-ALPHA-001"]),
                "Футболки",
            )
            claim = inventory.claim(42, "MARK-ALPHA-001")
            self.assertTrue(claim.claim_token)
            self.assertEqual(inventory.stats(42).selected, 1)

            restarted = MarksInventory(
                database_path,
                storage_dir,
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )

            self.assertEqual(restarted.stats(42).available, 0)
            self.assertEqual(restarted.stats(42).selected, 1)
            with closing(sqlite3.connect(database_path)) as connection:
                pending_tokens = connection.execute(
                    "SELECT COUNT(*) FROM marks WHERE claim_token <> ''"
                ).fetchone()[0]
            self.assertEqual(pending_tokens, 1)

    def test_current_document_is_persisted_and_duplicate_reupload_makes_it_current(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "marks.db"
            storage_dir = root / "pdfs"
            inventory = MarksInventory(
                database_path,
                storage_dir,
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            first_pdf = make_marks_pdf(["MARK-FIRST-001"])
            second_pdf = make_marks_pdf(["MARK-SECOND-001"])
            first = inventory.import_pdf(42, "first.pdf", first_pdf, "Первая")
            second = inventory.import_pdf(42, "second.pdf", second_pdf, "Вторая")

            restarted = MarksInventory(database_path, storage_dir)

            self.assertEqual(restarted.current_document_id(42), second.document_id)
            duplicate = restarted.import_pdf(42, "first.pdf", first_pdf, "Первая")
            self.assertTrue(duplicate.duplicate)
            self.assertEqual(restarted.current_document_id(42), first.document_id)

    def test_article_search_waits_for_current_document_import_to_finish(self):
        class BlockingImportInventory(MarksInventory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.block_next_decode = False
                self.import_started = threading.Event()
                self.continue_import = threading.Event()

            def _decode_pdf(self, *args, **kwargs):
                if self.block_next_decode:
                    self.block_next_decode = False
                    self.import_started.set()
                    self.continue_import.wait(timeout=2)
                return super()._decode_pdf(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = BlockingImportInventory(
                root / "marks.db",
                root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                42,
                "older.pdf",
                make_marks_pdf(
                    ["MARK-OLDER-001"],
                    ["COVANI SMS25-BHLM3-015H"],
                ),
                "Старый документ",
            )
            inventory.block_next_decode = True

            with ThreadPoolExecutor(max_workers=2) as executor:
                import_future = executor.submit(
                    inventory.import_pdf,
                    42,
                    "current.pdf",
                    make_marks_pdf(
                        ["MARK-CURRENT-001"],
                        ["COVANI OTHER-001"],
                    ),
                    "Текущий документ",
                )
                self.assertTrue(inventory.import_started.wait(timeout=1))
                search_future = executor.submit(
                    inventory.claim_current_by_article,
                    42,
                    "COVANI SMS25-BHLM3",
                )
                time.sleep(0.05)
                self.assertFalse(search_future.done())
                inventory.continue_import.set()
                current = import_future.result(timeout=2)
                search_result = search_future.result(timeout=2)

            self.assertEqual(inventory.current_document_id(42), current.document_id)
            self.assertEqual(search_result.matched, 0)

    def test_claim_by_product_name_selects_requested_quantity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=2,
                grid_rows=1,
                cell_width_pt=180,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                chat_id=42,
                file_name="shirts.pdf",
                pdf_bytes=make_marks_pdf(["MARK-ALPHA-001", "MARK-BETA-002"]),
                product_name="Зелёные футболки",
            )

            result = inventory.claim(42, "футболки", quantity=2)

            self.assertEqual(result.matched, 2)
            self.assertEqual(result.available, 0)

    def test_claim_result_reports_global_inventory_across_documents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                42,
                "first.pdf",
                make_marks_pdf(["MARK-FIRST-001"]),
                "Первая партия",
            )
            inventory.import_pdf(
                42,
                "second.pdf",
                make_marks_pdf(["MARK-SECOND-001"]),
                "Вторая партия",
            )

            result = inventory.claim(42, "MARK-SECOND-001")

            self.assertEqual(result.matched, 1)
            self.assertEqual(result.total, 2)
            self.assertEqual(result.available, 1)

    def test_reimporting_same_pdf_does_not_duplicate_inventory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                db_path=root / "marks.db",
                storage_dir=root / "pdfs",
                scan_dpi=200,
                grid_columns=2,
                grid_rows=1,
                cell_width_pt=180,
                cell_height_pt=180,
            )
            pdf_bytes = make_marks_pdf(["MARK-ALPHA-001", "MARK-BETA-002"])
            inventory.import_pdf(42, "shirts.pdf", pdf_bytes, "Футболки")

            duplicate = inventory.import_pdf(42, "shirts.pdf", pdf_bytes, "Футболки")

            self.assertTrue(duplicate.duplicate)
            self.assertEqual(duplicate.imported, 0)
            self.assertEqual(inventory.stats(42).total, 2)
            self.assertEqual(inventory.stats(42).documents, 1)

    def test_probe_distinguishes_marks_pdf_from_plain_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(root / "marks.db", root / "pdfs", scan_dpi=200)
            plain = pymupdf.open()
            plain.new_page(width=300, height=200)
            plain_bytes = plain.tobytes()
            plain.close()

            self.assertTrue(inventory.looks_like_marks_pdf(make_marks_pdf(["MARK-001"])))
            self.assertFalse(inventory.looks_like_marks_pdf(plain_bytes))

    def test_empty_query_never_consumes_a_mark(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                root / "marks.db",
                root / "pdfs",
                scan_dpi=200,
                grid_columns=2,
                grid_rows=1,
                cell_width_pt=180,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                42,
                "shirts.pdf",
                make_marks_pdf(["MARK-ALPHA-001", "MARK-BETA-002"]),
                "Футболки",
            )

            result = inventory.claim(42, "   ")

            self.assertEqual(result.matched, 0)
            self.assertEqual(inventory.stats(42).available, 2)

    def test_query_that_normalizes_to_empty_never_consumes_a_mark(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                root / "marks.db",
                root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=380,
                cell_height_pt=180,
            )
            inventory.import_pdf(
                42,
                "marks.pdf",
                make_marks_pdf(["MARK-ALPHA-001"]),
                "Футболки",
            )

            result = inventory.claim(42, "()")

            self.assertEqual(result.matched, 0)
            self.assertEqual(result.available, 1)
            self.assertEqual(inventory.stats(42).selected, 0)

    def test_database_connections_are_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                inventory = MarksInventory(root / "marks.db", root / "pdfs")
                inventory.stats(42)
                gc.collect()

            database_warnings = [
                warning
                for warning in caught
                if warning.category is ResourceWarning
                and "unclosed database" in str(warning.message)
            ]
            self.assertEqual(database_warnings, [])

    def test_inventory_database_and_pdfs_use_private_permissions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = MarksInventory(
                root / "marks.db",
                root / "pdfs",
                scan_dpi=200,
                grid_columns=1,
                grid_rows=1,
                cell_width_pt=180,
                cell_height_pt=180,
            )
            inventory.import_pdf(42, "marks.pdf", make_marks_pdf(["MARK-001"]))

            stored_pdf = next((root / "pdfs").glob("*.pdf"))
            self.assertEqual(stat.S_IMODE((root / "marks.db").stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE((root / "pdfs").stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(stored_pdf.stat().st_mode), 0o600)

    def _make_two_mark_inventory(self, root):
        inventory = MarksInventory(
            db_path=root / "marks.db",
            storage_dir=root / "pdfs",
            scan_dpi=200,
            grid_columns=2,
            grid_rows=1,
            cell_width_pt=180,
            cell_height_pt=180,
        )
        inventory.import_pdf(
            42,
            "marks.pdf",
            make_marks_pdf(
                ["MARK-ALPHA-001", "MARK-BETA-002"],
                ["COVANI SMS25-BHLM3-015H", "COVANI SMS25-BHLM3-017"],
            ),
            "Марки",
        )
        return inventory

    def test_unselect_marks_returns_selected_mark_by_article(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inventory = self._make_two_mark_inventory(Path(temp_dir))
            claim = inventory.claim(42, "COVANI SMS25-BHLM3-015H")
            inventory.finalize_claim(42, claim.claim_token)
            self.assertEqual(inventory.stats(42).selected, 1)
            self.assertEqual(inventory.stats(42).available, 1)

            released = inventory.unselect_marks(42, "COVANI SMS25-BHLM3-015H")

            self.assertEqual(released, 1)
            self.assertEqual(inventory.stats(42).selected, 0)
            self.assertEqual(inventory.stats(42).available, 2)

    def test_unselect_marks_respects_requested_quantity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inventory = self._make_two_mark_inventory(Path(temp_dir))
            claim = inventory.claim(42, "Марки", quantity=2)
            inventory.finalize_claim(42, claim.claim_token)
            self.assertEqual(inventory.stats(42).selected, 2)

            released = inventory.unselect_marks(42, "Марки", quantity=1)

            self.assertEqual(released, 1)
            self.assertEqual(inventory.stats(42).selected, 1)
            self.assertEqual(inventory.stats(42).available, 1)

    def test_unselect_marks_matches_by_product_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inventory = self._make_two_mark_inventory(Path(temp_dir))
            claim = inventory.claim(42, "Марки", quantity=2)
            inventory.finalize_claim(42, claim.claim_token)
            self.assertEqual(inventory.stats(42).selected, 2)

            released = inventory.unselect_marks(42, "Марки")

            self.assertEqual(released, 1)
            self.assertEqual(inventory.stats(42).selected, 1)

    def test_unselect_marks_does_not_touch_available_marks_or_other_chats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inventory = self._make_two_mark_inventory(Path(temp_dir))
            inventory.import_pdf(
                77,
                "other.pdf",
                make_marks_pdf(["MARK-OTHER-001"]),
                "Чужие",
            )
            claim = inventory.claim(42, "MARK-ALPHA-001")
            inventory.finalize_claim(42, claim.claim_token)

            released = inventory.unselect_marks(42, "не существует")

            self.assertEqual(released, 0)
            self.assertEqual(inventory.stats(42).selected, 1)
            self.assertEqual(inventory.stats(77).available, 1)

    def test_export_selected_marks_returns_csv_with_mark_details(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inventory = self._make_two_mark_inventory(Path(temp_dir))
            claim = inventory.claim(42, "MARK-ALPHA-001")
            inventory.finalize_claim(42, claim.claim_token)

            csv_bytes = inventory.export_selected_marks(42)

            self.assertIsNotNone(csv_bytes)
            text = csv_bytes.decode("utf-8-sig")
            rows = list(csv.reader(io.StringIO(text), delimiter=";"))
            self.assertEqual(
                rows[0],
                [
                    "Артикул",
                    "Название",
                    "Лист",
                    "Позиция на листе",
                    "Выбрана",
                    "Короткий код",
                ],
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1][0], "COVANI SMS25-BHLM3-015H")
            self.assertEqual(rows[1][2], "1")
            self.assertEqual(rows[1][3], "1")
            self.assertNotEqual(rows[1][4], "")

    def test_export_selected_marks_returns_none_when_nothing_selected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inventory = self._make_two_mark_inventory(Path(temp_dir))

            self.assertIsNone(inventory.export_selected_marks(42))


if __name__ == "__main__":
    unittest.main()
