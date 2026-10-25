import io
import gc
import stat
import tempfile
import unittest
import warnings
from pathlib import Path

import pymupdf
from PIL import Image
import zxingcpp

from marks_inventory import MarksInventory, normalize_code


def make_datamatrix_png(value: str) -> bytes:
    barcode = zxingcpp.create_barcode(value, zxingcpp.BarcodeFormat.DataMatrix)
    bitmap = zxingcpp.write_barcode_to_image(barcode, scale=5, add_quiet_zones=True)
    height, width = bitmap.shape
    image = Image.frombytes("L", (width, height), bytes(bitmap))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def make_marks_pdf(codes: list[str]) -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=400, height=200)
    for index, code in enumerate(codes):
        x0 = 25 + index * 200
        page.insert_image(
            pymupdf.Rect(x0, 25, x0 + 110, 135),
            stream=make_datamatrix_png(code),
        )
    data = document.tobytes(deflate=True)
    document.close()
    return data


class MarksInventoryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
