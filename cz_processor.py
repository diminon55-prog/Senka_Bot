import pymupdf as fitz
import io

PAGE_W = 1190.55118
PAGE_H = 841.889764
CELL_W = 136.063
CELL_H = 93.5433

def save_raw_cmyk_rez_pdf(page_streams):
    objects = []
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    kids_refs = " ".join([f"{3 + i*2} 0 R" for i in range(len(page_streams))])
    objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids_refs}] /Count {len(page_streams)} >>\nendobj\n".encode("latin1"))

    for i, stream_str in enumerate(page_streams):
        page_obj_id = 3 + i * 2
        content_obj_id = 4 + i * 2
        stream_bytes = stream_str.encode("latin1")
        page_obj = (
            f"{page_obj_id} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W:.5f} {PAGE_H:.6f}] "
            f"/Contents {content_obj_id} 0 R /Resources << /ProcSet [/PDF] >> >>\n"
            f"endobj\n"
        ).encode("latin1")
        content_obj = (
            f"{content_obj_id} 0 obj\n"
            f"<< /Length {len(stream_bytes)} >>\n"
            f"stream\n"
        ).encode("latin1") + stream_bytes + b"\nendstream\nendobj\n"
        objects.append(page_obj)
        objects.append(content_obj)

    out_bytes = io.BytesIO()
    out_bytes.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for obj in objects:
        offsets.append(out_bytes.tell())
        out_bytes.write(obj)
    xref_pos = out_bytes.tell()
    out_bytes.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin1"))
    for off in offsets:
        out_bytes.write(f"{off:010d} 00000 n \n".encode("latin1"))
    out_bytes.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode("latin1")
    )
    return out_bytes.getvalue()

def process_cz_archive(pdf_files_data, num_shift_x_mm=0.0, num_shift_y_mm=0.0, gap_x_mm=2.0, gap_y_mm=1.8):
    """
    Добавлены параметры смещения нумерации (в миллиметрах) и зазоров между марками.
    По умолчанию: gap_x=2.0 мм, gap_y=1.8 мм.
    """
    MM = 72.0 / 25.4
    label_png_bytes = []
    global_counter = 1
    
    # Динамический расчет сетки с центрированием на А3
    gap_x = gap_x_mm * MM
    gap_y = gap_y_mm * MM
    TOTAL_W = 8 * CELL_W + 7 * gap_x
    TOTAL_H = 8 * CELL_H + 7 * gap_y
    MARGIN_X = (PAGE_W - TOTAL_W) / 2.0
    MARGIN_Y = (PAGE_H - TOTAL_H) / 2.0
    
    X_COORDS = [MARGIN_X + col * (CELL_W + gap_x) for col in range(8)]
    # Y-координаты для PDF (отсчет снизу вверх)
    Y_COORDS_PDF = [PAGE_H - MARGIN_Y - CELL_H - row * (CELL_H + gap_y) for row in range(8)]
    
    # Смещение для текста и плашки
    shift_x = num_shift_x_mm * MM
    shift_y = num_shift_y_mm * MM
    
    for filename, pdf_bytes in pdf_files_data:
        src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(src_doc)):
            page = src_doc[page_num]
            rect = page.rect
            scale_x = rect.width / CELL_W
            scale_y = rect.height / CELL_H

            # Рисуем белый квадрат со смещением (shift_y вычитаем, чтобы двигать вверх)
	    # Белый прямоугольник (ластик) ДОЛЖЕН СТОЯТЬ НА МЕСТЕ, чтобы затирать исходный номер
            num_box = fitz.Rect(
                rect.x1 - 38 * scale_x,
                rect.y1 - 15 * scale_y,
                rect.x1 - 2 * scale_x,
                rect.y1 - 2 * scale_y
            )
            page.draw_rect(num_box, color=(1, 1, 1), fill=(1, 1, 1))		            
            # Пишем номер со смещением
            page.insert_text(
                fitz.Point(rect.x1 - 35 * scale_x + shift_x, rect.y1 - 5 * scale_y - shift_y),
                str(global_counter),
                fontsize=8.5 * scale_y,
                fontname="helv",
                color=(0, 0, 0)
            )
            global_counter += 1

            pix = page.get_pixmap(dpi=300, alpha=False)
            label_png_bytes.append(pix.tobytes("png"))
        src_doc.close()

    total_labels = len(label_png_bytes)
    if total_labels == 0:
        return None, None, 0

    doc_soderzhimoe_all = fitz.open()
    all_rez_streams = []

    per_sheet = 64
    num_sheets = (total_labels + per_sheet - 1) // per_sheet

    for sheet_idx in range(num_sheets):
        start_idx = sheet_idx * per_sheet
        end_idx = min(start_idx + per_sheet, total_labels)
        batch = label_png_bytes[start_idx:end_idx]

        doc_sod_sheet = fitz.open()
        page_sod = doc_sod_sheet.new_page(width=PAGE_W, height=PAGE_H)
        rez_lines = ["0 1 1 0 K", "0.25 w", "0 J", "0 j"]

        for i, png_data in enumerate(batch):
            row = i // 8
            col = i % 8

            x_pdf = X_COORDS[col]
            y_pdf = Y_COORDS_PDF[row]

            top_y = PAGE_H - (y_pdf + CELL_H)
            bottom_y = PAGE_H - y_pdf
            target_rect = fitz.Rect(x_pdf, top_y, x_pdf + CELL_W, bottom_y)

            page_sod.insert_image(target_rect, stream=png_data, keep_proportion=False)
            rez_lines.append(f"{x_pdf:.4f} {y_pdf:.4f} {CELL_W:.4f} {CELL_H:.4f} re S")

        doc_soderzhimoe_all.insert_pdf(doc_sod_sheet)
        doc_sod_sheet.close()

        stream_str = "\n".join(rez_lines)
        all_rez_streams.append(stream_str)

    sod_bytes = doc_soderzhimoe_all.tobytes(deflate=True, clean=True)
    doc_soderzhimoe_all.close()
    
    rez_bytes = save_raw_cmyk_rez_pdf(all_rez_streams)
    
    return sod_bytes, rez_bytes, total_labels
