"""
==================================================================================
 Smart Multi-PDF to Excel Converter (Grid-Aware OCR & Data Alignment)
==================================================================================
A production-grade Streamlit application that:
  - Accepts 100+ PDF uploads at once
  - Detects whether each PDF is text-based or scanned (image-based)
  - Extracts tables using pdfplumber/camelot (text PDFs) 
  - Uses advanced OpenCV Grid-Detection for scanned PDFs to perfectly align columns
  - Detects multi-page table continuity and merges continued tables
  - Cleans extracted data and formats output (bold headers, auto-width)
  - Produces Excel workbooks per PDF + an optional Master Excel for all PDFs
==================================================================================
"""

import io
import os
import re
import zipfile
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
import cv2
from rapidfuzz import fuzz

# For premium Excel formatting
from openpyxl.styles import Font, PatternFill, Alignment

# camelot is optional at import time
try:
    import camelot
    CAMELOT_AVAILABLE = True
except Exception:
    CAMELOT_AVAILABLE = False


st.set_page_config(
    page_title="Multi-PDF to Excel Converter",
    page_icon="📊",
    layout="wide",
)

TEXT_DENSITY_THRESHOLD = 40
HEADER_SIMILARITY_THRESHOLD = 78   
COLUMN_COUNT_TOLERANCE = 0         
DTYPE_SIMILARITY_THRESHOLD = 0.6   
EXCEL_SHEET_NAME_MAX_LEN = 31  


@dataclass
class ExtractedTable:
    """A single raw table pulled from a PDF, tagged with provenance."""
    dataframe: pd.DataFrame
    page_number: int
    source: str


@dataclass
class FileResult:
    """Holds the outcome of processing a single uploaded PDF."""
    filename: str
    pdf_type: str = "unknown"
    status: str = "pending"
    error_message: str = ""
    sheet_names: List[str] = field(default_factory=list)
    preview_frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    excel_bytes: Optional[bytes] = None
    tables: List[pd.DataFrame] = field(default_factory=list)
    num_pages: int = 0
    num_tables_found: int = 0
    num_tables_after_merge: int = 0


def detect_pdf_type(pdf_bytes: bytes) -> Tuple[str, int]:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            num_pages = len(pdf.pages)
            if num_pages == 0:
                return "scanned", 0

            total_chars = 0
            pages_sampled = min(num_pages, 5)
            for page in pdf.pages[:pages_sampled]:
                text = page.extract_text() or ""
                total_chars += len(text.strip())

            avg_chars_per_page = total_chars / max(pages_sampled, 1)

            if avg_chars_per_page >= TEXT_DENSITY_THRESHOLD:
                return "text", num_pages
            else:
                return "scanned", num_pages
    except Exception:
        return "scanned", 0


def extract_text_pdf(pdf_bytes: bytes, filename: str) -> List[ExtractedTable]:
    extracted: List[ExtractedTable] = []
    camelot_pages_with_tables = set()

    if CAMELOT_AVAILABLE:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            for flavor in ("lattice", "stream"):
                try:
                    tables = camelot.read_pdf(tmp_path, pages="all", flavor=flavor)
                except Exception:
                    tables = []
                for t in tables:
                    df = t.df
                    if df is not None and not df.empty:
                        page_num = int(t.page) if hasattr(t, "page") else 0
                        extracted.append(ExtractedTable(df, page_num, f"camelot-{flavor}"))
                        camelot_pages_with_tables.add(page_num)
                if extracted:
                    break
        except Exception:
            pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                if i in camelot_pages_with_tables:
                    continue
                try:
                    page_tables = page.extract_tables()
                except Exception:
                    page_tables = []

                if page_tables:
                    for raw_table in page_tables:
                        if not raw_table:
                            continue
                        df = pd.DataFrame(raw_table)
                        if not df.empty:
                            extracted.append(ExtractedTable(df, i, "pdfplumber"))
                else:
                    text = page.extract_text() or ""
                    rows = []
                    for line in text.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        cols = re.split(r"\s{2,}|\t", line)
                        rows.append(cols)
                    if rows:
                        max_cols = max(len(r) for r in rows)
                        rows = [r + [""] * (max_cols - len(r)) for r in rows]
                        df = pd.DataFrame(rows)
                        if not df.empty:
                            extracted.append(ExtractedTable(df, i, "pdfplumber-text"))
    except Exception:
        pass

    return extracted


def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return processed


def _extract_tables_from_ocr(ocr_data: Dict[str, list], img: np.ndarray) -> List[pd.DataFrame]:
    """
    Highly robust OCR table extractor. Uses OpenCV to find grid lines, isolates tables into blocks,
    and accurately aligns columns based on grid layouts or X-axis projection profiles.
    """
    words = []
    n = len(ocr_data["text"])
    for i in range(n):
        text = str(ocr_data["text"][i]).strip()
        if not text: continue
        try:
            conf = float(ocr_data.get("conf", ["0"] * n)[i])
            if conf < 15: continue
        except: continue
        
        x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
        words.append({
            "text": text, "left": x, "top": y, "right": x+w, "bottom": y+h,
            "width": w, "height": h, "cx": x + w/2, "cy": y + h/2
        })
        
    if not words: 
        return []
    
    # 1. Row Formation using strict vertical overlap heuristics
    words.sort(key=lambda w: w["cy"])
    lines = []
    current_line = [words[0]]
    
    for w in words[1:]:
        avg_cy = sum(x["cy"] for x in current_line) / len(current_line)
        avg_h = sum(x["height"] for x in current_line) / len(current_line)
        if abs(w["cy"] - avg_cy) < max(avg_h * 0.4, 5):
            current_line.append(w)
        else:
            current_line.sort(key=lambda x: x["left"])
            lines.append(current_line)
            current_line = [w]
    if current_line:
        current_line.sort(key=lambda x: x["left"])
        lines.append(current_line)
        
    # 2. Block Formation (split separate tables by checking large vertical gaps)
    blocks = []
    current_block = [lines[0]]
    for i in range(1, len(lines)):
        prev_line = lines[i-1]
        curr_line = lines[i]
        prev_bottom = max(w["bottom"] for w in prev_line)
        curr_top = min(w["top"] for w in curr_line)
        avg_h = sum(w["height"] for w in curr_line) / len(curr_line)
        
        if (curr_top - prev_bottom) > avg_h * 2.2: # Significant visual gap = new table
            blocks.append(current_block)
            current_block = [curr_line]
        else:
            current_block.append(curr_line)
    if current_block:
        blocks.append(current_block)
        
    # 3. Column parsing per block using OpenCV Grid intelligence
    image_width = img.shape[1]
    block_dfs = []
    
    for block in blocks:
        min_y = max(0, min(w["top"] for line in block for w in line) - 10)
        max_y = min(img.shape[0], max(w["bottom"] for line in block for w in line) + 10)
        block_img = img[min_y:max_y, :]
        
        # Detect vertical grid lines in this specific table block
        gray = cv2.cvtColor(block_img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        kernel_h = max(15, int(block_img.shape[0] * 0.3)) 
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
        v_lines_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        
        contours, _ = cv2.findContours(v_lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x_coords = []
        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            if h_box >= kernel_h * 0.8: 
                x_coords.append(x + w_box//2)
                
        x_coords.sort()
        v_lines = []
        if x_coords:
            curr = [x_coords[0]]
            for x in x_coords[1:]:
                if x - curr[-1] < 10:
                    curr.append(x)
                else:
                    v_lines.append(int(np.mean(curr)))
                    curr = [x]
            v_lines.append(int(np.mean(curr)))
            
        cols = []
        if len(v_lines) >= 2:
            # Table has clear grid lines. Use them directly to define columns.
            v_lines = [0] + v_lines + [image_width]
            for i in range(len(v_lines)-1):
                cols.append((v_lines[i], v_lines[i+1]))
        else:
            # Borderless Table Fallback: Use Vertical Projection Profile
            profile = np.zeros(image_width, dtype=int)
            for line in block:
                for w in line:
                    profile[max(0, w["left"]):min(image_width, w["right"])] += 1
            
            is_gap = profile <= 0
            in_col = False
            start = 0
            for x in range(image_width):
                if not is_gap[x] and not in_col:
                    in_col = True
                    start = x
                elif is_gap[x] and in_col:
                    in_col = False
                    cols.append((start, x))
            if in_col:
                cols.append((start, image_width))
                
        # Map words to their perfect grid cells
        block_data = []
        for line in block:
            row_data = [""] * len(cols)
            for w in line:
                cx = w["cx"]
                best_col = 0
                min_dist = float("inf")
                for idx, (cmin, cmax) in enumerate(cols):
                    if cmin <= cx <= cmax:
                        best_col = idx
                        break
                    dist = min(abs(cx - cmin), abs(cx - cmax))
                    if dist < min_dist:
                        min_dist = dist
                        best_col = idx
                        
                if row_data[best_col]:
                    row_data[best_col] += " " + w["text"]
                else:
                    row_data[best_col] = w["text"]
            block_data.append(row_data)
            
        if block_data:
            block_dfs.append(pd.DataFrame(block_data))
            
    return block_dfs


def extract_ocr_pdf(pdf_bytes: bytes, filename: str, dpi: int = 200) -> List[ExtractedTable]:
    extracted: List[ExtractedTable] = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    except Exception as e:
        raise RuntimeError(f"Failed to rasterize PDF for OCR: {e}")

    for i, pil_page in enumerate(pages, start=1):
        try:
            # Keep original RGB image for grid detection, pass processed to Tesseract
            img = np.array(pil_page.convert("RGB"))
            processed = preprocess_image(img)
            
            ocr_data = pytesseract.image_to_data(
                processed, output_type=pytesseract.Output.DICT
            )
            
            # Map unstructured text back into robust DataFrames
            dfs = _extract_tables_from_ocr(ocr_data, img)
            for df in dfs:
                if not df.empty:
                    extracted.append(ExtractedTable(df, i, "ocr-smart"))
        except Exception:
            continue

    return extracted


def _infer_column_dtype_signature(series: pd.Series) -> str:
    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample != ""].head(20)
    if sample.empty: return "empty"
    numeric_count = sum(bool(re.fullmatch(r"-?\d+(\.\d+)?%?", v)) for v in sample)
    date_count = sum(bool(re.fullmatch(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", v)) for v in sample)
    if numeric_count / len(sample) >= 0.6: return "numeric"
    if date_count / len(sample) >= 0.6: return "date"
    return "text"

def _dtype_signature_similarity(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    n = min(len(df1.columns), len(df2.columns))
    if n == 0: return 0.0
    matches = sum(1 for c in range(n) if _infer_column_dtype_signature(df1.iloc[:, c]) == _infer_column_dtype_signature(df2.iloc[:, c]))
    return matches / n

def _header_similarity(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    header1 = [str(x) for x in df1.iloc[0].tolist()] if len(df1) else []
    header2 = [str(x) for x in df2.iloc[0].tolist()] if len(df2) else []
    if not header1 or not header2: return 0.0
    n = min(len(header1), len(header2))
    scores = [fuzz.ratio(header1[i], header2[i]) for i in range(n)]
    return sum(scores) / len(scores) if scores else 0.0

def are_tables_similar(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    if df1.empty or df2.empty: return False
    if abs(len(df1.columns) - len(df2.columns)) > COLUMN_COUNT_TOLERANCE: return False
    
    header_score = _header_similarity(df1, df2)
    dtype_score = _dtype_signature_similarity(df1, df2)

    if header_score >= HEADER_SIMILARITY_THRESHOLD: return True
    if dtype_score >= DTYPE_SIMILARITY_THRESHOLD and header_score >= 40: return True
    return False

def merge_tables(tables: List[pd.DataFrame]) -> pd.DataFrame:
    if not tables: return pd.DataFrame()
    if len(tables) == 1: return tables[0].reset_index(drop=True)

    base_header = [str(x).strip().lower() for x in tables[0].iloc[0].tolist()]
    merged_frames = [tables[0]]

    for t in tables[1:]:
        if t.empty: continue
        first_row = [str(x).strip().lower() for x in t.iloc[0].tolist()]
        n = min(len(first_row), len(base_header))
        if n and fuzz.ratio(" ".join(first_row[:n]), " ".join(base_header[:n])) >= HEADER_SIMILARITY_THRESHOLD:
            merged_frames.append(t.iloc[1:])
        else:
            merged_frames.append(t)

    return pd.concat(merged_frames, ignore_index=True, sort=False)

def group_and_merge_tables(extracted_tables: List[ExtractedTable]) -> List[pd.DataFrame]:
    if not extracted_tables: return []
    ordered = sorted(extracted_tables, key=lambda t: t.page_number)
    groups: List[List[pd.DataFrame]] = [[ordered[0].dataframe]]

    for et in ordered[1:]:
        last_df = groups[-1][-1]
        if are_tables_similar(last_df, et.dataframe):
            groups[-1].append(et.dataframe)
        else:
            groups.append([et.dataframe])

    return [merge_tables(g) for g in groups]


def clean_tables(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    df = df.copy()

    df = df.astype(str).apply(lambda col: col.str.replace(r"\s+", " ", regex=True).str.strip())
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    if df.empty: return df
    df = df.reset_index(drop=True)

    first_row = df.iloc[0]
    non_null_ratio = first_row.notna().mean()
    if non_null_ratio > 0.5:
        df.columns = [str(c) if pd.notna(c) else f"col_{i}" for i, c in enumerate(first_row)]
        df = df.iloc[1:].reset_index(drop=True)
    else:
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

    seen = {}
    new_cols = []
    for c in df.columns:
        c = str(c).strip() or "col"
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols

    header_lower = [str(c).strip().lower() for c in df.columns]
    def _is_header_dup(row):
        vals = [str(v).strip().lower() for v in row.tolist()]
        n = min(len(vals), len(header_lower))
        if n == 0: return False
        return fuzz.ratio(" ".join(vals[:n]), " ".join(header_lower[:n])) >= HEADER_SIMILARITY_THRESHOLD

    if len(df) > 0:
        dup_mask = df.apply(_is_header_dup, axis=1)
        df = df[~dup_mask]

    df = df.drop_duplicates(keep="first")
    df = df.dropna(axis=0, how="all")
    df = df.reset_index(drop=True)

    # Fill NaN with empty string for cleaner Excel generation
    df = df.fillna("")
    return df


def _safe_sheet_name(name: str, used_names: set) -> str:
    name = re.sub(r"[\[\]:*?/\\]", "_", name).strip() or "Sheet"
    name = name[:EXCEL_SHEET_NAME_MAX_LEN]
    base = name
    counter = 1
    while name in used_names:
        suffix = f"_{counter}"
        name = base[: EXCEL_SHEET_NAME_MAX_LEN - len(suffix)] + suffix
        counter += 1
    used_names.add(name)
    return name

def build_excel_workbook(tables: List[pd.DataFrame], layout_mode: str = "multi") -> Tuple[bytes, List[str], Dict[str, pd.DataFrame]]:
    """
    Write a list of cleaned tables into an Excel workbook.
    Applies Openpyxl formatting (Bold headers, auto column widths) for a premium layout.
    """
    output = io.BytesIO()
    used_names: set = set()
    sheet_names: List[str] = []
    previews: Dict[str, pd.DataFrame] = {}

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if not tables:
            pd.DataFrame({"Notice": ["No tables could be extracted."]}).to_excel(
                writer, sheet_name="Sheet1", index=False
            )
            sheet_names.append("Sheet1")
        else:
            if layout_mode == "single":
                sheet_name = _safe_sheet_name("All_Tables", used_names)
                sheet_names.append(sheet_name)
                current_row = 0
                
                for idx, df in enumerate(tables):
                    if df.empty: continue
                    if current_row == 0:
                        previews[sheet_name] = df.head(5)
                        
                    label_df = pd.DataFrame(columns=[f"--- Table {idx+1} ---"])
                    label_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                    df.to_excel(writer, sheet_name=sheet_name, startrow=current_row + 1, index=False)
                    
                    worksheet = writer.sheets[sheet_name]
                    header_row = current_row + 2 
                    
                    # Style headers
                    for col_idx in range(1, len(df.columns) + 1):
                        cell = worksheet.cell(row=header_row, column=col_idx)
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color="F0F0F0", end_color="F0F0F0", fill_type="solid")
                        cell.alignment = Alignment(horizontal="center")
                    
                    current_row += len(df) + 4
                    
                # Auto-fit columns for Single Sheet
                worksheet = writer.sheets[sheet_name]
                for col in worksheet.columns:
                    max_length = 0
                    column = col[0].column_letter 
                    for cell in col:
                        try: 
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except: pass
                    worksheet.column_dimensions[column].width = min(max_length + 2, 55)

            else:
                for idx, df in enumerate(tables, start=1):
                    if df.empty: continue
                    sheet_name = _safe_sheet_name(f"Table_{idx}", used_names)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    sheet_names.append(sheet_name)
                    previews[sheet_name] = df.head(5)
                    
                    worksheet = writer.sheets[sheet_name]
                    
                    # Format Header & Columns
                    for col in worksheet.columns:
                        max_length = 0
                        column = col[0].column_letter
                        for i, cell in enumerate(col):
                            if i == 0:  # Header style
                                cell.font = Font(bold=True)
                                cell.fill = PatternFill(start_color="F0F0F0", end_color="F0F0F0", fill_type="solid")
                                cell.alignment = Alignment(horizontal="center")
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except: pass
                        worksheet.column_dimensions[column].width = min(max_length + 2, 55)

    return output.getvalue(), sheet_names, previews


def process_single_pdf(filename: str, pdf_bytes: bytes, layout_mode: str) -> FileResult:
    result = FileResult(filename=filename, status="processing")

    pdf_type, num_pages = detect_pdf_type(pdf_bytes)
    result.pdf_type = pdf_type
    result.num_pages = num_pages

    if pdf_type == "text":
        raw_tables = extract_text_pdf(pdf_bytes, filename)
    else:
        raw_tables = extract_ocr_pdf(pdf_bytes, filename)

    result.num_tables_found = len(raw_tables)
    merged_tables = group_and_merge_tables(raw_tables)
    
    cleaned_tables = [clean_tables(df) for df in merged_tables]
    cleaned_tables = [df for df in cleaned_tables if not df.empty]
    
    result.tables = cleaned_tables
    result.num_tables_after_merge = len(cleaned_tables)

    excel_bytes, sheet_names, previews = build_excel_workbook(cleaned_tables, layout_mode)
    result.excel_bytes = excel_bytes
    result.sheet_names = sheet_names
    result.preview_frames = previews
    result.status = "done"

    return result


def process_multiple_pdfs(
    uploaded_files: list,
    overall_progress_bar,
    status_container,
    layout_mode: str
) -> List[FileResult]:
    results: List[FileResult] = []
    total = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files, start=1):
        filename = uploaded_file.name
        row = status_container.container()
        row_placeholder = row.empty()
        row_placeholder.info(f"⏳ Processing **{filename}** ({i}/{total})...")

        try:
            pdf_bytes = uploaded_file.getvalue()
            result = process_single_pdf(filename, pdf_bytes, layout_mode)
            results.append(result)
            row_placeholder.success(
                f"✅ **{filename}** — type: `{result.pdf_type}` — "
                f"Generated **{len(result.sheet_names)}** sheet(s) "
                f"({result.num_pages} pages)"
            )
        except Exception as e:
            err_result = FileResult(
                filename=filename,
                status="error",
                error_message=f"{type(e).__name__}: {e}",
            )
            results.append(err_result)
            row_placeholder.error(f"❌ **{filename}** failed: {err_result.error_message}")

        overall_progress_bar.progress(i / total, text=f"Overall progress: {i}/{total} files")

    return results

def build_zip_of_excels(results: List[FileResult]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            if r.status == "done" and r.excel_bytes:
                base_name = os.path.splitext(r.filename)[0]
                zf.writestr(f"{base_name}.xlsx", r.excel_bytes)
    return zip_buffer.getvalue()


def main():
    st.title("📊 Multi-PDF to Excel Converter")
    st.caption(
        "Upload multiple PDFs (text-based or scanned). Scanned tables are natively processed "
        "using OpenCV Grid-Intelligence to guarantee perfect column alignment."
    )

    if not CAMELOT_AVAILABLE:
        st.warning(
            "⚠️ `camelot-py` (or its Ghostscript dependency) is not available. Text-based extraction will fall back to pdfplumber only."
        )

    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("Data Export Layout")
        sheet_layout = st.radio(
            "How should tables be formatted inside the Excel file?",
            options=[
                "Multi-Sheet (One table per tab)", 
                "Single-Sheet (All tables stacked vertically)"
            ],
            index=0,
            help="Choose whether to place each extracted table on its own Excel tab, or stack them all on a single tab."
        )
        layout_mode = "single" if "Single" in sheet_layout else "multi"

        st.markdown("---")
        ocr_dpi = st.slider("OCR rasterization DPI", 100, 300, 200, step=25,
                             help="Higher DPI improves OCR accuracy but is slower.")
        
        st.markdown("---")
        st.markdown("**Core Engines:** openCV, Tesseract, pdfplumber, openpyxl")

    uploaded_files = st.file_uploader(
        "Upload PDF files (you can select 100+ at once)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.subheader(f"📁 {len(uploaded_files)} file(s) uploaded")
        with st.expander("View uploaded file list", expanded=False):
            for f in uploaded_files:
                st.write(f"• {f.name}  ({f.size / 1024:.1f} KB)")

        col1, col2 = st.columns([1, 4])
        with col1:
            start_clicked = st.button("🚀 Start Conversion", type="primary")

        if start_clicked:
            global TEXT_DENSITY_THRESHOLD
            st.session_state["ocr_dpi"] = ocr_dpi

            st.subheader("🔄 Processing status")
            overall_progress_bar = st.progress(0, text="Overall progress: 0/0 files")
            status_container = st.container()

            global extract_ocr_pdf
            original_ocr_fn = extract_ocr_pdf

            def _ocr_with_dpi(pdf_bytes, filename, dpi=ocr_dpi):
                return original_ocr_fn(pdf_bytes, filename, dpi=dpi)
            extract_ocr_pdf = _ocr_with_dpi

            with st.spinner("Converting PDFs to Excel... this may take a while for scanned documents."):
                results = process_multiple_pdfs(uploaded_files, overall_progress_bar, status_container, layout_mode)

            extract_ocr_pdf = original_ocr_fn  
            st.session_state["results"] = results
            st.success("🎉 Batch processing complete!")


    results: List[FileResult] = st.session_state.get("results", [])

    if results:
        st.markdown("---")
        st.subheader("📥 Download Results")

        done_results = [r for r in results if r.status == "done"]
        error_results = [r for r in results if r.status == "error"]

        summary_cols = st.columns(3)
        summary_cols[0].metric("Total files", len(results))
        summary_cols[1].metric("Succeeded", len(done_results))
        summary_cols[2].metric("Failed", len(error_results))

        if done_results:
            st.markdown("#### Batch Actions")
            colA, colB = st.columns(2)
            
            zip_bytes = build_zip_of_excels(done_results)
            colA.download_button(
                label=f"⬇️ Download {len(done_results)} files as ZIP",
                data=zip_bytes,
                file_name="converted_excels.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )
            
            master_tables = []
            for r in done_results:
                for t in r.tables:
                    t_copy = t.copy()
                    if "Source PDF" not in t_copy.columns:
                        t_copy.insert(0, "Source PDF", r.filename)
                    master_tables.append(t_copy)
                    
            if master_tables:
                master_excel_bytes, _, _ = build_excel_workbook(master_tables, layout_mode="single")
                colB.download_button(
                    label="⬇️ Download ONE Master Excel (All PDFs Combined)",
                    data=master_excel_bytes,
                    file_name="Master_Combined_Data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="secondary",
                    use_container_width=True,
                )

        st.markdown("#### Individual files")
        for r in results:
            with st.expander(f"{'✅' if r.status == 'done' else '❌'} {r.filename}"):
                if r.status == "error":
                    st.error(f"Processing failed: {r.error_message}")
                    continue

                meta_cols = st.columns(4)
                meta_cols[0].write(f"**Type detected:** `{r.pdf_type}`")
                meta_cols[1].write(f"**Pages:** {r.num_pages}")
                meta_cols[2].write(f"**Tables found:** {r.num_tables_found}")
                meta_cols[3].write(f"**Sheets/Tables post-merge:** {r.num_tables_after_merge}")

                if r.excel_bytes:
                    base_name = os.path.splitext(r.filename)[0]
                    st.download_button(
                        label=f"⬇️ Download {base_name}.xlsx",
                        data=r.excel_bytes,
                        file_name=f"{base_name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_{r.filename}",
                    )

                if r.preview_frames:
                    st.markdown("**Preview:**")
                    for sheet_name, preview_df in r.preview_frames.items():
                        st.caption(f"Sheet / View: {sheet_name}")
                        st.dataframe(preview_df, use_container_width=True)
    else:
        st.info("Upload PDF files above and click **Start Conversion** to begin.")


if __name__ == "__main__":
    main()
