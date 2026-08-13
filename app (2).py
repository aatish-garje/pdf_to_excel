"""
==================================================================================
 Smart Multi-PDF to Excel Converter (Separate Excel per PDF & Master Export)
==================================================================================
A production-grade Streamlit application that:
  - Accepts 100+ PDF uploads at once
  - Detects whether each PDF is text-based or scanned (image-based)
  - Extracts tables using pdfplumber/camelot (text PDFs) or OCR (scanned PDFs)
  - Detects multi-page table continuity and merges continued tables
  - Cleans extracted data (empty rows/cols, duplicate headers, column alignment)
  - Allows exporting tables into a Single Sheet or Multiple Sheets
  - Produces Excel workbooks per PDF + an optional Master Excel for all PDFs
  - Handles errors per-file so one bad PDF never stops the batch

Run with:  streamlit run app.py
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

# camelot is optional at import time
try:
    import camelot
    CAMELOT_AVAILABLE = True
except Exception:
    CAMELOT_AVAILABLE = False


# ==================================================================================
# CONFIGURATION
# ==================================================================================

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


# ==================================================================================
# DATA STRUCTURES
# ==================================================================================

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
    tables: List[pd.DataFrame] = field(default_factory=list)  # Save extracted tables for master excel
    num_pages: int = 0
    num_tables_found: int = 0
    num_tables_after_merge: int = 0


# ==================================================================================
# STEP 1: PDF TYPE DETECTION
# ==================================================================================

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


# ==================================================================================
# STEP 2A: TEXT-BASED EXTRACTION
# ==================================================================================

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


# ==================================================================================
# STEP 2B: SCANNED PDF EXTRACTION (OCR)
# ==================================================================================

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


def _words_to_table(ocr_data: Dict[str, list]) -> List[pd.DataFrame]:
    """
    Reconstruct tables from pytesseract's word-level bounding boxes.
    Uses Block-wise Spatial Projection to handle pages with multiple 
    differing table layouts (e.g., Header Key-Values vs Main Data Grid).
    """
    words = []
    n = len(ocr_data["text"])
    
    for i in range(n):
        text = ocr_data["text"][i].strip()
        if not text:
            continue
        conf = ocr_data.get("conf", ["0"] * n)[i]
        try:
            conf_val = float(conf)
        except (ValueError, TypeError):
            conf_val = -1
        if conf_val != -1 and conf_val < 30:
            continue  # skip very low-confidence noise
            
        left = ocr_data["left"][i]
        top = ocr_data["top"][i]
        width = ocr_data["width"][i]
        height = ocr_data["height"][i]
            
        words.append({
            "text": text,
            "left": left,
            "right": left + width,
            "top": top,
            "bottom": top + height,
            "center_x": left + (width / 2.0),
            "center_y": top + (height / 2.0),
            "width": width,
            "height": height
        })

    if not words:
        return []

    # Filter out purely structural grid characters
    data_words = [w for w in words if not re.fullmatch(r'[|_\-]', w["text"])]
    if not data_words:
        data_words = words

    # Filter out extreme outliers (like QR codes or noise)
    heights = sorted([w["height"] for w in data_words])
    median_height = heights[len(heights) // 2]
    
    cleaned_words = []
    for w in data_words:
        if 0.4 * median_height <= w["height"] <= 3.5 * median_height:
            cleaned_words.append(w)
            
    if not cleaned_words:
        cleaned_words = data_words

    # 1. Group into rows via dynamic Y tolerance
    avg_height = sum(w["height"] for w in cleaned_words) / len(cleaned_words)
    dynamic_y_tol = max(avg_height * 0.4, 4) 
    
    cleaned_words.sort(key=lambda w: w["center_y"])
    
    rows = []
    current_row = [cleaned_words[0]]
    current_y = cleaned_words[0]["center_y"]

    for w in cleaned_words[1:]:
        if abs(w["center_y"] - current_y) <= dynamic_y_tol:
            current_row.append(w)
            current_y = sum(x["center_y"] for x in current_row) / len(current_row)
        else:
            rows.append(current_row)
            current_row = [w]
            current_y = w["center_y"]
    if current_row:
        rows.append(current_row)

    # 2. Group rows into layout Blocks (Zones) based on vertical spacing
    # If the gap between lines is significantly large, it's a new table layout.
    blocks = []
    current_block = [rows[0]]
    
    for i in range(1, len(rows)):
        prev_row = rows[i-1]
        curr_row = rows[i]
        
        prev_y = sum(w["center_y"] for w in prev_row) / len(prev_row)
        curr_y = sum(w["center_y"] for w in curr_row) / len(curr_row)
        gap_y = curr_y - prev_y
        
        # A gap larger than ~2.2x font height means a major layout break
        if gap_y > avg_height * 2.2:
            blocks.append(current_block)
            current_block = [curr_row]
        else:
            current_block.append(curr_row)
            
    if current_block:
        blocks.append(current_block)

    # 3. Process each Block independently to find its exact columns
    dfs = []
    for block_rows in blocks:
        block_words = [w for row in block_rows for w in row]
        if not block_words:
            continue
            
        block_max_right = max(w["right"] for w in block_words)
        
        # X-axis projection for THIS BLOCK ONLY
        dilation = int(max(avg_height * 0.25, 2))
        x_profile = np.zeros(int(block_max_right) + dilation + 2)
        
        for w in block_words:
            start = max(0, w["left"] - dilation)
            end = min(len(x_profile) - 1, w["right"] + dilation)
            x_profile[start:end] += 1
            
        raw_cols = []
        in_col = False
        start_idx = 0
        for i, val in enumerate(x_profile):
            if val > 0 and not in_col:
                in_col = True
                start_idx = i
            elif val == 0 and in_col:
                in_col = False
                raw_cols.append((start_idx, i))
        if in_col:
            raw_cols.append((start_idx, len(x_profile)))
            
        if not raw_cols:
            continue

        cols = [raw_cols[0]]
        for i in range(1, len(raw_cols)):
            prev_start, prev_end = cols[-1]
            curr_start, curr_end = raw_cols[i]
            gap = curr_start - prev_end
            
            # Bridge internal spaces but preserve true column gaps
            if gap <= max(avg_height * 0.35, 4):
                cols[-1] = (prev_start, curr_end)
            else:
                cols.append((curr_start, curr_end))

        # 4. Map words to the precise columns detected
        table_rows = []
        for row in block_rows:
            row_data = [""] * len(cols)
            row.sort(key=lambda w: w["left"])
            
            for w in row:
                word_center_x = w["center_x"]
                assigned_col_idx = -1
                
                # Strict containment
                for idx, (c_start, c_end) in enumerate(cols):
                    if c_start <= word_center_x <= c_end:
                        assigned_col_idx = idx
                        break
                        
                # Closest column fallback
                if assigned_col_idx == -1:
                    distances = []
                    for idx, (c_start, c_end) in enumerate(cols):
                        if word_center_x < c_start:
                            distances.append((idx, c_start - word_center_x))
                        else:
                            distances.append((idx, word_center_x - c_end))
                    assigned_col_idx = min(distances, key=lambda x: x[1])[0]
                
                if row_data[assigned_col_idx] == "":
                    row_data[assigned_col_idx] = w["text"]
                else:
                    row_data[assigned_col_idx] += " " + w["text"]
                    
            table_rows.append(row_data)

        df = pd.DataFrame(table_rows)
        
        # Cleanup block
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df.dropna(how="all", axis=1, inplace=True)
        df.fillna("", inplace=True)
        
        if not df.empty and len(df.columns) > 0:
            df.columns = [str(i) for i in range(len(df.columns))]
            dfs.append(df)

    return dfs


def extract_ocr_pdf(pdf_bytes: bytes, filename: str, dpi: int = 200) -> List[ExtractedTable]:
    extracted: List[ExtractedTable] = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    except Exception as e:
        raise RuntimeError(f"Failed to rasterize PDF for OCR: {e}")

    for i, pil_page in enumerate(pages, start=1):
        try:
            img = np.array(pil_page.convert("RGB"))
            processed = preprocess_image(img)
            
            ocr_data = pytesseract.image_to_data(
                processed, output_type=pytesseract.Output.DICT, config="--psm 6"
            )
            
            # Map each block layout into its own standalone table mapping
            dfs = _words_to_table(ocr_data)
            for df in dfs:
                if not df.empty:
                    extracted.append(ExtractedTable(df, i, "ocr"))
        except Exception:
            continue

    return extracted


# ==================================================================================
# STEP 3: CONTINUITY DETECTION & MERGING
# ==================================================================================

def _infer_column_dtype_signature(series: pd.Series) -> str:
    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample != ""].head(20)
    if sample.empty:
        return "empty"
    numeric_count = sum(bool(re.fullmatch(r"-?\d+(\.\d+)?%?", v)) for v in sample)
    date_count = sum(bool(re.fullmatch(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", v)) for v in sample)
    if numeric_count / len(sample) >= 0.6:
        return "numeric"
    if date_count / len(sample) >= 0.6:
        return "date"
    return "text"

def _dtype_signature_similarity(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    n = min(len(df1.columns), len(df2.columns))
    if n == 0: return 0.0
    matches = 0
    for c in range(n):
        sig1 = _infer_column_dtype_signature(df1.iloc[:, c])
        sig2 = _infer_column_dtype_signature(df2.iloc[:, c])
        if sig1 == sig2:
            matches += 1
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
    col_diff = abs(len(df1.columns) - len(df2.columns))
    if col_diff > COLUMN_COUNT_TOLERANCE: return False
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
        last_group = groups[-1]
        last_df = last_group[-1]
        if are_tables_similar(last_df, et.dataframe):
            last_group.append(et.dataframe)
        else:
            groups.append([et.dataframe])

    return [merge_tables(g) for g in groups]


# ==================================================================================
# STEP 4: DATA CLEANING
# ==================================================================================

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

    return df


# ==================================================================================
# EXCEL WORKBOOK GENERATION (Single Sheet vs Multi Sheet)
# ==================================================================================

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
    Supports single sheet (stacked tables) or multiple sheets (one table per tab).
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
                    if df.empty:
                        continue
                    if current_row == 0:
                        previews[sheet_name] = df.head(5)
                        
                    # Add a visual separator/label row for clarity between tables
                    label_df = pd.DataFrame(columns=[f"--- Table {idx+1} ---"])
                    label_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False)
                    
                    # Write the actual data underneath
                    df.to_excel(writer, sheet_name=sheet_name, startrow=current_row + 1, index=False)
                    
                    # Update current_row to position the next table (Data Length + Header + Label + Empty Gap)
                    current_row += len(df) + 4
            else:
                # Traditional Mode: One Table per Sheet
                for idx, df in enumerate(tables, start=1):
                    if df.empty:
                        continue
                    sheet_name = _safe_sheet_name(f"Table_{idx}", used_names)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    sheet_names.append(sheet_name)
                    previews[sheet_name] = df.head(5)

    return output.getvalue(), sheet_names, previews


# ==================================================================================
# STEP 5 & 6: PIPELINE EXECUTORS
# ==================================================================================

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
    
    result.tables = cleaned_tables  # Store for master-excel batching
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


# ==================================================================================
# STREAMLIT UI
# ==================================================================================

def main():
    st.title("📊 Multi-PDF to Excel Converter")
    st.caption(
        "Upload multiple PDFs (text-based or scanned). Each PDF is converted into "
        "an Excel workbook, with multi-page tables intelligently merged."
    )

    if not CAMELOT_AVAILABLE:
        st.warning(
            "⚠️ `camelot-py` (or its Ghostscript dependency) is not available in this "
            "environment. Text-based extraction will fall back to pdfplumber only."
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
            help="Choose whether to place each extracted table on its own Excel tab, or stack them all on a single tab with spacing."
        )
        layout_mode = "single" if "Single" in sheet_layout else "multi"

        st.markdown("---")
        ocr_dpi = st.slider("OCR rasterization DPI", 100, 300, 200, step=25,
                             help="Higher DPI improves OCR accuracy but is slower.")
        
        st.markdown("---")
        st.markdown("**Tech stack:** pdfplumber, camelot, pytesseract, pdf2image, OpenCV, rapidfuzz, openpyxl")

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

            with st.spinner("Converting PDFs to Excel... this may take a while for large batches."):
                results = process_multiple_pdfs(uploaded_files, overall_progress_bar, status_container, layout_mode)

            extract_ocr_pdf = original_ocr_fn  
            st.session_state["results"] = results
            st.success("🎉 Batch processing complete!")

    # ------------------------------------------------------------------
    # RESULTS / DOWNLOADS
    # ------------------------------------------------------------------
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
            
            # Action 1: ZIP of individual files
            zip_bytes = build_zip_of_excels(done_results)
            colA.download_button(
                label=f"⬇️ Download {len(done_results)} files as ZIP",
                data=zip_bytes,
                file_name="converted_excels.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )
            
            # Action 2: Massive master combined file
            master_tables = []
            for r in done_results:
                for t in r.tables:
                    t_copy = t.copy()
                    if "Source PDF" not in t_copy.columns:
                        t_copy.insert(0, "Source PDF", r.filename) # Insert origin filename
                    master_tables.append(t_copy)
                    
            if master_tables:
                # Master file is typically best dumped as a single stacked sheet
                master_excel_bytes, _, _ = build_excel_workbook(master_tables, layout_mode="single")
                colB.download_button(
                    label="⬇️ Download ONE Master Excel (All PDFs Combined)",
                    data=master_excel_bytes,
                    file_name="Master_Combined_Data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="secondary",
                    use_container_width=True,
                    help="Combine every table from every PDF into a single Excel file."
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
