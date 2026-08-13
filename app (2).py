"""
==================================================================================
 Smart Multi-PDF to Excel Converter (Separate Excel per PDF)
==================================================================================
A production-grade Streamlit application that:
  - Accepts 100+ PDF uploads at once
  - Detects whether each PDF is text-based or scanned (image-based)
  - Extracts tables using pdfplumber/camelot (text PDFs) or OCR (scanned PDFs)
  - Detects multi-page table continuity and merges continued tables
  - Cleans extracted data (empty rows/cols, duplicate headers, column alignment)
  - Produces ONE Excel workbook per PDF (in-memory, no disk dependency)
  - Offers individual downloads + a single "download all as ZIP" option
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

# camelot is optional at import time -- some environments (Windows without
# Ghostscript) may fail to import it. We degrade gracefully to pdfplumber-only
# extraction if that happens.
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

# Minimum characters of extractable text per page before we consider a PDF
# "text-based" rather than "scanned"
TEXT_DENSITY_THRESHOLD = 40

# Similarity thresholds used when deciding whether two tables (typically on
# consecutive pages) are actually one logical table split by a page break.
HEADER_SIMILARITY_THRESHOLD = 78   # rapidfuzz ratio (0-100)
COLUMN_COUNT_TOLERANCE = 0         # allowed difference in column count
DTYPE_SIMILARITY_THRESHOLD = 0.6   # fraction of columns whose inferred dtype matches

EXCEL_SHEET_NAME_MAX_LEN = 31  # hard Excel limit


# ==================================================================================
# DATA STRUCTURES
# ==================================================================================

@dataclass
class ExtractedTable:
    """A single raw table pulled from a PDF, tagged with provenance."""
    dataframe: pd.DataFrame
    page_number: int
    source: str  # "camelot", "pdfplumber", "ocr"


@dataclass
class FileResult:
    """Holds the outcome of processing a single uploaded PDF."""
    filename: str
    pdf_type: str = "unknown"          # "text" or "scanned"
    status: str = "pending"            # pending / processing / done / error
    error_message: str = ""
    sheet_names: List[str] = field(default_factory=list)
    preview_frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    excel_bytes: Optional[bytes] = None
    num_pages: int = 0
    num_tables_found: int = 0
    num_tables_after_merge: int = 0


# ==================================================================================
# STEP 1: PDF TYPE DETECTION
# ==================================================================================

def detect_pdf_type(pdf_bytes: bytes) -> Tuple[str, int]:
    """
    Determine whether a PDF is text-based or scanned (image-based).

    Strategy: open with pdfplumber and measure the amount of extractable
    text per page. If the average extractable text per page is below a
    threshold, we assume the PDF is scanned/image-based and route it to OCR.

    Returns:
        (pdf_type, num_pages) where pdf_type is "text" or "scanned"
    """
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            num_pages = len(pdf.pages)
            if num_pages == 0:
                return "scanned", 0

            total_chars = 0
            pages_sampled = min(num_pages, 5)  # sample first few pages for speed
            for page in pdf.pages[:pages_sampled]:
                text = page.extract_text() or ""
                total_chars += len(text.strip())

            avg_chars_per_page = total_chars / max(pages_sampled, 1)

            if avg_chars_per_page >= TEXT_DENSITY_THRESHOLD:
                return "text", num_pages
            else:
                return "scanned", num_pages
    except Exception:
        # If pdfplumber can't even open it, fall back to OCR path which
        # rasterizes the PDF regardless of internal structure.
        return "scanned", 0


# ==================================================================================
# STEP 2A: TEXT-BASED EXTRACTION (pdfplumber + camelot)
# ==================================================================================

def extract_text_pdf(pdf_bytes: bytes, filename: str) -> List[ExtractedTable]:
    """
    Extract tables from a text-based PDF using camelot first (better table
    structure detection for ruled/lattice tables), falling back to
    pdfplumber (works well for whitespace/stream-style tables) on pages
    where camelot finds nothing.
    """
    extracted: List[ExtractedTable] = []
    camelot_pages_with_tables = set()

    # --- Try camelot first (needs a real file path, so use a temp file) ---
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
                # If lattice already found tables, skip stream to avoid duplicates
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

    # --- pdfplumber fallback / complement for pages camelot missed ---
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                if i in camelot_pages_with_tables:
                    continue  # already captured by camelot
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
                    # No detected table grid -- fall back to structured text:
                    # split lines on multiple spaces/tabs to approximate columns.
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
    """
    Clean up a page image before OCR to improve accuracy:
      - Convert to grayscale
      - Denoise
      - Adaptive thresholding (binarization)
      - Slight dilation to make text more solid for Tesseract
    """
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


def _words_to_table(ocr_data: Dict[str, list], y_tolerance: int = 12) -> pd.DataFrame:
    """
    Reconstruct a pseudo-table from pytesseract's word-level bounding boxes.
    Words are grouped into rows by vertical (top) proximity, then ordered
    left-to-right within each row to approximate columns.
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
        words.append({
            "text": text,
            "left": ocr_data["left"][i],
            "top": ocr_data["top"][i],
        })

    if not words:
        return pd.DataFrame()

    words.sort(key=lambda w: w["top"])

    rows = []
    current_row = [words[0]]
    current_top = words[0]["top"]

    for w in words[1:]:
        if abs(w["top"] - current_top) <= y_tolerance:
            current_row.append(w)
        else:
            rows.append(current_row)
            current_row = [w]
            current_top = w["top"]
    rows.append(current_row)

    table_rows = []
    for row in rows:
        row_sorted = sorted(row, key=lambda w: w["left"])
        table_rows.append([w["text"] for w in row_sorted])

    max_cols = max(len(r) for r in table_rows)
    table_rows = [r + [""] * (max_cols - len(r)) for r in table_rows]

    return pd.DataFrame(table_rows)


def extract_ocr_pdf(pdf_bytes: bytes, filename: str, dpi: int = 200) -> List[ExtractedTable]:
    """
    Extract structured tabular data from a scanned PDF via OCR:
      1. Rasterize each page with pdf2image (requires Poppler)
      2. Preprocess each page image with OpenCV
      3. Run pytesseract in "data" mode to get word bounding boxes
      4. Reconstruct rows/columns from those bounding boxes
    """
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
                processed, output_type=pytesseract.Output.DICT
            )
            df = _words_to_table(ocr_data)
            if not df.empty:
                extracted.append(ExtractedTable(df, i, "ocr"))
        except Exception:
            # Skip pages that fail OCR but keep processing the rest
            continue

    return extracted


# ==================================================================================
# STEP 3: MULTI-PAGE TABLE CONTINUITY DETECTION
# ==================================================================================

def _infer_column_dtype_signature(series: pd.Series) -> str:
    """Cheap heuristic dtype classification per column: numeric / date / text."""
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
    """Fraction of corresponding columns whose inferred dtype signature matches."""
    n = min(len(df1.columns), len(df2.columns))
    if n == 0:
        return 0.0
    matches = 0
    for c in range(n):
        sig1 = _infer_column_dtype_signature(df1.iloc[:, c])
        sig2 = _infer_column_dtype_signature(df2.iloc[:, c])
        if sig1 == sig2:
            matches += 1
    return matches / n


def _header_similarity(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Average fuzzy-match ratio between the two tables' first (header) rows."""
    header1 = [str(x) for x in df1.iloc[0].tolist()] if len(df1) else []
    header2 = [str(x) for x in df2.iloc[0].tolist()] if len(df2) else []
    if not header1 or not header2:
        return 0.0
    n = min(len(header1), len(header2))
    scores = [fuzz.ratio(header1[i], header2[i]) for i in range(n)]
    return sum(scores) / len(scores) if scores else 0.0


def are_tables_similar(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """
    Decide whether two tables (usually from consecutive pages) are actually
    a single logical table that was split across a page break.

    Compares:
      - Column count (must match, within tolerance)
      - Header similarity (fuzzy string match via rapidfuzz)
      - Data type pattern similarity per column
      - General row/column consistency
    """
    if df1.empty or df2.empty:
        return False

    col_diff = abs(len(df1.columns) - len(df2.columns))
    if col_diff > COLUMN_COUNT_TOLERANCE:
        return False

    header_score = _header_similarity(df1, df2)
    dtype_score = _dtype_signature_similarity(df1, df2)

    # Two ways to qualify as "the same table":
    #  (a) headers look alike (repeated header row on the new page), or
    #  (b) headers don't match (no repeated header) but the data pattern
    #      (column dtypes) lines up well, implying continuous data rows.
    if header_score >= HEADER_SIMILARITY_THRESHOLD:
        return True
    if dtype_score >= DTYPE_SIMILARITY_THRESHOLD and header_score >= 40:
        return True

    return False


def merge_tables(tables: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge a list of tables believed to be one logical table split across
    pages. Removes repeated header rows that appear again on later pages.
    """
    if not tables:
        return pd.DataFrame()
    if len(tables) == 1:
        return tables[0].reset_index(drop=True)

    base_header = [str(x).strip().lower() for x in tables[0].iloc[0].tolist()]
    merged_frames = [tables[0]]

    for t in tables[1:]:
        if t.empty:
            continue
        first_row = [str(x).strip().lower() for x in t.iloc[0].tolist()]
        # If this table's first row matches the base table's header, drop it
        # (it's a repeated header, not real data) before appending.
        n = min(len(first_row), len(base_header))
        if n and fuzz.ratio(" ".join(first_row[:n]), " ".join(base_header[:n])) >= HEADER_SIMILARITY_THRESHOLD:
            merged_frames.append(t.iloc[1:])
        else:
            merged_frames.append(t)

    merged = pd.concat(merged_frames, ignore_index=True, sort=False)
    return merged


def group_and_merge_tables(extracted_tables: List[ExtractedTable]) -> List[pd.DataFrame]:
    """
    Walk through extracted tables in page order and greedily merge any
    that appear to be continuations of one another. Returns a list of
    final logical tables (one per output sheet).
    """
    if not extracted_tables:
        return []

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
    """
    Clean a single extracted table:
      - Drop fully empty rows and columns
      - Normalize whitespace in text cells
      - Promote the first row to header when it looks like a header
      - Remove duplicate header rows left over from merges
      - Align/pad inconsistent row lengths
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Normalize all cell text: strip whitespace, collapse internal spaces
    df = df.astype(str).apply(
        lambda col: col.str.replace(r"\s+", " ", regex=True).str.strip()
    )

    # Replace empty-string cells with NaN so dropna works correctly
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Drop fully empty rows/columns
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    if df.empty:
        return df

    df = df.reset_index(drop=True)

    # Promote first row to header if it doesn't look like a data row
    # (heuristic: header row usually has more distinct/non-numeric values)
    first_row = df.iloc[0]
    non_null_ratio = first_row.notna().mean()
    if non_null_ratio > 0.5:
        df.columns = [str(c) if pd.notna(c) else f"col_{i}" for i, c in enumerate(first_row)]
        df = df.iloc[1:].reset_index(drop=True)
    else:
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

    # Deduplicate column names (Excel/pandas requires uniqueness)
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

    # Remove any remaining rows that exactly duplicate the header text
    # (leftover repeated headers from page breaks)
    header_lower = [str(c).strip().lower() for c in df.columns]

    def _is_header_dup(row):
        vals = [str(v).strip().lower() for v in row.tolist()]
        n = min(len(vals), len(header_lower))
        if n == 0:
            return False
        return fuzz.ratio(" ".join(vals[:n]), " ".join(header_lower[:n])) >= HEADER_SIMILARITY_THRESHOLD

    if len(df) > 0:
        dup_mask = df.apply(_is_header_dup, axis=1)
        df = df[~dup_mask]

    # Drop exact duplicate rows and fully empty rows again post-cleanup
    df = df.drop_duplicates(keep="first")
    df = df.dropna(axis=0, how="all")
    df = df.reset_index(drop=True)

    return df


# ==================================================================================
# EXCEL WORKBOOK GENERATION
# ==================================================================================

def _safe_sheet_name(name: str, used_names: set) -> str:
    """Sanitize and de-duplicate an Excel sheet name (max 31 chars, no bad chars)."""
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


def build_excel_workbook(tables: List[pd.DataFrame]) -> Tuple[bytes, List[str], Dict[str, pd.DataFrame]]:
    """
    Write a list of cleaned tables into a single in-memory Excel workbook,
    one sheet per table. Returns the workbook bytes, the sheet names used,
    and a preview dict (sheet_name -> first rows) for the UI.
    """
    output = io.BytesIO()
    used_names: set = set()
    sheet_names: List[str] = []
    previews: Dict[str, pd.DataFrame] = {}

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if not tables:
            pd.DataFrame({"Notice": ["No tables could be extracted from this PDF."]}).to_excel(
                writer, sheet_name="Sheet1", index=False
            )
            sheet_names.append("Sheet1")
        else:
            for idx, df in enumerate(tables, start=1):
                if df.empty:
                    continue
                sheet_name = _safe_sheet_name(f"Table_{idx}", used_names)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                sheet_names.append(sheet_name)
                previews[sheet_name] = df.head(5)

    return output.getvalue(), sheet_names, previews


# ==================================================================================
# STEP 5: SINGLE-PDF PIPELINE
# ==================================================================================

def process_single_pdf(filename: str, pdf_bytes: bytes) -> FileResult:
    """
    Run the full pipeline on one PDF: detect type -> extract -> merge
    continued tables -> clean -> build Excel workbook.
    """
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

    result.num_tables_after_merge = len(cleaned_tables)

    excel_bytes, sheet_names, previews = build_excel_workbook(cleaned_tables)
    result.excel_bytes = excel_bytes
    result.sheet_names = sheet_names
    result.preview_frames = previews
    result.status = "done"

    return result


# ==================================================================================
# STEP 6: BATCH PIPELINE (MULTIPLE PDFs)
# ==================================================================================

def process_multiple_pdfs(
    uploaded_files: list,
    overall_progress_bar,
    status_container,
) -> List[FileResult]:
    """
    Process a batch of uploaded PDFs sequentially, updating a shared
    progress bar and a live status area. Errors in one file are caught
    and recorded without stopping the batch.
    """
    results: List[FileResult] = []
    total = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files, start=1):
        filename = uploaded_file.name
        row = status_container.container()
        row_placeholder = row.empty()
        row_placeholder.info(f"⏳ Processing **{filename}** ({i}/{total})...")

        try:
            pdf_bytes = uploaded_file.getvalue()
            result = process_single_pdf(filename, pdf_bytes)
            results.append(result)
            row_placeholder.success(
                f"✅ **{filename}** — type: `{result.pdf_type}` — "
                f"{result.num_tables_after_merge} sheet(s) generated "
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
    """Bundle every successfully-generated Excel file into a single ZIP (in-memory)."""
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
        "its own Excel workbook, with multi-page tables intelligently merged."
    )

    if not CAMELOT_AVAILABLE:
        st.warning(
            "⚠️ `camelot-py` (or its Ghostscript dependency) is not available in this "
            "environment. Text-based extraction will fall back to pdfplumber only, "
            "which still works but may be slightly less precise for ruled tables."
        )

    with st.sidebar:
        st.header("⚙️ Settings")
        ocr_dpi = st.slider("OCR rasterization DPI", 100, 300, 200, step=25,
                             help="Higher DPI improves OCR accuracy but is slower.")
        st.markdown("---")
        st.markdown(
            "**Tech stack:** pdfplumber, camelot, pytesseract, pdf2image, "
            "OpenCV, rapidfuzz, openpyxl"
        )

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

            # Monkeypatch dpi into extract_ocr_pdf via a wrapper so the sidebar
            # setting is respected without changing the function signature.
            global extract_ocr_pdf
            original_ocr_fn = extract_ocr_pdf

            def _ocr_with_dpi(pdf_bytes, filename, dpi=ocr_dpi):
                return original_ocr_fn(pdf_bytes, filename, dpi=dpi)

            extract_ocr_pdf = _ocr_with_dpi

            with st.spinner("Converting PDFs to Excel... this may take a while for large batches."):
                results = process_multiple_pdfs(uploaded_files, overall_progress_bar, status_container)

            extract_ocr_pdf = original_ocr_fn  # restore

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
            zip_bytes = build_zip_of_excels(done_results)
            st.download_button(
                label=f"⬇️ Download ALL as ZIP ({len(done_results)} files)",
                data=zip_bytes,
                file_name="converted_excels.zip",
                mime="application/zip",
                type="primary",
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
                meta_cols[3].write(f"**Sheets after merge:** {r.num_tables_after_merge}")

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
                    st.markdown("**Preview (first rows of each sheet):**")
                    for sheet_name, preview_df in r.preview_frames.items():
                        st.caption(f"Sheet: {sheet_name}")
                        st.dataframe(preview_df, use_container_width=True)
    else:
        st.info("Upload PDF files above and click **Start Conversion** to begin.")


if __name__ == "__main__":
    main()
