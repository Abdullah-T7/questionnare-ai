import io
import re
from typing import BinaryIO
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .answer import answer_question

QUESTION_KEYWORDS = ["question", "requirement", "control", "item", "security requirement", "inquiry", "prompt"]
ANSWER_KEYWORDS = ["answer", "response", "vendor response", "vendor answer", "compliance status", "reply"]
CITATION_KEYWORDS = ["citation", "source", "policy reference", "evidence", "source document"]

# Styling for human-review escalation
EXPERT_FILL = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")  # Soft Amber
DRAFT_FILL = PatternFill(start_color="E8F5E9", end_color="E8F5E9", fill_type="solid")   # Soft Green
HEADER_FILL = PatternFill(start_color="2A4365", end_color="2A4365", fill_type="solid")  # Deep Blue
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")


def _normalize(val: any) -> str:
    if val is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(val).lower())


def detect_columns(sheet) -> tuple[int | None, int | None, int | None, int]:
    """
    Scans the top 10 rows to detect:
    (question_col, answer_col, citation_col, header_row)
    1-indexed for openpyxl.
    """
    for r in range(1, min(sheet.max_row + 1, 11)):
        row_cells = [sheet.cell(row=r, column=c) for c in range(1, sheet.max_column + 1)]
        row_vals = [_normalize(cell.value) for cell in row_cells]

        q_col = None
        a_col = None
        c_col = None

        for idx, val in enumerate(row_vals, start=1):
            if any(k in val for k in QUESTION_KEYWORDS) and q_col is None:
                q_col = idx
            elif any(k in val for k in ANSWER_KEYWORDS) and a_col is None:
                a_col = idx
            elif any(k in val for k in CITATION_KEYWORDS) and c_col is None:
                c_col = idx

        if q_col is not None:
            return q_col, a_col, c_col, r

    # Fallback default: Col 1 is Question, Col 2 is Answer
    return 1, 2, None, 1


def process_workbook(file_stream: BinaryIO | bytes) -> tuple[bytes, dict]:
    """
    Reads an Excel file, processes questions through LangGraph, fills answers
    and citations while preserving all existing sheets, formatting, and formulas.
    """
    if isinstance(file_stream, bytes):
        file_stream = io.BytesIO(file_stream)

    wb = openpyxl.load_workbook(file_stream)
    
    total_processed = 0
    total_drafted = 0
    total_needs_expert = 0

    for sheet in wb.worksheets:
        if sheet.max_row < 2:
            continue

        q_col, a_col, c_col, header_row = detect_columns(sheet)
        if q_col is None:
            continue

        # If answer column was not found, create one next to max column
        if a_col is None:
            a_col = sheet.max_column + 1
            header_cell = sheet.cell(row=header_row, column=a_col, value="AI Drafted Response")
            header_cell.fill = HEADER_FILL
            header_cell.font = HEADER_FONT
            sheet.column_dimensions[get_column_letter(a_col)].width = 50

        # If citation column doesn't exist, create one
        if c_col is None:
            c_col = max(sheet.max_column + 1, a_col + 1)
            header_cell = sheet.cell(row=header_row, column=c_col, value="Source Citations / Review Role")
            header_cell.fill = HEADER_FILL
            header_cell.font = HEADER_FONT
            sheet.column_dimensions[get_column_letter(c_col)].width = 40

        for r in range(header_row + 1, sheet.max_row + 1):
            q_val = sheet.cell(row=r, column=q_col).value
            if not q_val or str(q_val).strip() == "":
                continue

            q_text = str(q_val).strip()
            # Skip rows that appear to be section subheaders
            if len(q_text) < 5 or q_text.startswith("==="):
                continue

            total_processed += 1
            result = answer_question(q_text)

            ans_cell = sheet.cell(row=r, column=a_col)
            cite_cell = sheet.cell(row=r, column=c_col)

            ans_cell.alignment = Alignment(wrap_text=True, vertical="top")
            cite_cell.alignment = Alignment(wrap_text=True, vertical="top")

            if result["status"] == "drafted":
                total_drafted += 1
                ans_cell.value = result["answer"]
                ans_cell.fill = DRAFT_FILL
                
                citations_text = "; ".join(
                    f"{c['document']} ({c['section']})" for c in result.get("citations", [])
                )
                cite_cell.value = f"Confidence: {int(result['confidence'] * 100)}% | Sources: {citations_text}"
            else:
                total_needs_expert += 1
                role = result.get("expert_role", "Security Team")
                reason = result.get("expert_reason", "Policy information not found.")
                ans_cell.value = f"[NEEDS EXPERT REVIEW]"
                ans_cell.fill = EXPERT_FILL
                cite_cell.value = f"Assignee: {role}\nReason: {reason}"
                cite_cell.fill = EXPERT_FILL

    out = io.BytesIO()
    wb.save(out)
    out.seek(0)

    stats = {
        "total_questions": total_processed,
        "drafted": total_drafted,
        "needs_expert": total_needs_expert,
        "automation_rate": f"{(total_drafted / total_processed * 100):.1f}%" if total_processed else "0%",
    }
    return out.getvalue(), stats

