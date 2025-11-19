from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class XLSXParseError(Exception):
    pass


def parse_xlsx_tables(xlsx_path: Path, max_rows: int = 5000) -> list[dict[str, Any]]:
    """
    Return a list of tables with normalized columns.
    [
      {"name":"Sheet1!A1", "rows":[{"date":"2025-01-01","amount":123.45,...}, ...]},
      ...
    ]
    """
    if not xlsx_path.exists():
        raise XLSXParseError(f"file not found: {xlsx_path}")

    try:
        xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
        tables: list[dict[str, Any]] = []

        for sheet in xl.sheet_names:
            df = xl.parse(sheet, dtype=str).head(max_rows)  # read as strings first
            if df.empty:
                continue

            # basic cleaning: drop all-empty cols/rows, trim, lower headers
            df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
            df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

            # try to coerce common fields if present
            for col in list(df.columns):
                if "date" in col and df[col].notna().any():
                    # let Mongo store as strings for now; conversion happens in validation step
                    pass
                if "amount" in col or "debit" in col or "credit" in col or "balance" in col:
                    # try numeric where possible, else keep string
                    df[col] = pd.to_numeric(
                        df[col].str.replace(",", "").str.replace(" ", ""), errors="ignore"
                    )

            # build rows
            rows = df.to_dict(orient="records")
            if rows:
                tables.append({"name": f"{sheet}", "rows": rows})

        return tables

    except Exception as e:  # noqa: BLE001
        raise XLSXParseError(str(e))
