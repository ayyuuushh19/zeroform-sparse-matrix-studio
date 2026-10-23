"""
validator.py — HTTP request and matrix input validation for ZEROFORM API.

ZEROFORM Sparse Matrix Studio — Stage 4
Senior C11 / Python Systems & API Architecture.

Enforces:
1. Missing request bodies and incorrect JSON types.
2. Ragged matrix rejection.
3. Dimension limits (MAX_DIM: 100,000, MAX_CELLS: 10,000,000, MAX_NNZ: 1,000,000).
4. Non-finite values rejection (NaN, Infinity).
5. Incompatible matrix dimension checks for Add and Multiply.
6. Coordinate bounds checking and duplicate sparse coordinate rejection.
7. Proper HTTP status code assignment (400, 413, 422).
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Tuple, Optional

MAX_DIM = 100_000
MAX_CELLS = 10_000_000
MAX_NNZ = 1_000_000

SUPPORTED_OPERATIONS = {"to_sparse", "to_dense", "add", "multiply", "transpose"}
SUPPORTED_ACTIONS = {"insert", "set", "delete", "search", "traverse", "sort"}


class ValidationError(Exception):
    """Raised when an HTTP request fails input validation."""
    def __init__(self, message: str, status_code: int = 400, code: str = "VALIDATION_ERROR"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message
            }
        }


def validate_number(val: Any, field_name: str) -> float:
    """Validate that val is a real, finite number (not bool, not NaN, not Inf)."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValidationError(f"Field '{field_name}' must be a numeric value, got {type(val).__name__}", 400)
    try:
        num = float(val)
    except (OverflowError, ValueError):
        raise ValidationError(
            f"Field '{field_name}' must be finite and within representable range",
            422,
            "ZF_ERR_VALUE"
        )
    except TypeError:
        raise ValidationError(
            f"Field '{field_name}' must be a numeric value, got {type(val).__name__}",
            400
        )

    if not math.isfinite(num):
        raise ValidationError(f"Field '{field_name}' must be finite (got {val})", 422, "ZF_ERR_VALUE")
    return num


def validate_dimension(dim: Any, field_name: str) -> int:
    """Validate dimension is an integer in range [1, MAX_DIM]."""
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise ValidationError(f"Field '{field_name}' must be an integer, got {type(dim).__name__}", 400)
    if dim <= 0:
        raise ValidationError(f"Field '{field_name}' must be strictly positive (got {dim})", 400, "ZF_ERR_PARAM")
    if dim > MAX_DIM:
        raise ValidationError(
            f"Field '{field_name}' exceeds maximum allowed dimension of {MAX_DIM} (got {dim})",
            413,
            "ZF_ERR_OVERFLOW"
        )
    return dim


def validate_coordinate(coord: Any, limit: int, coord_name: str, context: str = "") -> int:
    """
    Validate that coord is an integer in range [0, limit - 1] and strictly not a boolean.
    Enforces 0-based coordinate indexing across sparse entries and entry actions.
    Raises ValidationError with status 422 and code ZF_ERR_INDEX on failure.
    """
    ctx_prefix = f"{context}: " if context else ""
    if isinstance(coord, bool) or not isinstance(coord, int):
        raise ValidationError(
            f"{ctx_prefix}{coord_name.capitalize()} coordinate must be an integer, got {type(coord).__name__}",
            422,
            "ZF_ERR_INDEX"
        )
    if coord < 0 or coord >= limit:
        try:
            coord_str = str(coord)
        except ValueError:
            coord_str = f"integer (~2^{coord.bit_length()})"
        raise ValidationError(
            f"{ctx_prefix}{coord_name.capitalize()} {coord_str} out of bounds for matrix dimension {limit}",
            422,
            "ZF_ERR_INDEX"
        )
    return coord


def validate_cell_limit(rows: int, cols: int, name: str = "matrix") -> None:
    """
    Validate that total logical cells (rows * cols) do not exceed MAX_CELLS.
    Enforces the universal 10,000,000-cell ceiling on both dense and sparse matrices.
    Uses overflow-safe check: rows > MAX_CELLS // cols.
    """
    if rows <= 0 or cols <= 0:
        return
    if rows > MAX_CELLS // cols:
        cells = rows * cols
        raise ValidationError(
            f"Total cell count {cells} exceeds limit of {MAX_CELLS}",
            413,
            "ZF_ERR_OVERFLOW"
        )


def validate_matrix_dict(raw: Any, name: str = "matrix") -> Dict[str, Any]:
    """
    Validate and normalize a matrix input.
    Accepts:
    1. A raw 2D list: [[1, 2], [3, 4]]
    2. A dictionary with format 'dense' or 'sparse'.
    
    Returns normalized dictionary:
      {'format': 'DENSE', 'rows': R, 'cols': C, 'data': [[...]]}
      or
      {'format': 'SPARSE', 'rows': R, 'cols': C, 'nnz': N, 'entries': [...]}
    """
    if raw is None:
        raise ValidationError(f"Missing required matrix input '{name}'", 400)

    # ── Case 1: 2D list (implicit dense) ───────────────────────────
    if isinstance(raw, list):
        if not raw:
            raise ValidationError(f"Matrix '{name}' cannot be an empty list", 400, "ZF_ERR_PARAM")
        rows = len(raw)
        validate_dimension(rows, f"{name}.rows")

        first_row = raw[0]
        if not isinstance(first_row, list) or not first_row:
            raise ValidationError(f"Matrix '{name}' must contain non-empty lists of rows", 400)
        cols = len(first_row)
        validate_dimension(cols, f"{name}.cols")

        validate_cell_limit(rows, cols, name)

        validated_grid: List[List[float]] = []
        for r, row_list in enumerate(raw):
            if not isinstance(row_list, list):
                raise ValidationError(f"Row {r} in matrix '{name}' must be a list", 400)
            if len(row_list) != cols:
                raise ValidationError(
                    f"Ragged matrix in '{name}': row {r} has {len(row_list)} columns, expected {cols}",
                    400,
                    "ZF_ERR_PARAM"
                )
            row_floats: List[float] = []
            for c, val in enumerate(row_list):
                v = validate_number(val, f"{name}[{r}][{c}]")
                row_floats.append(v)
            validated_grid.append(row_floats)

        return {
            "format": "DENSE",
            "rows": rows,
            "cols": cols,
            "data": validated_grid
        }

    # ── Case 2: Dictionary input ────────────────────────────────────
    if not isinstance(raw, dict):
        raise ValidationError(f"Matrix '{name}' must be a 2D list or dictionary", 400)

    rows = validate_dimension(raw.get("rows"), f"{name}.rows")
    cols = validate_dimension(raw.get("cols"), f"{name}.cols")
    validate_cell_limit(rows, cols, name)
    fmt = str(raw.get("format", "dense")).upper()

    if fmt == "DENSE":

        if "data" in raw:
            grid = raw["data"]
            if not isinstance(grid, list) or len(grid) != rows:
                raise ValidationError(f"Dense matrix '{name}.data' must be a list of {rows} rows", 400)
            validated_grid = []
            for r, row_list in enumerate(grid):
                if not isinstance(row_list, list) or len(row_list) != cols:
                    raise ValidationError(
                        f"Row {r} in '{name}.data' must contain {cols} columns",
                        400,
                        "ZF_ERR_PARAM"
                    )
                row_floats = []
                for c, val in enumerate(row_list):
                    v = validate_number(val, f"{name}.data[{r}][{c}]")
                    row_floats.append(v)
                validated_grid.append(row_floats)
            return {"format": "DENSE", "rows": rows, "cols": cols, "data": validated_grid}

        elif "cells" in raw:
            cells = raw["cells"]
            if not isinstance(cells, list) or len(cells) != rows * cols:
                raise ValidationError(f"Dense matrix '{name}.cells' must contain exactly {rows * cols} cells", 400)
            seen_cells = set()
            validated_cells = []
            for idx, c_item in enumerate(cells):
                if not isinstance(c_item, dict):
                    raise ValidationError(f"Cell at index {idx} in '{name}' must be a dictionary", 400)
                cr = validate_coordinate(c_item.get("row"), rows, "row", f"Cell {idx} in '{name}'")
                cc = validate_coordinate(c_item.get("col"), cols, "col", f"Cell {idx} in '{name}'")
                if (cr, cc) in seen_cells:
                    raise ValidationError(f"Duplicate cell coordinate ({cr}, {cc}) in '{name}'", 422, "ZF_ERR_DUPLICATE")
                seen_cells.add((cr, cc))
                cv = validate_number(c_item.get("value"), f"{name}.cells[{idx}].value")
                validated_cells.append({"row": cr, "col": cc, "value": cv})
            return {"format": "DENSE", "rows": rows, "cols": cols, "cells": validated_cells}
        else:
            raise ValidationError(f"Dense matrix '{name}' requires 'data' or 'cells' field", 400)

    elif fmt == "SPARSE":
        entries = raw.get("entries", raw.get("triplets", []))
        if not isinstance(entries, list):
            raise ValidationError(f"Sparse matrix '{name}' entries must be a list", 400)
        
        if "nnz" in raw:
            declared_nnz = raw["nnz"]
            if isinstance(declared_nnz, bool) or not isinstance(declared_nnz, int) or declared_nnz < 0:
                raise ValidationError(f"Field '{name}.nnz' must be a non-negative integer", 400)
            if declared_nnz != len(entries):
                raise ValidationError(
                    f"Declared nnz {declared_nnz} does not match actual entry count {len(entries)}",
                    400,
                    "ZF_ERR_PARAM"
                )
        nnz = len(entries)
        if nnz > MAX_NNZ:
            raise ValidationError(f"NNZ count {nnz} exceeds limit of {MAX_NNZ}", 413, "ZF_ERR_OVERFLOW")

        seen_coords = set()
        validated_entries: List[Dict[str, Any]] = []
        for idx, item in enumerate(entries):
            if isinstance(item, dict):
                r_raw = item.get("row")
                c_raw = item.get("col")
                v = item.get("value")
            elif isinstance(item, (list, tuple)) and len(item) == 3:
                r_raw, c_raw, v = item[0], item[1], item[2]
            else:
                raise ValidationError(f"Entry {idx} in '{name}' must be dict or [r, c, v] tuple", 400)

            r = validate_coordinate(r_raw, rows, "row", f"Entry {idx} in '{name}'")
            c = validate_coordinate(c_raw, cols, "col", f"Entry {idx} in '{name}'")

            coord = (r, c)
            if coord in seen_coords:
                raise ValidationError(f"Duplicate coordinate ({r}, {c}) in sparse matrix '{name}'", 422, "ZF_ERR_DUPLICATE")
            seen_coords.add(coord)

            val_float = validate_number(v, f"{name}.entries[{idx}].value")
            if val_float == 0:
                raise ValidationError(
                    f"Entry {idx} in sparse matrix '{name}' has explicit zero value; explicit zeros are invalid in sparse format",
                    422,
                    "ZF_ERR_PARAM"
                )
            validated_entries.append({"row": r, "col": c, "value": val_float})

        return {
            "format": "SPARSE",
            "rows": rows,
            "cols": cols,
            "nnz": len(validated_entries),
            "entries": validated_entries
        }

    else:
        raise ValidationError(f"Unsupported matrix format '{fmt}'. Must be DENSE or SPARSE", 400)


def validate_operation_payload(data: Any) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Validate POST /api/operation payload.
    Returns: (operation_name, matrix_a, matrix_b)
    """
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a valid JSON object", 400)

    raw_op = data.get("operation")
    if not raw_op or not isinstance(raw_op, str):
        raise ValidationError("Missing or invalid 'operation' field", 400)

    op = raw_op.strip().lower()
    if op not in SUPPORTED_OPERATIONS:
        raise ValidationError(
            f"Unsupported operation '{raw_op}'. Supported: {sorted(list(SUPPORTED_OPERATIONS))}",
            400
        )

    # Matrix A is required for all operations
    raw_a = data.get("matrix_a", data.get("matrix"))
    if raw_a is None:
        raise ValidationError("Missing required 'matrix_a' field", 400)
    mat_a = validate_matrix_dict(raw_a, "matrix_a")

    mat_b: Optional[Dict[str, Any]] = None

    if op in ("add", "multiply"):
        raw_b = data.get("matrix_b")
        if raw_b is None:
            raise ValidationError(f"Operation '{op}' requires 'matrix_b'", 400)
        mat_b = validate_matrix_dict(raw_b, "matrix_b")

        if op == "add":
            if mat_a["rows"] != mat_b["rows"] or mat_a["cols"] != mat_b["cols"]:
                raise ValidationError(
                    f"Dimension mismatch for addition: Matrix A is {mat_a['rows']}x{mat_a['cols']}, "
                    f"Matrix B is {mat_b['rows']}x{mat_b['cols']}",
                    422,
                    "ZF_ERR_DIMENSION"
                )
        elif op == "multiply":
            if mat_a["cols"] != mat_b["rows"]:
                raise ValidationError(
                    f"Dimension mismatch for multiplication: Matrix A cols ({mat_a['cols']}) "
                    f"must match Matrix B rows ({mat_b['rows']})",
                    422,
                    "ZF_ERR_DIMENSION"
                )
            validate_cell_limit(mat_a["rows"], mat_b["cols"], "multiplication result")

    return op, mat_a, mat_b


def validate_entry_payload(data: Any) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Validate POST /api/entry payload.
    Returns: (action, matrix, params)
    """
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a valid JSON object", 400)

    raw_action = data.get("action")
    if not raw_action or not isinstance(raw_action, str):
        raise ValidationError("Missing or invalid 'action' field", 400)

    action = raw_action.strip().lower()
    if action not in SUPPORTED_ACTIONS:
        raise ValidationError(
            f"Unsupported entry action '{raw_action}'. Supported: {sorted(list(SUPPORTED_ACTIONS))}",
            400
        )

    raw_mat = data.get("matrix")
    if raw_mat is None:
        raise ValidationError("Missing required 'matrix' field", 400)
    mat = validate_matrix_dict(raw_mat, "matrix")

    params: Dict[str, Any] = {"action": action.upper()}

    if action in ("insert", "set"):
        if "row" not in data or "col" not in data:
            raise ValidationError(f"Action '{action}' requires 'row' and 'col' coordinates", 400)
        r = validate_coordinate(data["row"], mat["rows"], "row")
        c = validate_coordinate(data["col"], mat["cols"], "col")
        params["row"] = r
        params["col"] = c

        if "value" not in data:
            raise ValidationError(f"Action '{action}' requires a 'value' field", 400)
        params["value"] = validate_number(data["value"], "value")

        if action == "insert":
            if params["value"] == 0.0:
                raise ValidationError("Cannot insert zero value into sparse matrix", 422, "ZF_ERR_PARAM")

    elif action in ("delete", "search"):
        if "row" not in data or "col" not in data:
            raise ValidationError(f"Action '{action}' requires 'row' and 'col' coordinates", 400)
        r = validate_coordinate(data["row"], mat["rows"], "row")
        c = validate_coordinate(data["col"], mat["cols"], "col")
        params["row"] = r
        params["col"] = c

    # traverse and sort do not require coordinate params
    return action, mat, params
