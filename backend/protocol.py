"""
protocol.py — Strict version 1 serialization & parsing for ZEROFORM engine.

ZEROFORM Sparse Matrix Studio — Stage 4
Senior C11 / Python Systems & API Architecture.

Enforces:
1. Exact v1 line-oriented protocol format (protocol/PROTOCOL.md).
2. Strict decimal formatting and numeric validation (rejecting NaN, Inf).
3. Full verification of response framing (OK/ERR ... END).
4. Full validation of returned entry counts vs NNZ.
5. Distinction between C protocol error (OK vs ERR) and framing/process errors.
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Optional


class ProtocolError(Exception):
    """Base exception for protocol framing or parsing issues."""
    pass


class ProtocolFramingError(ProtocolError):
    """Raised when response missing OK/ERR header or END terminator."""
    pass


class ProtocolParseError(ProtocolError):
    """Raised when response fields are malformed or fail count checks."""
    pass


def format_float(val: float | int) -> str:
    """Format floating point values deterministically without trailing decimal clutter."""
    if not isinstance(val, (int, float)):
        raise ProtocolParseError(f"Expected numeric value, got {type(val).__name__}")
    if not math.isfinite(val):
        raise ProtocolParseError(f"Non-finite numeric value cannot be serialized: {val}")
    if isinstance(val, int):
        return str(val)
    if abs(val) < 1e15 and val.is_integer():
        return str(int(val))
    return str(val)


def serialize_matrix(prefix: str, matrix: Dict[str, Any]) -> List[str]:
    """
    Serialize a matrix dictionary into protocol D lines.
    Supports both DENSE and SPARSE representations.
    """
    lines: List[str] = []
    fmt = matrix.get("format", "dense").upper()
    rows = int(matrix["rows"])
    cols = int(matrix["cols"])

    if rows <= 0 or cols <= 0:
        raise ProtocolParseError(f"Matrix dimensions must be positive: {rows}x{cols}")

    lines.append(f"D {prefix}_ROWS {rows}")
    lines.append(f"D {prefix}_COLS {cols}")

    if fmt == "DENSE":
        lines.append(f"D {prefix}_FORMAT DENSE")
        # May be provided as 2D list `data` or list of cells `cells`
        if "data" in matrix:
            data = matrix["data"]
            if len(data) != rows:
                raise ProtocolParseError(f"Matrix row count {len(data)} does not match {rows}")
            for r in range(rows):
                row_vals = data[r]
                if len(row_vals) != cols:
                    raise ProtocolParseError(f"Row {r} column count {len(row_vals)} does not match {cols}")
                for c in range(cols):
                    v = row_vals[c]
                    lines.append(f"D {prefix}_CELL {r} {c} {format_float(v)}")
        elif "cells" in matrix:
            cells = matrix["cells"]
            if len(cells) != rows * cols:
                raise ProtocolParseError(f"Dense cell count {len(cells)} does not match {rows * cols}")
            for cell in cells:
                r, c, v = cell["row"], cell["col"], cell["value"]
                lines.append(f"D {prefix}_CELL {r} {c} {format_float(v)}")
        else:
            raise ProtocolParseError(f"Dense matrix missing 'data' or 'cells' field")

    elif fmt == "SPARSE":
        lines.append(f"D {prefix}_FORMAT SPARSE")
        entries = matrix.get("entries", matrix.get("triplets", []))
        nnz = int(matrix.get("nnz", len(entries)))
        if len(entries) != nnz:
            raise ProtocolParseError(f"Sparse entry count {len(entries)} does not match NNZ {nnz}")
        lines.append(f"D {prefix}_NNZ {nnz}")
        for entry in entries:
            if isinstance(entry, dict):
                r = int(entry["row"])
                c = int(entry["col"])
                v = entry["value"]
            elif isinstance(entry, (list, tuple)) and len(entry) == 3:
                r, c, v = int(entry[0]), int(entry[1]), entry[2]
            else:
                raise ProtocolParseError(f"Invalid sparse entry element: {entry}")
            if v == 0:
                raise ProtocolParseError(f"Explicit zero entry at ({r}, {c}) is invalid in sparse protocol serialization")
            lines.append(f"D {prefix}_ENTRY {r} {c} {format_float(v)}")
    else:
        raise ProtocolParseError(f"Unsupported matrix format: {fmt}")

    return lines


def serialize_request(
    command: str,
    matrix_a: Optional[Dict[str, Any]] = None,
    matrix_b: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Construct complete line-oriented request string ending with END\n.
    """
    cmd = command.strip().upper()
    lines: List[str] = [f"CMD {cmd} v1"]

    if matrix_a is not None:
        lines.extend(serialize_matrix("A", matrix_a))

    if matrix_b is not None:
        lines.extend(serialize_matrix("B", matrix_b))

    if params is not None:
        if "row" in params and params["row"] is not None:
            lines.append(f"D ROW {int(params['row'])}")
        if "col" in params and params["col"] is not None:
            lines.append(f"D COL {int(params['col'])}")
        if "value" in params and params["value"] is not None:
            lines.append(f"D VALUE {format_float(params['value'])}")
        if "action" in params and params["action"] is not None:
            lines.append(f"D ACTION {str(params['action']).upper()}")

    lines.append("END\n")
    return "\n".join(lines)


def parse_response(stdout_text: str) -> Dict[str, Any]:
    """
    Parse a deterministic C engine response into structured Python dictionary.
    
    Returns:
        On success: {'ok': True, 'rows': ..., 'cols': ..., 'entries': [...], ...}
        On C error:  {'ok': False, 'code': 'ZF_ERR_...', 'message': '...'}
        
    Raises:
        ProtocolFramingError: If header (OK/ERR) or terminator (END) is invalid.
        ProtocolParseError: If fields are corrupt, counts mismatch, or numbers invalid.
    """
    raw_lines = stdout_text.strip().splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]

    if not lines:
        raise ProtocolFramingError("Empty response received from engine")

    status_header = lines[0]
    if status_header not in ("OK", "ERR"):
        raise ProtocolFramingError(f"Invalid status header line: '{status_header}'")

    if lines[-1] != "END":
        raise ProtocolFramingError(f"Missing terminating END marker in response (got: '{lines[-1]}')")

    data_lines = lines[1:-1]

    # Handle C error response
    if status_header == "ERR":
        err_res: Dict[str, Any] = {"ok": False, "code": "ZF_ERR_INTERNAL", "message": "unknown_error"}
        for line in data_lines:
            if not line.startswith("D "):
                raise ProtocolFramingError(f"Malformed error data line: '{line}'")
            parts = line[2:].split(" ", 1)
            key = parts[0]
            val = parts[1] if len(parts) > 1 else ""
            if key == "CODE":
                err_res["code"] = val
            elif key == "MESSAGE":
                err_res["message"] = val
        return err_res

    # Handle OK response
    res: Dict[str, Any] = {
        "ok": True,
        "entries": [],
        "cells": []
    }

    for line in data_lines:
        if not line.startswith("D "):
            raise ProtocolFramingError(f"Malformed response data line: '{line}'")
        
        parts = line[2:].split()
        if not parts:
            continue
        key = parts[0]

        if key in ("STATUS", "VERSION"):
            res[key.lower()] = parts[1] if len(parts) > 1 else ""

        elif key in ("ROWS", "COLS", "TOTAL_CELLS", "NNZ", "ZERO_COUNT",
                     "DENSE_BYTES", "SPARSE_BYTES", "SAVINGS_BYTES", "ROW", "COL", "POS"):
            try:
                res[key.lower()] = int(parts[1])
            except (IndexError, ValueError) as exc:
                raise ProtocolParseError(f"Invalid integer value for {key}: {line}") from exc

        elif key in ("DENSITY", "SPARSITY", "PERCENT_SAVED"):
            try:
                res[key.lower()] = float(parts[1])
            except (IndexError, ValueError) as exc:
                raise ProtocolParseError(f"Invalid float value for {key}: {line}") from exc

        elif key in ("SPARSE_IS_SMALLER", "FOUND"):
            try:
                res[key.lower()] = bool(int(parts[1]))
            except (IndexError, ValueError) as exc:
                raise ProtocolParseError(f"Invalid boolean flag for {key}: {line}") from exc

        elif key == "VALUE":
            try:
                res["value"] = float(parts[1])
            except (IndexError, ValueError) as exc:
                raise ProtocolParseError(f"Invalid float for VALUE: {line}") from exc

        elif key == "ENTRY":
            if len(parts) != 4:
                raise ProtocolParseError(f"Malformed ENTRY line: '{line}'")
            try:
                r = int(parts[1])
                c = int(parts[2])
                v = float(parts[3])
                if not math.isfinite(v):
                    raise ProtocolParseError(f"Non-finite float in ENTRY: {parts[3]}")
                res["entries"].append({"row": r, "col": c, "value": v})
            except ValueError as exc:
                raise ProtocolParseError(f"Invalid numbers in ENTRY line: '{line}'") from exc

        elif key == "CELL":
            if len(parts) != 4:
                raise ProtocolParseError(f"Malformed CELL line: '{line}'")
            try:
                r = int(parts[1])
                c = int(parts[2])
                v = float(parts[3])
                if not math.isfinite(v):
                    raise ProtocolParseError(f"Non-finite float in CELL: {parts[3]}")
                res["cells"].append({"row": r, "col": c, "value": v})
            except ValueError as exc:
                raise ProtocolParseError(f"Invalid numbers in CELL line: '{line}'") from exc

    # Validate NNZ count if present
    if "nnz" in res:
        if len(res["entries"]) != res["nnz"]:
            raise ProtocolParseError(
                f"Response NNZ mismatch: expected {res['nnz']}, received {len(res['entries'])} entries"
            )

    # Validate dense cell count if dense cells returned
    if "rows" in res and "cols" in res and res["cells"]:
        expected_cells = res["rows"] * res["cols"]
        if len(res["cells"]) != expected_cells:
            raise ProtocolParseError(
                f"Response cell count mismatch: expected {expected_cells}, received {len(res['cells'])} cells"
            )

    return res
