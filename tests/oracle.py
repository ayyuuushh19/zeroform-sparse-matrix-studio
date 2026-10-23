"""
oracle.py — Independent test oracle for verifying C engine matrix results.

ZEROFORM Sparse Matrix Studio — Stage 4
Senior C11 / Python Systems & API Architecture.

NOTE: This file is STRICTLY for test assertions and verification.
It is NEVER imported by production backend code (app.py, engine.py, protocol.py, validator.py).
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple


def oracle_dense_add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Pure Python reference addition for test validation."""
    rows = len(a)
    cols = len(a[0])
    return [[a[r][c] + b[r][c] for c in range(cols)] for r in range(rows)]


def oracle_dense_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Pure Python reference multiplication for test validation."""
    m = len(a)
    k = len(a[0])
    n = len(b[0])
    res = [[0.0 for _ in range(n)] for _ in range(m)]
    for r in range(m):
        for c in range(n):
            val = 0.0
            for i in range(k):
                val += a[r][i] * b[i][c]
            res[r][c] = val
    return res


def oracle_dense_transpose(a: List[List[float]]) -> List[List[float]]:
    """Pure Python reference transpose for test validation."""
    rows = len(a)
    cols = len(a[0])
    return [[a[r][c] for r in range(rows)] for c in range(cols)]


def oracle_dense_to_triplets(a: List[List[float]]) -> List[Dict[str, Any]]:
    """Convert dense grid to sorted non-zero triplets."""
    triplets = []
    for r in range(len(a)):
        for c in range(len(a[0])):
            if a[r][c] != 0.0:
                triplets.append({"row": r, "col": c, "value": a[r][c]})
    return triplets


def oracle_memory_analysis(rows: int, cols: int, nnz: int) -> Dict[str, Any]:
    """Theoretical memory calculation matching Stage 3 specification."""
    total_cells = rows * cols
    zero_count = total_cells - nnz
    density = (nnz / total_cells) * 100.0 if total_cells > 0 else 0.0
    sparsity = (zero_count / total_cells) * 100.0 if total_cells > 0 else 0.0
    dense_bytes = total_cells * 8
    sparse_bytes = nnz * 16  # Stage 3 theoretical payload: 16 bytes per entry
    sparse_is_smaller = sparse_bytes < dense_bytes
    savings_bytes = (dense_bytes - sparse_bytes) if sparse_is_smaller else (sparse_bytes - dense_bytes)
    percent_saved = (savings_bytes / dense_bytes) * 100.0 if (sparse_is_smaller and dense_bytes > 0) else 0.0

    return {
        "rows": rows,
        "cols": cols,
        "total_cells": total_cells,
        "nnz": nnz,
        "zero_count": zero_count,
        "density_percent": density,
        "sparsity_percent": sparsity,
        "dense_payload_bytes": dense_bytes,
        "sparse_payload_bytes": sparse_bytes,
        "sparse_is_smaller": sparse_is_smaller,
        "savings_bytes": savings_bytes,
        "percent_saved": percent_saved
    }
