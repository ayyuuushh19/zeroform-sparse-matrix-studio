/*
 * memory_analysis.c — Implementation of theoretical matrix memory analysis
 *
 * ZEROFORM Sparse Matrix Studio — Stage 3
 */

#include "memory_analysis.h"
#include "matrix.h"
#include "zeroform.h"

#include <stdint.h>

/* ───────────────────────────────────────────────────────────────────
 * Internal helper: overflow-safe multiplication of two size_t values.
 * ─────────────────────────────────────────────────────────────────── */
static int ma_safe_mul(size_t a, size_t b, size_t *result)
{
    if (a == 0 || b == 0) { *result = 0; return 0; }
    if (a > SIZE_MAX / b) return -1;
    *result = a * b;
    return 0;
}

ZF_Status memory_analysis_compute(size_t rows,
                                  size_t cols,
                                  size_t nnz,
                                  MemoryAnalysis *out)
{
    if (out == NULL)            return ZF_ERR_PARAM;
    if (rows == 0 || cols == 0) return ZF_ERR_PARAM;

    /* Cell ceiling check with overflow protection */
    if (cols > 0 && rows > MAX_CELLS / cols) {
        return ZF_ERR_OVERFLOW;
    }

    /* Check rows * cols overflow */
    size_t total_cells;
    if (ma_safe_mul(rows, cols, &total_cells) != 0) {
        return ZF_ERR_OVERFLOW;
    }

    /* nnz cannot exceed total cell capacity */
    if (nnz > total_cells) {
        return ZF_ERR_PARAM;
    }

    /* Check payload byte overflow: total_cells * 8 and nnz * 16 (fixed theoretical payload) */
    size_t dense_bytes;
    if (ma_safe_mul(total_cells, THEORETICAL_DENSE_BYTES_PER_CELL, &dense_bytes) != 0) {
        return ZF_ERR_OVERFLOW;
    }

    size_t sparse_bytes;
    if (ma_safe_mul(nnz, THEORETICAL_SPARSE_BYTES_PER_ENTRY, &sparse_bytes) != 0) {
        return ZF_ERR_OVERFLOW;
    }

    MemoryAnalysis res;
    res.rows                  = rows;
    res.cols                  = cols;
    res.total_cells           = total_cells;
    res.nnz                   = nnz;
    res.zero_count            = total_cells - nnz;
    res.density_percent       = 100.0 * ((double)nnz / (double)total_cells);
    res.sparsity_percent      = 100.0 * ((double)res.zero_count / (double)total_cells);
    res.dense_payload_bytes   = dense_bytes;
    res.sparse_payload_bytes  = sparse_bytes;

    if (dense_bytes > sparse_bytes) {
        res.sparse_is_smaller = 1;
        res.savings_bytes     = dense_bytes - sparse_bytes;
        res.percent_saved     = 100.0 * ((double)res.savings_bytes / (double)dense_bytes);
    } else if (dense_bytes < sparse_bytes) {
        res.sparse_is_smaller = 0;
        res.savings_bytes     = sparse_bytes - dense_bytes; /* magnitude of extra cost */
        res.percent_saved     = 0.0;
    } else {
        res.sparse_is_smaller = 0;
        res.savings_bytes     = 0;
        res.percent_saved     = 0.0;
    }

    *out = res;
    return ZF_OK;
}

ZF_Status memory_analysis_dense(const DenseMatrix *dm, MemoryAnalysis *out)
{
    if (dm == NULL || !dm->init || out == NULL) return ZF_ERR_PARAM;

    size_t total_cells = dm->rows * dm->cols;
    size_t nnz = 0;
    for (size_t i = 0; i < total_cells; i++) {
        double v = 0.0;
        da_get(&dm->data, i, &v);
        if (v != 0.0) nnz++;
    }

    return memory_analysis_compute(dm->rows, dm->cols, nnz, out);
}

ZF_Status memory_analysis_sparse(const SparseMatrix *sm, MemoryAnalysis *out)
{
    if (sm == NULL || !sm->init || out == NULL) return ZF_ERR_PARAM;
    size_t nnz = sparse_matrix_nonzero_count(sm);
    return memory_analysis_compute(sm->rows, sm->cols, nnz, out);
}
