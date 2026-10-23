/*
 * dense_matrix.c — DenseMatrix implementation for ZEROFORM Stage 2
 *
 * All storage uses the DynamicArray from Stage 1 (zeroform.h / zeroform.c).
 * No third-party matrix libraries; no VLAs; no C++.
 *
 * Layout: element (r, c) lives at flat index  r * cols + c  inside da.
 *
 * Numeric policy
 * ──────────────
 * • Non-finite values (NaN, ±inf) are rejected with ZF_ERR_VALUE.
 * • Zero dimensions (rows == 0 or cols == 0) are rejected with ZF_ERR_PARAM.
 * • Positive-dimension all-zero matrices are fully legal.
 *
 * Lifecycle rule
 * ──────────────
 * Call dense_matrix_init() or dense_matrix_from_flat() exactly once per
 * struct.  After dense_matrix_destroy() the struct is zeroed and safe to
 * re-initialize.  Double destruction is safe.
 */

#include "matrix.h"
#include "zeroform.h"

#include <stdlib.h>    /* malloc, free      */
#include <string.h>    /* memcpy            */
#include <math.h>      /* isfinite          */
#include <stdint.h>    /* SIZE_MAX          */

/* ───────────────────────────────────────────────────────────────────
 * Internal helper: overflow-safe multiplication of two size_t values.
 * Returns 0 on success, -1 on overflow.
 * (Cannot reuse checked_mul from zeroform.c — it is static there.)
 * ─────────────────────────────────────────────────────────────────── */
static int dm_safe_mul(size_t a, size_t b, size_t *result)
{
    if (a == 0 || b == 0) { *result = 0; return 0; }
    if (a > SIZE_MAX / b) return -1;
    *result = a * b;
    return 0;
}

/* ───────────────────────────────────────────────────────────────────
 * Internal helper: compute flat index for (row, col) after validation.
 * Returns ZF_OK and writes *idx, or returns ZF_ERR_INDEX / ZF_ERR_OVERFLOW.
 * ─────────────────────────────────────────────────────────────────── */
static ZF_Status dm_flat_index(const DenseMatrix *dm,
                                size_t row, size_t col,
                                size_t *idx)
{
    if (row >= dm->rows || col >= dm->cols) return ZF_ERR_INDEX;
    /* row * cols + col: first check row*cols won't overflow, then add col. */
    size_t prod;
    if (dm_safe_mul(row, dm->cols, &prod) != 0) return ZF_ERR_OVERFLOW;
    if (prod > SIZE_MAX - col)                   return ZF_ERR_OVERFLOW;
    *idx = prod + col;
    return ZF_OK;
}

/* ═══════════════════════════════════════════════════════════════════
 * Lifecycle
 * ═══════════════════════════════════════════════════════════════════ */

ZF_Status dense_matrix_init(DenseMatrix *dm, size_t rows, size_t cols)
{
    if (dm == NULL) return ZF_ERR_PARAM;
    memset(dm, 0, sizeof(*dm));
    if (rows == 0 || cols == 0) {
        return ZF_ERR_PARAM;
    }

    /* Universal cell limit check with overflow protection */
    if (cols > 0 && rows > MAX_CELLS / cols) {
        return ZF_ERR_OVERFLOW;
    }

    /* Check rows * cols won't overflow before we allocate. */
    size_t ncells;
    if (dm_safe_mul(rows, cols, &ncells) != 0) {
        return ZF_ERR_OVERFLOW;
    }

    /* Initialize the backing DynamicArray with ncells double slots. */
    ZF_Status s = da_init(&dm->data, sizeof(double), ncells);
    if (s != ZF_OK) {
        return s;
    }

    /*
     * da_init pre-allocates capacity but leaves length == 0.
     * We need length == ncells so that da_get/da_set work by index.
     * We zero-fill the raw bytes directly: since the allocation is already
     * large enough (capacity == ncells), we can memset the whole block
     * and then manually advance the length field.
     */
    size_t nbytes;
    if (dm_safe_mul(ncells, sizeof(double), &nbytes) != 0) {
        da_destroy(&dm->data);
        return ZF_ERR_OVERFLOW;
    }
    memset(dm->data.data, 0, nbytes);
    dm->data.length = ncells;

    dm->rows = rows;
    dm->cols = cols;
    dm->init = 1;
    return ZF_OK;
}

void dense_matrix_destroy(DenseMatrix *dm)
{
    if (dm == NULL) return;
    da_destroy(&dm->data);   /* safe even if never initialized (data==NULL) */
    dm->rows = 0;
    dm->cols = 0;
    dm->init = 0;
}

ZF_Status dense_matrix_copy(DenseMatrix *dst, const DenseMatrix *src)
{
    if (dst == NULL || src == NULL || dst == src) return ZF_ERR_PARAM;
    if (!src->init)                               return ZF_ERR_PARAM;
    memset(dst, 0, sizeof(*dst));

    /* Initialize dst with the same dimensions (zeros it). */
    ZF_Status s = dense_matrix_init(dst, src->rows, src->cols);
    if (s != ZF_OK) return s;

    /* Overwrite with the source bytes — length is already == ncells. */
    size_t ncells = src->rows * src->cols;          /* no overflow: src is valid */
    memcpy(dst->data.data, src->data.data, ncells * sizeof(double));
    return ZF_OK;
}

/* ═══════════════════════════════════════════════════════════════════
 * Element access
 * ═══════════════════════════════════════════════════════════════════ */

ZF_Status dense_matrix_get(const DenseMatrix *dm,
                            size_t row, size_t col, double *out)
{
    if (dm == NULL || out == NULL) return ZF_ERR_PARAM;
    if (!dm->init)                 return ZF_ERR_PARAM;

    size_t idx;
    ZF_Status s = dm_flat_index(dm, row, col, &idx);
    if (s != ZF_OK) return s;

    /* da_get copies sizeof(double) bytes into *out. */
    return da_get(&dm->data, idx, out);
}

ZF_Status dense_matrix_set(DenseMatrix *dm,
                            size_t row, size_t col, double value)
{
    if (dm == NULL)  return ZF_ERR_PARAM;
    if (!dm->init)   return ZF_ERR_PARAM;
    /* Reject NaN and ±infinity. */
    if (!isfinite(value)) return ZF_ERR_VALUE;

    size_t idx;
    ZF_Status s = dm_flat_index(dm, row, col, &idx);
    if (s != ZF_OK) return s;

    return da_set(&dm->data, idx, &value);
}

/* ═══════════════════════════════════════════════════════════════════
 * Construction from flat array
 * ═══════════════════════════════════════════════════════════════════ */

ZF_Status dense_matrix_from_flat(DenseMatrix *dm,
                                  size_t rows, size_t cols,
                                  const double *flat)
{
    if (dm == NULL) return ZF_ERR_PARAM;
    memset(dm, 0, sizeof(*dm));
    if (flat == NULL || rows == 0 || cols == 0) return ZF_ERR_PARAM;

    /* Universal cell limit check with overflow protection */
    if (cols > 0 && rows > MAX_CELLS / cols) return ZF_ERR_OVERFLOW;

    /* Validate all values first — avoids partial construction on bad input. */
    size_t ncells;
    if (dm_safe_mul(rows, cols, &ncells) != 0) return ZF_ERR_OVERFLOW;

    for (size_t i = 0; i < ncells; i++) {
        if (!isfinite(flat[i])) return ZF_ERR_VALUE;
    }

    /* Build the zero-filled matrix, then bulk-copy the validated data. */
    ZF_Status s = dense_matrix_init(dm, rows, cols);
    if (s != ZF_OK) return s;

    /* safe: ncells * sizeof(double) was already checked in dm_init. */
    memcpy(dm->data.data, flat, ncells * sizeof(double));
    return ZF_OK;
}

/* ═══════════════════════════════════════════════════════════════════
 * Accessors
 * ═══════════════════════════════════════════════════════════════════ */

size_t dense_matrix_rows(const DenseMatrix *dm)
{
    return (dm != NULL && dm->init) ? dm->rows : 0;
}

size_t dense_matrix_cols(const DenseMatrix *dm)
{
    return (dm != NULL && dm->init) ? dm->cols : 0;
}
