/*
 * sparse_matrix.c — SparseMatrix implementation for ZEROFORM Stage 2
 *
 * Storage: a DynamicArray of Triplet structures (row, col, value).
 * Ordering invariant: triplets are always sorted in ascending row-major
 * (lexicographic) order — primary key = row, secondary key = col.
 *
 * This means binary search is always valid, and every mutation that adds
 * a new entry inserts it at the correct sorted position.
 *
 * Algorithms implemented from scratch (no qsort, no C++ containers):
 *   - Binary search     O(log nnz)
 *   - Insert at position via da_insert  O(nnz) shift
 *   - Delete at position via da_remove  O(nnz) shift
 *   - Merge sort        O(nnz log nnz)
 *
 * Numeric policy (matches matrix.h docs):
 *   - Implicit zero: reading a missing coord returns 0.0 (no error).
 *   - Zero input: set(0.0) deletes; insert(0.0) returns ZF_ERR_VALUE.
 *   - NaN / Inf: always rejected with ZF_ERR_VALUE.
 *   - Duplicate: insert on existing coord returns ZF_ERR_DUPLICATE.
 *   - Not found: delete on missing coord returns ZF_ERR_NOT_FOUND.
 */

#include "matrix.h"
#include "zeroform.h"

#include <stdlib.h>   /* malloc, free           */
#include <string.h>   /* memcpy                 */
#include <math.h>     /* isfinite               */
#include <stdint.h>   /* SIZE_MAX               */

/* ───────────────────────────────────────────────────────────────────
 * Overflow-safe multiply (same pattern as dense_matrix.c — static)
 * ─────────────────────────────────────────────────────────────────── */
static int sm_safe_mul(size_t a, size_t b, size_t *result)
{
    if (a == 0 || b == 0) { *result = 0; return 0; }
    if (a > SIZE_MAX / b) return -1;
    *result = a * b;
    return 0;
}

/* ───────────────────────────────────────────────────────────────────
 * Triplet comparison for row-major ordering.
 * Returns <0, 0, or >0.
 * ─────────────────────────────────────────────────────────────────── */
static int triplet_cmp(const Triplet *a, const Triplet *b)
{
    if (a->row != b->row) {
        return (a->row < b->row) ? -1 : 1;
    }
    if (a->col != b->col) {
        return (a->col < b->col) ? -1 : 1;
    }
    return 0;
}

/* ───────────────────────────────────────────────────────────────────
 * Binary search: find the position of (row, col) in the sorted entries.
 *
 * On success: returns 1 and writes the index to *out_pos.
 * On failure (not found): returns 0 and writes the insertion point to
 *   *out_pos (i.e. the index where (row,col) should be inserted to keep
 *   the array sorted).
 *
 * This dual-mode binary search avoids a second scan in insert/delete.
 * ─────────────────────────────────────────────────────────────────── */
static int sm_bsearch(const SparseMatrix *sm,
                       size_t row, size_t col,
                       size_t *out_pos)
{
    size_t lo = 0;
    size_t hi = da_length(&sm->entries); /* one-past-end */
    Triplet key; key.row = row; key.col = col; key.value = 0.0;

    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;   /* avoid (lo+hi)/2 overflow */
        const Triplet *t = (const Triplet *)da_ptr(&sm->entries, mid);
        int cmp = triplet_cmp(&key, t);
        if (cmp == 0) {
            *out_pos = mid;
            return 1;   /* found */
        } else if (cmp < 0) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    *out_pos = lo;  /* insertion point */
    return 0;       /* not found */
}

/* ═══════════════════════════════════════════════════════════════════
 * Lifecycle
 * ═══════════════════════════════════════════════════════════════════ */

ZF_Status sparse_matrix_init(SparseMatrix *sm, size_t rows, size_t cols)
{
    if (sm == NULL) return ZF_ERR_PARAM;
    memset(sm, 0, sizeof(*sm));
    if (rows == 0 || cols == 0) {
        return ZF_ERR_PARAM;
    }

    /* Universal cell limit check with overflow protection */
    if (cols > 0 && rows > MAX_CELLS / cols) {
        return ZF_ERR_OVERFLOW;
    }

    /* Initialize the DynamicArray to hold Triplet objects.
     * Pass 0 as initial capacity so no allocation happens until
     * the first nonzero is inserted. */
    ZF_Status s = da_init(&sm->entries, sizeof(Triplet), 0);
    if (s != ZF_OK) {
        sm->rows = 0; sm->cols = 0; sm->init = 0;
        return s;
    }

    sm->rows = rows;
    sm->cols = cols;
    sm->init = 1;
    return ZF_OK;
}

void sparse_matrix_destroy(SparseMatrix *sm)
{
    if (sm == NULL) return;
    da_destroy(&sm->entries);
    sm->rows = 0;
    sm->cols = 0;
    sm->init = 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Merge sort — O(nnz log nnz), no qsort()
 * ═══════════════════════════════════════════════════════════════════
 *
 * Bottom-up iterative merge sort operating on the DynamicArray's raw
 * backing storage.  A temporary buffer of nnz Triplets is heap-allocated
 * once, used for merging, then freed.
 *
 * Time:  O(nnz log nnz)
 * Space: O(nnz) temporary heap allocation.
 */

/*
 * merge_runs — merge two sorted runs within buf[].
 * Left run:  buf[l .. m-1]
 * Right run: buf[m .. r-1]
 * Result written into tmp[l .. r-1], then copied back to buf[l .. r-1].
 */
static void merge_runs(Triplet *buf, Triplet *tmp,
                        size_t l, size_t m, size_t r)
{
    size_t i = l, j = m, k = l;
    while (i < m && j < r) {
        if (triplet_cmp(&buf[i], &buf[j]) <= 0) {
            tmp[k++] = buf[i++];
        } else {
            tmp[k++] = buf[j++];
        }
    }
    while (i < m) tmp[k++] = buf[i++];
    while (j < r) tmp[k++] = buf[j++];
    /* Copy merged result back. */
    memcpy(&buf[l], &tmp[l], (r - l) * sizeof(Triplet));
}

ZF_Status sparse_matrix_sort(SparseMatrix *sm)
{
    if (sm == NULL || !sm->init) return ZF_ERR_PARAM;

    size_t n = da_length(&sm->entries);
    if (n <= 1) return ZF_OK;   /* 0 or 1 entries: already sorted */

    /*
     * Get a direct pointer to the backing array.
     * We must NOT call da_append / da_insert / da_remove during this
     * function, so the pointer remains valid throughout.
     */
    Triplet *buf = (Triplet *)da_ptr(&sm->entries, 0);

    /* Allocate the temporary merge buffer. */
    size_t buf_bytes;
    if (sm_safe_mul(n, sizeof(Triplet), &buf_bytes) != 0) return ZF_ERR_OVERFLOW;
    Triplet *tmp = (Triplet *)malloc(buf_bytes);
    if (tmp == NULL) return ZF_ERR_NOMEM;

    /*
     * Bottom-up merge sort: start with run size 1, double each pass.
     * For run size `width`, merge pairs of adjacent runs:
     *   [0 .. width-1] with [width .. 2*width-1],
     *   [2*width .. 3*width-1] with [3*width .. 4*width-1], ...
     */
    for (size_t width = 1; width < n; width *= 2) {
        for (size_t l = 0; l < n; l += 2 * width) {
            size_t m = l + width;
            if (m >= n) break;   /* no right run exists */
            size_t r = m + width;
            if (r > n) r = n;    /* clamp to array end */
            merge_runs(buf, tmp, l, m, r);
        }
        if (width > SIZE_MAX / 2) break;
    }

    free(tmp);
    return ZF_OK;
}

/* ═══════════════════════════════════════════════════════════════════
 * Construction
 * ═══════════════════════════════════════════════════════════════════ */

ZF_Status sparse_matrix_from_dense(SparseMatrix *sm, const DenseMatrix *dm)
{
    if (sm == NULL) return ZF_ERR_PARAM;
    memset(sm, 0, sizeof(*sm));
    if (dm == NULL || !dm->init) return ZF_ERR_PARAM;

    ZF_Status s = sparse_matrix_init(sm, dm->rows, dm->cols);
    if (s != ZF_OK) return s;

    /* Scan every cell; store only nonzero finite values.
     * Because we scan in row-major order, the entries are inserted
     * in sorted order — no explicit sort needed. */
    for (size_t r = 0; r < dm->rows; r++) {
        for (size_t c = 0; c < dm->cols; c++) {
            double v = 0.0;
            /* da_get cannot fail here: r*cols+c is always valid. */
            da_get(&dm->data, r * dm->cols + c, &v);

            if (v == 0.0) continue;   /* skip structural zeros */
            /* isfinite check: dense_matrix_set already enforced this,
             * but we guard defensively. */
            if (!isfinite(v)) {
                sparse_matrix_destroy(sm);
                return ZF_ERR_VALUE;
            }

            Triplet t; t.row = r; t.col = c; t.value = v;
            s = da_append(&sm->entries, &t);
            if (s != ZF_OK) {
                sparse_matrix_destroy(sm);
                return s;
            }
        }
    }
    return ZF_OK;
}

ZF_Status sparse_matrix_from_triplets(SparseMatrix *sm,
                                       size_t rows, size_t cols,
                                       const Triplet *triplets,
                                       size_t count)
{
    if (sm == NULL) return ZF_ERR_PARAM;
    memset(sm, 0, sizeof(*sm));
    if (rows == 0 || cols == 0) return ZF_ERR_PARAM;
    if (count > 0 && triplets == NULL) return ZF_ERR_PARAM;

    ZF_Status s = sparse_matrix_init(sm, rows, cols);
    if (s != ZF_OK) return s;

    /* First pass: validate all input triplets. */
    for (size_t i = 0; i < count; i++) {
        const Triplet *t = &triplets[i];
        if (t->row >= rows || t->col >= cols) {
            sparse_matrix_destroy(sm);
            return ZF_ERR_INDEX;
        }
        if (!isfinite(t->value)) {
            sparse_matrix_destroy(sm);
            return ZF_ERR_VALUE;
        }
        /* Explicit zero triplets are silently skipped (checked below). */
    }

    /* Second pass: append non-zero validated triplets. */
    for (size_t i = 0; i < count; i++) {
        if (triplets[i].value == 0.0) continue;  /* skip explicit zeros */
        s = da_append(&sm->entries, &triplets[i]);
        if (s != ZF_OK) {
            sparse_matrix_destroy(sm);
            return s;
        }
    }

    /* Sort into row-major order. */
    s = sparse_matrix_sort(sm);
    if (s != ZF_OK) {
        sparse_matrix_destroy(sm);
        return s;
    }

    /* Check for duplicates after sorting (adjacent duplicates only). */
    size_t n = da_length(&sm->entries);
    for (size_t i = 1; i < n; i++) {
        const Triplet *prev = (const Triplet *)da_ptr(&sm->entries, i - 1);
        const Triplet *curr = (const Triplet *)da_ptr(&sm->entries, i);
        if (triplet_cmp(prev, curr) == 0) {
            sparse_matrix_destroy(sm);
            return ZF_ERR_DUPLICATE;
        }
    }

    return ZF_OK;
}

/* ═══════════════════════════════════════════════════════════════════
 * Read access
 * ═══════════════════════════════════════════════════════════════════ */

ZF_Status sparse_matrix_search(const SparseMatrix *sm,
                                size_t row, size_t col,
                                double *out_val, size_t *out_pos)
{
    if (sm == NULL || !sm->init) return ZF_ERR_PARAM;
    if (row >= sm->rows || col >= sm->cols) return ZF_ERR_INDEX;

    size_t pos;
    int found = sm_bsearch(sm, row, col, &pos);

    if (!found) {
        if (out_pos) *out_pos = pos;
        return ZF_ERR_NOT_FOUND;
    }

    if (out_val) {
        const Triplet *t = (const Triplet *)da_ptr(&sm->entries, pos);
        *out_val = t->value;
    }
    if (out_pos) *out_pos = pos;
    return ZF_OK;
}

ZF_Status sparse_matrix_get(const SparseMatrix *sm,
                              size_t row, size_t col, double *out)
{
    if (sm == NULL || out == NULL) return ZF_ERR_PARAM;
    if (!sm->init)                 return ZF_ERR_PARAM;
    if (row >= sm->rows || col >= sm->cols) return ZF_ERR_INDEX;

    double val = 0.0;
    ZF_Status s = sparse_matrix_search(sm, row, col, &val, NULL);
    if (s == ZF_ERR_NOT_FOUND) {
        /* Implicit zero — not an error for get(). */
        *out = 0.0;
        return ZF_OK;
    }
    if (s == ZF_OK) {
        *out = val;
    }
    return s;
}

/* ═══════════════════════════════════════════════════════════════════
 * Mutation
 * ═══════════════════════════════════════════════════════════════════ */

ZF_Status sparse_matrix_insert(SparseMatrix *sm,
                                size_t row, size_t col, double value)
{
    if (sm == NULL || !sm->init)    return ZF_ERR_PARAM;
    if (row >= sm->rows || col >= sm->cols) return ZF_ERR_INDEX;
    /* Insert requires a nonzero finite value. */
    if (value == 0.0 || !isfinite(value)) return ZF_ERR_VALUE;

    size_t pos;
    int found = sm_bsearch(sm, row, col, &pos);
    if (found) return ZF_ERR_DUPLICATE;

    /* pos is the correct insertion index to keep sorted order. */
    Triplet t; t.row = row; t.col = col; t.value = value;
    return da_insert(&sm->entries, pos, &t);
}

ZF_Status sparse_matrix_delete(SparseMatrix *sm, size_t row, size_t col)
{
    if (sm == NULL || !sm->init)    return ZF_ERR_PARAM;
    if (row >= sm->rows || col >= sm->cols) return ZF_ERR_INDEX;

    size_t pos;
    int found = sm_bsearch(sm, row, col, &pos);
    if (!found) return ZF_ERR_NOT_FOUND;

    return da_remove(&sm->entries, pos);
}

ZF_Status sparse_matrix_set(SparseMatrix *sm,
                              size_t row, size_t col, double value)
{
    if (sm == NULL || !sm->init)    return ZF_ERR_PARAM;
    if (row >= sm->rows || col >= sm->cols) return ZF_ERR_INDEX;
    if (!isfinite(value))           return ZF_ERR_VALUE;

    size_t pos;
    int found = sm_bsearch(sm, row, col, &pos);

    if (value == 0.0) {
        /* Setting to zero: delete the entry if it exists, no-op otherwise. */
        if (found) return da_remove(&sm->entries, pos);
        return ZF_OK;
    }

    if (found) {
        /*
         * Update the existing triplet in place.
         * We MUST NOT call da_ptr here and then use that pointer across
         * a mutating call.  Instead we use da_set which copies by index.
         */
        Triplet t; t.row = row; t.col = col; t.value = value;
        return da_set(&sm->entries, pos, &t);
    } else {
        /* Insert at the sorted position. */
        Triplet t; t.row = row; t.col = col; t.value = value;
        return da_insert(&sm->entries, pos, &t);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Traversal
 * ═══════════════════════════════════════════════════════════════════ */

ZF_Status sparse_matrix_traverse(const SparseMatrix *sm,
                                  SparseTraverseFn fn, void *ctx)
{
    if (sm == NULL || !sm->init || fn == NULL) return ZF_ERR_PARAM;

    size_t n = da_length(&sm->entries);
    for (size_t i = 0; i < n; i++) {
        /*
         * We pass a const Triplet pointer.  The callback signature
         * declares it const, so the compiler enforces read-only access.
         *
         * We use da_ptr() here for efficiency (no copy per element).
         * Since fn must not mutate the SparseMatrix, da_ptr() remains
         * valid for the lifetime of the callback call.
         */
        const Triplet *t = (const Triplet *)da_ptr(&sm->entries, i);
        int rv = fn(t, ctx);
        if (rv != 0) return (ZF_Status)rv;
    }
    return ZF_OK;
}

/* ═══════════════════════════════════════════════════════════════════
 * Conversion
 * ═══════════════════════════════════════════════════════════════════ */

ZF_Status sparse_matrix_to_dense(const SparseMatrix *sm, DenseMatrix *dm)
{
    if (dm == NULL) return ZF_ERR_PARAM;
    memset(dm, 0, sizeof(*dm));
    if (sm == NULL || !sm->init) return ZF_ERR_PARAM;

    /* Guard against logical dimensions exceeding universal cell ceiling */
    if (sm->cols > 0 && sm->rows > MAX_CELLS / sm->cols) {
        return ZF_ERR_OVERFLOW;
    }

    /* Create a zero-filled dense matrix of the same dimensions. */
    ZF_Status s = dense_matrix_init(dm, sm->rows, sm->cols);
    if (s != ZF_OK) return s;

    /* Populate stored nonzero values. */
    size_t n = da_length(&sm->entries);
    for (size_t i = 0; i < n; i++) {
        /*
         * Copy the triplet out of the DynamicArray before calling
         * dense_matrix_set() — which does NOT mutate sm->entries,
         * so the pointer would remain valid anyway, but copying first
         * is the safest practice.
         */
        Triplet t;
        da_get(&sm->entries, i, &t);
        s = dense_matrix_set(dm, t.row, t.col, t.value);
        if (s != ZF_OK) {
            dense_matrix_destroy(dm);
            return s;
        }
    }
    return ZF_OK;
}

/* ═══════════════════════════════════════════════════════════════════
 * Inspection
 * ═══════════════════════════════════════════════════════════════════ */

size_t sparse_matrix_nonzero_count(const SparseMatrix *sm)
{
    return (sm != NULL && sm->init) ? da_length(&sm->entries) : 0;
}

ZF_Status sparse_matrix_validate(const SparseMatrix *sm)
{
    if (sm == NULL || !sm->init) return ZF_ERR_PARAM;
    if (sm->rows == 0 || sm->cols == 0) return ZF_ERR_PARAM;

    size_t n = da_length(&sm->entries);

    for (size_t i = 0; i < n; i++) {
        const Triplet *t = (const Triplet *)da_ptr(&sm->entries, i);

        /* Invariant 2: coordinates in bounds. */
        if (t->row >= sm->rows || t->col >= sm->cols) return ZF_ERR_INDEX;

        /* Invariant 4 & 5: no zero, no non-finite. */
        if (t->value == 0.0 || !isfinite(t->value))  return ZF_ERR_VALUE;

        /* Invariants 3 & 6: sorted order and no duplicates. */
        if (i > 0) {
            const Triplet *prev = (const Triplet *)da_ptr(&sm->entries, i - 1);
            int cmp = triplet_cmp(prev, t);
            if (cmp > 0) return ZF_ERR_NOT_FOUND;  /* out of order */
            if (cmp == 0) return ZF_ERR_DUPLICATE;  /* duplicate coord */
        }
    }
    return ZF_OK;
}
