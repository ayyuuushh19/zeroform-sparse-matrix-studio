/*
 * matrix_operations.c — Implementation of SparseMatrix operations
 *
 * ZEROFORM Sparse Matrix Studio — Stage 3
 *
 * Algorithms implemented from scratch:
 *   - Two-way merge addition               : O(nnzA + nnzB)
 *   - Gustavson row-indexed multiplication : O(R_B + nnzB + C_B + F + SUM(t_i log t_i) + nnzC)
 *   - Counting-sort linear-time transpose   : O(cols + nnz)
 *   - Bottom-up iterative index merge sort : O(t_i log t_i) without qsort()
 */

#include "matrix_operations.h"
#include "matrix.h"
#include "zeroform.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ───────────────────────────────────────────────────────────────────
 * Internal helper: overflow-safe multiplication of two size_t values.
 * Returns 0 on success, -1 on overflow.
 * ─────────────────────────────────────────────────────────────────── */
static int mo_safe_mul(size_t a, size_t b, size_t *result)
{
    if (a == 0 || b == 0) { *result = 0; return 0; }
    if (a > SIZE_MAX / b) return -1;
    *result = a * b;
    return 0;
}

/* ───────────────────────────────────────────────────────────────────
 * Internal helper: append an already-validated triplet to sm->entries.
 * Precondition: caller guarantees coordinates are valid and in order.
 * ─────────────────────────────────────────────────────────────────── */
static ZF_Status append_triplet(SparseMatrix *sm, size_t r, size_t c, double val)
{
    Triplet t;
    t.row = r;
    t.col = c;
    t.value = val;
    return da_append(&sm->entries, &t);
}

/* ───────────────────────────────────────────────────────────────────
 * Custom bottom-up iterative merge sort for size_t column indices.
 * Operates without qsort() or recursion.
 * ─────────────────────────────────────────────────────────────────── */
static void merge_size_t_runs(size_t *buf, size_t *tmp,
                              size_t l, size_t m, size_t r)
{
    size_t i = l, j = m, k = l;
    while (i < m && j < r) {
        if (buf[i] <= buf[j]) tmp[k++] = buf[i++];
        else                  tmp[k++] = buf[j++];
    }
    while (i < m) tmp[k++] = buf[i++];
    while (j < r) tmp[k++] = buf[j++];
    memcpy(&buf[l], &tmp[l], (r - l) * sizeof(size_t));
}

static void sort_size_t_indices(size_t *buf, size_t n, size_t *tmp)
{
    if (n <= 1) return;
    for (size_t width = 1; width < n; width *= 2) {
        for (size_t l = 0; l < n; l += 2 * width) {
            size_t m = l + width;
            if (m >= n) break;
            size_t r = m + width;
            if (r > n) r = n;
            merge_size_t_runs(buf, tmp, l, m, r);
        }
        if (width > SIZE_MAX / 2) break;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Matrix Addition: C = A + B
 * ═══════════════════════════════════════════════════════════════════ */

ZF_Status sparse_matrix_add(const SparseMatrix *a,
                            const SparseMatrix *b,
                            SparseMatrix *out)
{
    /* Reject NULL pointers and unsafe output aliasing. */
    if (a == NULL || b == NULL || out == NULL) return ZF_ERR_PARAM;
    if (out == a || out == b)                   return ZF_ERR_PARAM;

    /* Validate structural invariants of both input matrices. */
    if (sparse_matrix_validate(a) != ZF_OK || sparse_matrix_validate(b) != ZF_OK) {
        return ZF_ERR_PARAM;
    }

    /* Addition requires identical dimensions. */
    if (a->rows != b->rows || a->cols != b->cols) {
        return ZF_ERR_DIMENSION;
    }

    /* Build temporary result object (transactional safety). */
    SparseMatrix tmp;
    ZF_Status s = sparse_matrix_init(&tmp, a->rows, a->cols);
    if (s != ZF_OK) return s;

    size_t nnzA = da_length(&a->entries);
    size_t nnzB = da_length(&b->entries);
    size_t idxA = 0, idxB = 0;

    /* Two-way merge in O(nnzA + nnzB) time. */
    while (idxA < nnzA && idxB < nnzB) {
        const Triplet *tA = (const Triplet *)da_ptr(&a->entries, idxA);
        const Triplet *tB = (const Triplet *)da_ptr(&b->entries, idxB);

        int cmp;
        if (tA->row != tB->row) {
            cmp = (tA->row < tB->row) ? -1 : 1;
        } else if (tA->col != tB->col) {
            cmp = (tA->col < tB->col) ? -1 : 1;
        } else {
            cmp = 0;
        }

        if (cmp < 0) {
            s = append_triplet(&tmp, tA->row, tA->col, tA->value);
            if (s != ZF_OK) { sparse_matrix_destroy(&tmp); return s; }
            idxA++;
        } else if (cmp > 0) {
            s = append_triplet(&tmp, tB->row, tB->col, tB->value);
            if (s != ZF_OK) { sparse_matrix_destroy(&tmp); return s; }
            idxB++;
        } else {
            /* Overlapping coordinate: compute sum. */
            double sum = tA->value + tB->value;
            if (!isfinite(sum)) {
                sparse_matrix_destroy(&tmp);
                return ZF_ERR_VALUE;
            }
            /* Nonzero sum is preserved; exact zero cancellation is omitted. */
            if (sum != 0.0) {
                s = append_triplet(&tmp, tA->row, tA->col, sum);
                if (s != ZF_OK) { sparse_matrix_destroy(&tmp); return s; }
            }
            idxA++;
            idxB++;
        }
    }

    /* Copy remaining entries from A. */
    while (idxA < nnzA) {
        const Triplet *tA = (const Triplet *)da_ptr(&a->entries, idxA);
        s = append_triplet(&tmp, tA->row, tA->col, tA->value);
        if (s != ZF_OK) { sparse_matrix_destroy(&tmp); return s; }
        idxA++;
    }

    /* Copy remaining entries from B. */
    while (idxB < nnzB) {
        const Triplet *tB = (const Triplet *)da_ptr(&b->entries, idxB);
        s = append_triplet(&tmp, tB->row, tB->col, tB->value);
        if (s != ZF_OK) { sparse_matrix_destroy(&tmp); return s; }
        idxB++;
    }

    /* Validate output invariants before publishing. */
    s = sparse_matrix_validate(&tmp);
    if (s != ZF_OK) {
        sparse_matrix_destroy(&tmp);
        return s;
    }

    /* Publish result transactionally. */
    *out = tmp;
    return ZF_OK;
}

/* ═══════════════════════════════════════════════════════════════════
 * Matrix Multiplication: C = A * B
 * ═══════════════════════════════════════════════════════════════════ */

ZF_Status sparse_matrix_multiply(const SparseMatrix *a,
                                 const SparseMatrix *b,
                                 SparseMatrix *out)
{
    /* Reject NULL pointers and unsafe output aliasing. */
    if (a == NULL || b == NULL || out == NULL) return ZF_ERR_PARAM;
    if (out == a || out == b)                   return ZF_ERR_PARAM;

    /* Validate structural invariants of both input matrices. */
    if (sparse_matrix_validate(a) != ZF_OK || sparse_matrix_validate(b) != ZF_OK) {
        return ZF_ERR_PARAM;
    }

    /* Matrix multiplication dimension compatibility: A.cols == B.rows. */
    if (a->cols != b->rows) {
        return ZF_ERR_DIMENSION;
    }

    /* Cell ceiling check on output dimensions: A.rows * B.cols. */
    if (b->cols > 0 && a->rows > MAX_CELLS / b->cols) {
        return ZF_ERR_OVERFLOW;
    }

    /* Overflow check on output dimensions: A.rows * B.cols. */
    size_t total_out_cells;
    if (mo_safe_mul(a->rows, b->cols, &total_out_cells) != 0) {
        return ZF_ERR_OVERFLOW;
    }

    /* Initialize temporary output matrix. */
    SparseMatrix tmp;
    ZF_Status s = sparse_matrix_init(&tmp, a->rows, b->cols);
    if (s != ZF_OK) return s;

    /* Step B: Build row offsets for matrix B.
     * b_row_ptr[r] to b_row_ptr[r+1] bounds the entries of row r in B. */
    size_t *b_row_ptr = (size_t *)calloc(b->rows + 1, sizeof(size_t));
    if (b_row_ptr == NULL) {
        sparse_matrix_destroy(&tmp);
        return ZF_ERR_NOMEM;
    }

    size_t nnzB = da_length(&b->entries);
    for (size_t i = 0; i < nnzB; i++) {
        const Triplet *t = (const Triplet *)da_ptr(&b->entries, i);
        b_row_ptr[t->row + 1]++;
    }
    for (size_t r = 0; r < b->rows; r++) {
        b_row_ptr[r + 1] += b_row_ptr[r];
    }

    /* Step C: Create dense row accumulator for B's columns. */
    size_t colsB = b->cols;
    double  *acc_val      = (double *)calloc(colsB, sizeof(double));
    uint8_t *acc_touched  = (uint8_t *)calloc(colsB, sizeof(uint8_t));
    size_t  *touched_cols = (size_t *)malloc(colsB * sizeof(size_t));
    size_t  *sort_scratch = (size_t *)malloc(colsB * sizeof(size_t));

    if (acc_val == NULL || acc_touched == NULL ||
        touched_cols == NULL || sort_scratch == NULL) {
        free(b_row_ptr);
        free(acc_val);
        free(acc_touched);
        free(touched_cols);
        free(sort_scratch);
        sparse_matrix_destroy(&tmp);
        return ZF_ERR_NOMEM;
    }

    /* Step D: Iterate through rows of A. */
    size_t nnzA = da_length(&a->entries);
    size_t a_idx = 0;

    ZF_Status status = ZF_OK;

    for (size_t rA = 0; rA < a->rows; rA++) {
        size_t num_touched = 0;

        /* Visit all nonzeros in row rA of A. */
        while (a_idx < nnzA) {
            const Triplet *tA = (const Triplet *)da_ptr(&a->entries, a_idx);
            if (tA->row != rA) break;
            a_idx++;

            size_t k  = tA->col;
            double vA = tA->value;

            /* Visit matching row k of B using precomputed offsets. */
            size_t b_start = b_row_ptr[k];
            size_t b_end   = b_row_ptr[k + 1];

            for (size_t b_i = b_start; b_i < b_end; b_i++) {
                const Triplet *tB = (const Triplet *)da_ptr(&b->entries, b_i);
                size_t col_j = tB->col;
                double vB    = tB->value;

                double prod = vA * vB;
                if (!isfinite(prod)) {
                    status = ZF_ERR_VALUE;
                    goto cleanup_multiply;
                }

                if (!acc_touched[col_j]) {
                    acc_touched[col_j] = 1;
                    acc_val[col_j] = prod;
                    touched_cols[num_touched++] = col_j;
                } else {
                    acc_val[col_j] += prod;
                    if (!isfinite(acc_val[col_j])) {
                        status = ZF_ERR_VALUE;
                        goto cleanup_multiply;
                    }
                }
            }
        }

        /* Step E: Sort touched column indices and emit non-zero products. */
        if (num_touched > 0) {
            sort_size_t_indices(touched_cols, num_touched, sort_scratch);

            for (size_t ci = 0; ci < num_touched; ci++) {
                size_t col_j     = touched_cols[ci];
                double final_val = acc_val[col_j];

                if (final_val != 0.0) {
                    s = append_triplet(&tmp, rA, col_j, final_val);
                    if (s != ZF_OK) {
                        status = s;
                        goto cleanup_multiply;
                    }
                }

                /* Reset accumulator slot for the next row. */
                acc_val[col_j]     = 0.0;
                acc_touched[col_j] = 0;
            }
        }
    }

cleanup_multiply:
    free(b_row_ptr);
    free(acc_val);
    free(acc_touched);
    free(touched_cols);
    free(sort_scratch);

    if (status != ZF_OK) {
        sparse_matrix_destroy(&tmp);
        return status;
    }

    /* Step F: Validate and publish result transactionally. */
    s = sparse_matrix_validate(&tmp);
    if (s != ZF_OK) {
        sparse_matrix_destroy(&tmp);
        return s;
    }

    *out = tmp;
    return ZF_OK;
}

/* ═══════════════════════════════════════════════════════════════════
 * Matrix Transpose: C = A^T
 * ═══════════════════════════════════════════════════════════════════ */

ZF_Status sparse_matrix_transpose(const SparseMatrix *input,
                                  SparseMatrix *out)
{
    /* Reject NULL pointers and unsafe output aliasing. */
    if (input == NULL || out == NULL || out == input) return ZF_ERR_PARAM;

    /* Validate structural invariants of the input matrix. */
    if (sparse_matrix_validate(input) != ZF_OK) {
        return ZF_ERR_PARAM;
    }

    /* Swapped dimensions: rows_out = cols_in, cols_out = rows_in. */
    size_t rows_out = input->cols;
    size_t cols_out = input->rows;
    size_t nnz      = da_length(&input->entries);

    /* Initialize temporary output matrix. */
    SparseMatrix tmp;
    ZF_Status s = sparse_matrix_init(&tmp, rows_out, cols_out);
    if (s != ZF_OK) return s;

    if (nnz == 0) {
        *out = tmp;
        return ZF_OK;
    }

    /* Fast counting-sort transpose: O(cols_in + nnz) linear time. */
    size_t *col_counts  = (size_t *)calloc(input->cols, sizeof(size_t));
    size_t *col_offsets = (size_t *)malloc(input->cols * sizeof(size_t));
    Triplet *buf        = (Triplet *)malloc(nnz * sizeof(Triplet));

    if (col_counts == NULL || col_offsets == NULL || buf == NULL) {
        free(col_counts);
        free(col_offsets);
        free(buf);
        sparse_matrix_destroy(&tmp);
        return ZF_ERR_NOMEM;
    }

    /* Count number of nonzeros in each column of the input matrix. */
    for (size_t i = 0; i < nnz; i++) {
        const Triplet *t = (const Triplet *)da_ptr(&input->entries, i);
        col_counts[t->col]++;
    }

    /* Compute starting bucket offsets for each transposed row. */
    size_t prefix_sum = 0;
    for (size_t c = 0; c < input->cols; c++) {
        col_offsets[c] = prefix_sum;
        prefix_sum += col_counts[c];
    }

    /* Distribute swapped triplets directly into their bucket positions.
     * Because input is ordered by row, entries for each column are visited
     * in strictly ascending row order, ensuring canonical row-major order. */
    for (size_t i = 0; i < nnz; i++) {
        const Triplet *t = (const Triplet *)da_ptr(&input->entries, i);
        size_t dest = col_offsets[t->col]++;
        buf[dest].row   = t->col;
        buf[dest].col   = t->row;
        buf[dest].value = t->value;
    }

    /* Populate the output DynamicArray in sorted order. */
    for (size_t i = 0; i < nnz; i++) {
        s = da_append(&tmp.entries, &buf[i]);
        if (s != ZF_OK) {
            free(col_counts);
            free(col_offsets);
            free(buf);
            sparse_matrix_destroy(&tmp);
            return s;
        }
    }

    free(col_counts);
    free(col_offsets);
    free(buf);

    /* Validate invariants and publish result transactionally. */
    s = sparse_matrix_validate(&tmp);
    if (s != ZF_OK) {
        sparse_matrix_destroy(&tmp);
        return s;
    }

    *out = tmp;
    return ZF_OK;
}
