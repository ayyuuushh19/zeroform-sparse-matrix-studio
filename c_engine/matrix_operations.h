/*
 * matrix_operations.h — Mathematical operations for SparseMatrix
 *
 * ZEROFORM Sparse Matrix Studio — Stage 3
 *
 * This header defines the public interface for:
 *   - sparse_matrix_add      : C = A + B   (O(nnzA + nnzB))
 *   - sparse_matrix_multiply : C = A * B   (Row-indexed accumulator algorithm)
 *   - sparse_matrix_transpose: C = A^T     (Fast counting-sort transpose O(cols + nnz))
 *
 * Transactional Contract:
 * ───────────────────────
 * All operations follow a strict transactional contract:
 * 1. Output object: Caller must pass a pointer to an uninitialized (or destroyed)
 *    SparseMatrix struct for `out`.
 * 2. Aliasing: In-place calculation / aliasing (e.g. out == a, out == b) is rejected
 *    with ZF_ERR_PARAM to protect against data hazards and partial-state corruption.
 * 3. Failure safety: If any validation, dimension check, arithmetic, or memory
 *    allocation fails, `*out` is left untouched. Internal temporary state is
 *    fully cleaned up.
 * 4. Success: On ZF_OK, `*out` contains the canonical, valid, sorted SparseMatrix
 *    and caller assumes ownership (must call sparse_matrix_destroy(out) when done).
 *
 * Numeric Policy:
 * ───────────────
 * - Calculations use IEEE 754 double precision.
 * - Results that evaluate to exactly 0.0 (including -0.0) or cancel out are
 *   never stored.
 * - Intermediate arithmetic resulting in NaN or ±Inf returns ZF_ERR_VALUE.
 */

#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include "matrix.h"
#include "zeroform.h"

/*
 * sparse_matrix_add — compute C = A + B.
 *
 * Requirements:
 *   - Both `a` and `b` must be valid, initialized SparseMatrix structs.
 *   - Dimensions must be identical: a->rows == b->rows and a->cols == b->cols.
 *     If dimensions differ, returns ZF_ERR_DIMENSION.
 *   - `out` must not alias `a` or `b`.
 *
 * Algorithm:
 *   Direct two-way merge of row-major triplet sequences in O(nnzA + nnzB) time.
 *   Disjoint coordinates are copied; overlapping coordinates have their values
 *   summed. Exact zero cancellation results are omitted from the output.
 *
 * Complexity:
 *   Time:  O(nnzA + nnzB)
 *   Space: O(1) auxiliary heap space beyond the result storage.
 *
 * Returns:
 *   ZF_OK             — success; *out contains the canonical sum.
 *   ZF_ERR_PARAM      — NULL pointers, uninitialized matrices, or aliasing.
 *   ZF_ERR_DIMENSION  — shape mismatch (rows or cols differ).
 *   ZF_ERR_VALUE      — non-finite arithmetic result encountered.
 *   ZF_ERR_NOMEM      — memory allocation failure.
 */
ZF_Status sparse_matrix_add(const SparseMatrix *a,
                            const SparseMatrix *b,
                            SparseMatrix *out);

/*
 * sparse_matrix_multiply — compute C = A * B.
 *
 * Requirements:
 *   - Both `a` and `b` must be valid, initialized SparseMatrix structs.
 *   - Inner dimensions must match: a->cols == b->rows.
 *     If shapes are incompatible, returns ZF_ERR_DIMENSION.
 *   - The result has dimensions a->rows x b->cols.
 *   - `out` must not alias `a` or `b`.
 *
 * Algorithm:
 *   Gustavson row-indexed sparse matrix multiplication:
 *   1. Builds row offsets for B in O(R_B + nnzB) time.
 *   2. Maintains a dense accumulator array and touched flags for B's columns.
 *   3. For each row i of A, multiplies A[i, k] by matching entries in B[k, :],
 *      accumulating into output column slots.
 *   4. Custom-sorts touched column indices in O(t_i * log t_i) time.
 *   5. Emits non-zero products in strict row-major order.
 *   6. Clears accumulator slots in O(t_i) time per row.
 *
 * Complexity:
 *   Time:  O(R_B + nnzB + C_B + F + SUM(t_i * log t_i) + nnzC)
 *          where F = number of matching nonzero products visited,
 *                t_i = distinct output columns touched in row i.
 *   Space: O(R_B + C_B) auxiliary heap space for row index and accumulator.
 *
 * Returns:
 *   ZF_OK             — success; *out contains the canonical product.
 *   ZF_ERR_PARAM      — NULL pointers, uninitialized matrices, or aliasing.
 *   ZF_ERR_DIMENSION  — incompatible dimensions (a->cols != b->rows).
 *   ZF_ERR_OVERFLOW   — output dimension product overflows size_t.
 *   ZF_ERR_VALUE      — non-finite intermediate product or sum.
 *   ZF_ERR_NOMEM      — memory allocation failure.
 */
ZF_Status sparse_matrix_multiply(const SparseMatrix *a,
                                 const SparseMatrix *b,
                                 SparseMatrix *out);

/*
 * sparse_matrix_transpose — compute C = A^T.
 *
 * Requirements:
 *   - `input` must be a valid, initialized SparseMatrix struct.
 *   - Output dimensions are: out->rows = input->cols, out->cols = input->rows.
 *   - `out` must not alias `input`.
 *
 * Algorithm:
 *   Fast counting-sort transpose:
 *   1. Computes column nonzero frequencies in input: O(cols + nnz).
 *   2. Computes starting offsets for each output row.
 *   3. Places each triplet (r, c, v) at its exact bucket location as (c, r, v).
 *   Since input triplets are traversed in increasing row order, entries within
 *   each output row are placed in strictly increasing column order.
 *
 * Complexity:
 *   Time:  O(cols + nnz) linear time — faster than general O(nnz log nnz) sort.
 *   Space: O(cols + nnz) temporary auxiliary buffer.
 *
 * Returns:
 *   ZF_OK             — success; *out contains the transposed matrix.
 *   ZF_ERR_PARAM      — NULL pointers, uninitialized matrix, or aliasing.
 *   ZF_ERR_NOMEM      — memory allocation failure.
 */
ZF_Status sparse_matrix_transpose(const SparseMatrix *input,
                                  SparseMatrix *out);

#endif /* MATRIX_OPERATIONS_H */
