/*
 * matrix.h — Public types and API for DenseMatrix and SparseMatrix
 *
 * ZEROFORM Sparse Matrix Studio — Stage 2
 *
 * This header defines:
 *   Triplet      — a (row, col, value) entry for sparse storage
 *   DenseMatrix  — row-major flat storage of doubles via DynamicArray
 *   SparseMatrix — triplet-list storage of nonzero entries via DynamicArray
 *
 * Dependency: zeroform.h (DynamicArray, ZF_Status)
 * Do NOT include this header before zeroform.h.
 *
 * ─── Memory ownership ───────────────────────────────────────────────
 *   Both DenseMatrix and SparseMatrix OWN their DynamicArray allocations.
 *   Always call dense_matrix_destroy() / sparse_matrix_destroy() when done.
 *   After destruction, structs are zeroed and must be re-initialized
 *   before use.  Double destruction is safe (free(NULL) is a no-op).
 *
 * ─── Coordinate convention ──────────────────────────────────────────
 *   All row and column indices are zero-based.
 *   Valid coordinates: row in [0, rows-1], col in [0, cols-1].
 *
 * ─── SparseMatrix invariants ────────────────────────────────────────
 *   After every successful public mutation these invariants hold:
 *   1. rows > 0, cols > 0 (dimensions never change after init).
 *   2. Every stored coordinate is within [0, rows-1] × [0, cols-1].
 *   3. No coordinate appears more than once.
 *   4. No stored triplet has value == 0.0 (including -0.0).
 *   5. All stored values are finite (isfinite(value) == true).
 *   6. Triplets are sorted in ascending row-major (lexicographic) order:
 *      primary key = row, secondary key = col.
 *
 * ─── Numeric policy ─────────────────────────────────────────────────
 *   - Reading a missing coordinate returns 0.0 (implicit zero).
 *   - Setting a coordinate to 0.0 deletes it.
 *   - Inserting a duplicate is a hard error (ZF_ERR_DUPLICATE).
 *   - Deleting a missing coordinate returns ZF_ERR_NOT_FOUND.
 *   - NaN or infinite input values return ZF_ERR_VALUE.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>   /* size_t   */
#include <math.h>     /* isfinite */
#include "zeroform.h" /* DynamicArray, ZF_Status */

/*
 * Workaround for MinGW-w64 math.h fpclassify macro which triggers
 * -Wfloat-conversion when compiling with -Wconversion -Werror.
 * GCC and Clang provide __builtin_isfinite for type-generic finite check.
 */
#if defined(__GNUC__) || defined(__clang__)
#  undef isfinite
#  define isfinite(x) __builtin_isfinite(x)
#endif

/* ─── Universal Matrix Limits ─────────────────────────────────────── */
#define ZF_MAX_DIM   100000ULL
#define ZF_MAX_CELLS 10000000ULL
#define ZF_MAX_NNZ   1000000ULL

#ifndef MAX_CELLS
#define MAX_CELLS ZF_MAX_CELLS
#endif

#ifndef MAX_DIM
#define MAX_DIM ZF_MAX_DIM
#endif

#ifndef MAX_NNZ
#define MAX_NNZ ZF_MAX_NNZ
#endif

/* ═══════════════════════════════════════════════════════════════════
 * Triplet
 * ═══════════════════════════════════════════════════════════════════
 * A single nonzero entry: (row, col, value).
 * row and col are zero-based.
 */
typedef struct {
    size_t row;
    size_t col;
    double value;
} Triplet;

/* ═══════════════════════════════════════════════════════════════════
 * DenseMatrix
 * ═══════════════════════════════════════════════════════════════════
 *
 * Stores all rows*cols cells as doubles in a DynamicArray using
 * flat row-major layout.  Element (r, c) is at flat index r*cols + c.
 *
 * Fields (treat as read-only; always use the accessor functions):
 *   rows     — number of rows
 *   cols     — number of columns
 *   data     — DynamicArray of double; length == rows*cols after init
 *   init     — 1 if the matrix has been successfully initialized, else 0
 */
typedef struct {
    size_t       rows;
    size_t       cols;
    DynamicArray data;   /* elem_size == sizeof(double) */
    int          init;   /* lifecycle guard              */
} DenseMatrix;

/* ─── DenseMatrix lifecycle ─────────────────────────────────────── */

/*
 * dense_matrix_init — allocate a rows×cols zero-filled matrix.
 *
 * Returns ZF_OK, ZF_ERR_PARAM (rows or cols == 0),
 * ZF_ERR_OVERFLOW (rows*cols overflows), ZF_ERR_NOMEM.
 * On any failure, *dm is left in a safe zeroed state.
 */
ZF_Status dense_matrix_init(DenseMatrix *dm, size_t rows, size_t cols);

/*
 * dense_matrix_destroy — free all storage and zero the struct.
 *
 * Safe to call on an already-destroyed struct.
 */
void dense_matrix_destroy(DenseMatrix *dm);

/*
 * dense_matrix_copy — create a deep copy of src in dst.
 *
 * dst must be uninitialized (or destroyed).
 * Returns ZF_OK, ZF_ERR_PARAM, ZF_ERR_NOMEM.
 * On failure, dst is left in a safe zeroed state.
 */
ZF_Status dense_matrix_copy(DenseMatrix *dst, const DenseMatrix *src);

/* ─── DenseMatrix element access ────────────────────────────────── */

/*
 * dense_matrix_get — read the value at (row, col) into *out.
 *
 * Returns ZF_OK, ZF_ERR_PARAM (NULL pointers or uninit matrix),
 * ZF_ERR_INDEX (coordinates out of range).
 */
ZF_Status dense_matrix_get(const DenseMatrix *dm,
                            size_t row, size_t col, double *out);

/*
 * dense_matrix_set — write value at (row, col).
 *
 * Rejects non-finite values (ZF_ERR_VALUE).
 * Returns ZF_OK, ZF_ERR_PARAM, ZF_ERR_INDEX, ZF_ERR_VALUE.
 */
ZF_Status dense_matrix_set(DenseMatrix *dm,
                            size_t row, size_t col, double value);

/* ─── DenseMatrix construction ──────────────────────────────────── */

/*
 * dense_matrix_from_flat — construct a DenseMatrix from a flat array
 * of rows*cols doubles in row-major order.
 *
 * Parameters
 *   dm    : output; must be uninitialized.
 *   rows  : number of rows (must be > 0).
 *   cols  : number of columns (must be > 0).
 *   flat  : pointer to rows*cols doubles in row-major order; must not be NULL.
 *
 * Non-finite values in `flat` are rejected (ZF_ERR_VALUE).
 * Returns ZF_OK, ZF_ERR_PARAM, ZF_ERR_OVERFLOW, ZF_ERR_NOMEM, ZF_ERR_VALUE.
 * On any failure, *dm is left in a safe zeroed state.
 */
ZF_Status dense_matrix_from_flat(DenseMatrix *dm,
                                  size_t rows, size_t cols,
                                  const double *flat);

/* ─── DenseMatrix accessors ─────────────────────────────────────── */

/* Return rows; 0 if dm is NULL or uninitialized. */
size_t dense_matrix_rows(const DenseMatrix *dm);

/* Return cols; 0 if dm is NULL or uninitialized. */
size_t dense_matrix_cols(const DenseMatrix *dm);

/* ═══════════════════════════════════════════════════════════════════
 * SparseMatrix
 * ═══════════════════════════════════════════════════════════════════
 *
 * Stores only nonzero entries as Triplet objects inside a DynamicArray.
 * The logical matrix dimensions are preserved independently.
 *
 * Fields (read-only from outside):
 *   rows     — number of rows
 *   cols     — number of columns
 *   entries  — DynamicArray of Triplet; length == number of nonzero entries
 *   init     — lifecycle guard
 *
 * Invariants: see file-level comment above.
 */
typedef struct {
    size_t       rows;
    size_t       cols;
    DynamicArray entries;  /* elem_size == sizeof(Triplet) */
    int          init;     /* lifecycle guard               */
} SparseMatrix;

/* ─── Traversal callback ────────────────────────────────────────── */

/*
 * SparseTraverseFn — called once per stored Triplet during traversal.
 *
 * Parameters
 *   t    : const pointer to the current Triplet; must not be modified.
 *   ctx  : user-supplied context pointer (may be NULL).
 *
 * The callback must not mutate the SparseMatrix being traversed.
 * Return 0 to continue traversal; return any nonzero value to stop early.
 */
typedef int (*SparseTraverseFn)(const Triplet *t, void *ctx);

/* ─── SparseMatrix lifecycle ────────────────────────────────────── */

/*
 * sparse_matrix_init — create an empty matrix with given dimensions.
 *
 * Returns ZF_OK, ZF_ERR_PARAM (rows or cols == 0), ZF_ERR_NOMEM.
 * On failure, *sm is in a safe zeroed state.
 */
ZF_Status sparse_matrix_init(SparseMatrix *sm, size_t rows, size_t cols);

/*
 * sparse_matrix_destroy — free all storage and zero the struct.
 *
 * Safe to call on an already-destroyed struct.
 */
void sparse_matrix_destroy(SparseMatrix *sm);

/* ─── SparseMatrix construction ─────────────────────────────────── */

/*
 * sparse_matrix_from_dense — convert a DenseMatrix to sparse form.
 *
 * Only nonzero (and finite) values are stored.  Zero values are skipped.
 * Returns ZF_OK, ZF_ERR_PARAM, ZF_ERR_NOMEM.
 * On failure, *sm is in a safe zeroed state.
 */
ZF_Status sparse_matrix_from_dense(SparseMatrix *sm, const DenseMatrix *dm);

/*
 * sparse_matrix_from_triplets — build a SparseMatrix from a Triplet array.
 *
 * Rules:
 *   - triplets may be in any order (they are sorted internally).
 *   - Explicit zero values are silently skipped.
 *   - Non-finite values return ZF_ERR_VALUE.
 *   - Out-of-bounds coordinates return ZF_ERR_INDEX.
 *   - Duplicate coordinates (after skipping zeros) return ZF_ERR_DUPLICATE.
 *
 * On any failure, *sm is destroyed and in a safe zeroed state.
 * Returns ZF_OK, ZF_ERR_PARAM, ZF_ERR_NOMEM, ZF_ERR_VALUE,
 *         ZF_ERR_INDEX, ZF_ERR_DUPLICATE.
 */
ZF_Status sparse_matrix_from_triplets(SparseMatrix *sm,
                                       size_t rows, size_t cols,
                                       const Triplet *triplets,
                                       size_t count);

/* ─── SparseMatrix read access ──────────────────────────────────── */

/*
 * sparse_matrix_get — return the value at (row, col).
 *
 * If the coordinate is not stored, returns 0.0 in *out (implicit zero).
 * Does not distinguish "found" from "not found" in the return status;
 * use sparse_matrix_search() if that distinction matters.
 *
 * Returns ZF_OK, ZF_ERR_PARAM, ZF_ERR_INDEX.
 */
ZF_Status sparse_matrix_get(const SparseMatrix *sm,
                              size_t row, size_t col, double *out);

/*
 * sparse_matrix_search — locate (row, col) using binary search.
 *
 * Writes the found value into *out_val (may be NULL if caller doesn't care).
 * Writes the position in the entries array into *out_pos (may be NULL).
 *
 * Returns ZF_OK if found, ZF_ERR_NOT_FOUND if absent,
 *         ZF_ERR_PARAM if sm or coordinates invalid, ZF_ERR_INDEX.
 */
ZF_Status sparse_matrix_search(const SparseMatrix *sm,
                                size_t row, size_t col,
                                double *out_val, size_t *out_pos);

/* ─── SparseMatrix mutation ─────────────────────────────────────── */

/*
 * sparse_matrix_set — set (row, col) to value, maintaining invariants.
 *
 * Behavior:
 *   value == 0.0  →  delete entry if present; no-op if absent.
 *   value != 0.0  →  update existing entry, or insert new one.
 *
 * Returns ZF_OK, ZF_ERR_PARAM, ZF_ERR_INDEX, ZF_ERR_VALUE, ZF_ERR_NOMEM.
 */
ZF_Status sparse_matrix_set(SparseMatrix *sm,
                              size_t row, size_t col, double value);

/*
 * sparse_matrix_insert — insert a new nonzero entry.
 *
 * Fails with ZF_ERR_DUPLICATE if (row, col) already exists.
 * Fails with ZF_ERR_VALUE    if value is zero or non-finite.
 *
 * Returns ZF_OK, ZF_ERR_PARAM, ZF_ERR_INDEX, ZF_ERR_VALUE,
 *         ZF_ERR_DUPLICATE, ZF_ERR_NOMEM.
 */
ZF_Status sparse_matrix_insert(SparseMatrix *sm,
                                size_t row, size_t col, double value);

/*
 * sparse_matrix_delete — remove the entry at (row, col).
 *
 * Returns ZF_OK, ZF_ERR_NOT_FOUND, ZF_ERR_PARAM, ZF_ERR_INDEX.
 */
ZF_Status sparse_matrix_delete(SparseMatrix *sm, size_t row, size_t col);

/* ─── SparseMatrix traversal ────────────────────────────────────── */

/*
 * sparse_matrix_traverse — call fn once per stored Triplet in row-major order.
 *
 * ctx is passed through to fn unchanged.
 * Stops early if fn returns nonzero; that nonzero value is returned.
 * Returns ZF_OK if all entries were visited, ZF_ERR_PARAM if sm or fn NULL.
 */
ZF_Status sparse_matrix_traverse(const SparseMatrix *sm,
                                  SparseTraverseFn fn, void *ctx);

/* ─── SparseMatrix sorting ──────────────────────────────────────── */

/*
 * sparse_matrix_sort — sort the entries array in ascending row-major order.
 *
 * Uses a custom bottom-up merge sort implemented without qsort().
 * Time complexity: O(nnz * log(nnz)).
 * Space complexity: O(nnz) for the temporary merge buffer.
 *
 * A canonical matrix is unchanged.  An out-of-order matrix is sorted in place
 * (logically; the DynamicArray backing storage may be rebuilt).
 *
 * Returns ZF_OK, ZF_ERR_PARAM, ZF_ERR_NOMEM.
 */
ZF_Status sparse_matrix_sort(SparseMatrix *sm);

/* ─── SparseMatrix conversion ────────────────────────────────────── */

/*
 * sparse_matrix_to_dense — convert to a DenseMatrix.
 *
 * *dm must be uninitialized (or destroyed).
 * Returns ZF_OK, ZF_ERR_PARAM, ZF_ERR_NOMEM.
 * On failure, *dm is in a safe zeroed state.
 */
ZF_Status sparse_matrix_to_dense(const SparseMatrix *sm, DenseMatrix *dm);

/* ─── SparseMatrix inspection ───────────────────────────────────── */

/*
 * sparse_matrix_nonzero_count — return the number of stored nonzero entries.
 * Returns 0 if sm is NULL or uninitialized.
 */
size_t sparse_matrix_nonzero_count(const SparseMatrix *sm);

/*
 * sparse_matrix_validate — verify all structural invariants.
 *
 * Returns ZF_OK if the matrix is canonical, or the first error found:
 *   ZF_ERR_PARAM     — sm is NULL or uninitialized
 *   ZF_ERR_INDEX     — a stored coordinate is out of matrix bounds
 *   ZF_ERR_VALUE     — a stored value is zero or non-finite
 *   ZF_ERR_DUPLICATE — a coordinate appears more than once
 *   ZF_ERR_NOT_FOUND — entries are out of row-major order (reusing NOT_FOUND
 *                      to mean "order violation detected")
 * Never silently repairs the matrix.
 */
ZF_Status sparse_matrix_validate(const SparseMatrix *sm);

#endif /* MATRIX_H */
