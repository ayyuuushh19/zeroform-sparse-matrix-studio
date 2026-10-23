/*
 * memory_analysis.h — Theoretical payload memory analysis for matrices
 *
 * ZEROFORM Sparse Matrix Studio — Stage 3
 *
 * This header defines:
 *   - MemoryAnalysis: Structure containing theoretical storage metrics.
 *   - memory_analysis_compute: Core calculation function.
 *   - memory_analysis_dense  : Convenience wrapper for DenseMatrix.
 *   - memory_analysis_sparse : Convenience wrapper for SparseMatrix.
 *
 * Theoretical Model Assumptions:
 * ──────────────────────────────
 * - Dense storage : 8 bytes per cell (1 x 64-bit IEEE 754 double).
 * - Sparse storage: 16 bytes per entry (8 bytes double value +
 *                   4 bytes 32-bit row index + 4 bytes 32-bit col index).
 *
 * IMPORTANT ARCHITECTURAL DISTINCTION:
 * ────────────────────────────────────
 * The in-memory C Triplet structure in this engine uses `size_t` coordinates
 * (8 bytes on 64-bit Windows x64, resulting in sizeof(Triplet) == 24 bytes).
 * This module implements the course assignment's FIXED-ASSUMPTION theoretical
 * payload model:
 *   dense_payload_bytes  = total_cells * 8
 *   sparse_payload_bytes = nnz * 16
 *
 * This theoretical model explicitly evaluates idealized algorithmic data
 * payload sizes. It does NOT equal `nnz * sizeof(Triplet)` on 64-bit architectures,
 * and deliberately excludes struct alignment padding, allocation metadata,
 * capacity growth buffers, pointers, and operating system heap overhead.
 */

#ifndef MEMORY_ANALYSIS_H
#define MEMORY_ANALYSIS_H

#include <stddef.h>
#include "zeroform.h"
#include "matrix.h"

/* Fixed theoretical payload size constants (bytes) */
#define THEORETICAL_DENSE_BYTES_PER_CELL   8ULL
#define THEORETICAL_SPARSE_BYTES_PER_ENTRY 16ULL

typedef struct {
    size_t rows;                  /* Matrix row count                                     */
    size_t cols;                  /* Matrix column count                                  */
    size_t total_cells;           /* rows * cols                                          */
    size_t nnz;                   /* Number of stored nonzero entries                     */
    size_t zero_count;            /* total_cells - nnz                                    */
    double density_percent;       /* 100.0 * nnz / total_cells                            */
    double sparsity_percent;      /* 100.0 * zero_count / total_cells                     */
    size_t dense_payload_bytes;   /* Theoretical dense payload: total_cells * 8           */
    size_t sparse_payload_bytes;  /* Theoretical sparse payload: nnz * 16 (NOT sizeof)    */
    int    sparse_is_smaller;     /* 1 if sparse is strictly smaller, else 0              */
    size_t savings_bytes;         /* Absolute magnitude of savings or penalty             */
    double percent_saved;         /* Percent saved when sparse is smaller, else 0.0       */
} MemoryAnalysis;

/*
 * memory_analysis_compute — compute theoretical memory metrics for given
 * dimensions and nonzero count.
 *
 * Parameters:
 *   rows : number of matrix rows (must be > 0).
 *   cols : number of matrix columns (must be > 0).
 *   nnz  : number of nonzeros (must be <= rows * cols).
 *   out  : pointer to destination struct; populated ONLY on ZF_OK.
 *
 * Returns:
 *   ZF_OK           — success; *out is populated.
 *   ZF_ERR_PARAM    — rows == 0, cols == 0, out == NULL, or nnz > rows * cols.
 *   ZF_ERR_OVERFLOW — size_t overflow in dimension or byte calculations.
 */
ZF_Status memory_analysis_compute(size_t rows,
                                  size_t cols,
                                  size_t nnz,
                                  MemoryAnalysis *out);

/* Convenience wrapper for an initialized DenseMatrix. */
ZF_Status memory_analysis_dense(const DenseMatrix *dm, MemoryAnalysis *out);

/* Convenience wrapper for an initialized SparseMatrix. */
ZF_Status memory_analysis_sparse(const SparseMatrix *sm, MemoryAnalysis *out);

#endif /* MEMORY_ANALYSIS_H */
