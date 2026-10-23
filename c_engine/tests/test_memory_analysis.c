/*
 * test_memory_analysis.c — Unit tests for theoretical memory analysis
 *
 * ZEROFORM Sparse Matrix Studio — Stage 3
 */

#include "../memory_analysis.h"
#include "../matrix.h"
#include "../zeroform.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

static int g_pass = 0;
static int g_fail = 0;
static const char *g_test = "(none)";

#define CHECK(expr)                                                         \
    do {                                                                    \
        if (!(expr)) {                                                      \
            fprintf(stderr, "  FAIL [%s] line %d: %s\n",                   \
                    g_test, __LINE__, #expr);                               \
            g_fail++;                                                       \
        } else {                                                            \
            g_pass++;                                                       \
        }                                                                   \
    } while (0)

#define BEGIN(name) do { g_test = (name); printf("  %-58s\n", (name)); fflush(stdout); } while(0)

/* MA-01: All-zero 3x4 matrix */
static void test_ma_all_zero(void)
{
    BEGIN("MA-01 all-zero 3x4 matrix");
    MemoryAnalysis ma;
    CHECK(memory_analysis_compute(3, 4, 0, &ma) == ZF_OK);
    CHECK(ma.rows == 3);
    CHECK(ma.cols == 4);
    CHECK(ma.total_cells == 12);
    CHECK(ma.nnz == 0);
    CHECK(ma.zero_count == 12);
    CHECK(fabs(ma.density_percent - 0.0) < 1e-9);
    CHECK(fabs(ma.sparsity_percent - 100.0) < 1e-9);
    CHECK(ma.dense_payload_bytes == 96);  /* 12 * 8 */
    CHECK(ma.sparse_payload_bytes == 0);  /* 0 * 16 */
    CHECK(ma.sparse_is_smaller == 1);
    CHECK(ma.savings_bytes == 96);
    CHECK(fabs(ma.percent_saved - 100.0) < 1e-9);
}

/* MA-02: Half-dense 2x2 matrix with 2 nonzeros (equal payloads) */
static void test_ma_equal_payload(void)
{
    BEGIN("MA-02 2x2 with 2 nonzeros (equal 32-byte payload)");
    MemoryAnalysis ma;
    CHECK(memory_analysis_compute(2, 2, 2, &ma) == ZF_OK);
    CHECK(ma.total_cells == 4);
    CHECK(ma.nnz == 2);
    CHECK(ma.zero_count == 2);
    CHECK(fabs(ma.density_percent - 50.0) < 1e-9);
    CHECK(fabs(ma.sparsity_percent - 50.0) < 1e-9);
    CHECK(ma.dense_payload_bytes == 32);  /* 4 * 8 */
    CHECK(ma.sparse_payload_bytes == 32); /* 2 * 16 */
    CHECK(ma.sparse_is_smaller == 0);
    CHECK(ma.savings_bytes == 0);
    CHECK(fabs(ma.percent_saved - 0.0) < 1e-9);
}

/* MA-03: Mostly-dense 3x3 matrix with 8 nonzeros (sparse is larger) */
static void test_ma_mostly_dense(void)
{
    BEGIN("MA-03 3x3 with 8 nonzeros (sparse is larger)");
    MemoryAnalysis ma;
    CHECK(memory_analysis_compute(3, 3, 8, &ma) == ZF_OK);
    CHECK(ma.total_cells == 9);
    CHECK(ma.nnz == 8);
    CHECK(ma.zero_count == 1);
    CHECK(ma.dense_payload_bytes == 72);   /* 9 * 8 */
    CHECK(ma.sparse_payload_bytes == 128); /* 8 * 16 */
    CHECK(ma.sparse_is_smaller == 0);
    CHECK(ma.savings_bytes == 56);         /* 128 - 72 penalty */
    CHECK(fabs(ma.percent_saved - 0.0) < 1e-9);
}

/* MA-04: Single nonzero entry in 10x10 */
static void test_ma_single_nonzero(void)
{
    BEGIN("MA-04 10x10 with 1 nonzero (high sparsity)");
    MemoryAnalysis ma;
    CHECK(memory_analysis_compute(10, 10, 1, &ma) == ZF_OK);
    CHECK(ma.total_cells == 100);
    CHECK(ma.nnz == 1);
    CHECK(ma.zero_count == 99);
    CHECK(fabs(ma.density_percent - 1.0) < 1e-9);
    CHECK(fabs(ma.sparsity_percent - 99.0) < 1e-9);
    CHECK(ma.dense_payload_bytes == 800);
    CHECK(ma.sparse_payload_bytes == 16);
    CHECK(ma.sparse_is_smaller == 1);
    CHECK(ma.savings_bytes == 784);
    CHECK(fabs(ma.percent_saved - 98.0) < 1e-9);
}

/* MA-05: Density and sparsity sum to 100% */
static void test_ma_density_sparsity_sum(void)
{
    BEGIN("MA-05 density and sparsity sum to 100%");
    size_t test_cases[][3] = {
        {5, 5, 5},
        {10, 20, 37},
        {100, 100, 1500},
        {7, 13, 23}
    };
    for (size_t i = 0; i < 4; i++) {
        MemoryAnalysis ma;
        CHECK(memory_analysis_compute(test_cases[i][0], test_cases[i][1], test_cases[i][2], &ma) == ZF_OK);
        CHECK(fabs((ma.density_percent + ma.sparsity_percent) - 100.0) < 1e-9);
    }
}

/* MA-06: Invalid dimensions */
static void test_ma_invalid_dimensions(void)
{
    BEGIN("MA-06 zero dimensions rejected with ZF_ERR_PARAM");
    MemoryAnalysis ma;
    CHECK(memory_analysis_compute(0, 5, 0, &ma) == ZF_ERR_PARAM);
    CHECK(memory_analysis_compute(5, 0, 0, &ma) == ZF_ERR_PARAM);
    CHECK(memory_analysis_compute(0, 0, 0, &ma) == ZF_ERR_PARAM);
    CHECK(memory_analysis_compute(5, 5, 0, NULL) == ZF_ERR_PARAM);
}

/* MA-07: nnz greater than total cells */
static void test_ma_nnz_exceeds_cells(void)
{
    BEGIN("MA-07 nnz > total_cells rejected with ZF_ERR_PARAM");
    MemoryAnalysis ma;
    CHECK(memory_analysis_compute(3, 3, 10, &ma) == ZF_ERR_PARAM);
    CHECK(memory_analysis_compute(2, 5, 11, &ma) == ZF_ERR_PARAM);
}

/* MA-08: Size overflow guard */
static void test_ma_overflow_guard(void)
{
    BEGIN("MA-08 dimension and byte overflow rejected with ZF_ERR_OVERFLOW");
    MemoryAnalysis ma;
    CHECK(memory_analysis_compute(SIZE_MAX, 2, 0, &ma) == ZF_ERR_OVERFLOW);
    /* total_cells * 8 would overflow SIZE_MAX */
    size_t huge_cells = SIZE_MAX / 4;
    CHECK(memory_analysis_compute(huge_cells, 1, 0, &ma) == ZF_ERR_OVERFLOW);
}

/* MA-09: Convenience functions for DenseMatrix and SparseMatrix */
static void test_ma_convenience_wrappers(void)
{
    BEGIN("MA-09 convenience wrappers match core compute");
    DenseMatrix dm;
    dense_matrix_init(&dm, 3, 3);
    dense_matrix_set(&dm, 0, 1, 2.0);
    dense_matrix_set(&dm, 2, 2, 5.0);

    SparseMatrix sm;
    sparse_matrix_init(&sm, 3, 3);
    sparse_matrix_set(&sm, 0, 1, 2.0);
    sparse_matrix_set(&sm, 2, 2, 5.0);

    MemoryAnalysis mad, mas, mac;
    CHECK(memory_analysis_dense(&dm, &mad) == ZF_OK);
    CHECK(memory_analysis_sparse(&sm, &mas) == ZF_OK);
    CHECK(memory_analysis_compute(3, 3, 2, &mac) == ZF_OK);

    CHECK(mad.total_cells == mac.total_cells);
    CHECK(mad.nnz == mac.nnz);
    CHECK(mad.dense_payload_bytes == mac.dense_payload_bytes);
    CHECK(mad.sparse_payload_bytes == mac.sparse_payload_bytes);
    CHECK(mad.savings_bytes == mac.savings_bytes);

    CHECK(mas.total_cells == mac.total_cells);
    CHECK(mas.nnz == mac.nnz);
    CHECK(mas.dense_payload_bytes == mac.dense_payload_bytes);
    CHECK(mas.sparse_payload_bytes == mac.sparse_payload_bytes);

    dense_matrix_destroy(&dm);
    sparse_matrix_destroy(&sm);
}

/* MA-10: 2x3 matrix with 3 nonzeros (equal 48-byte theoretical payload) */
static void test_ma_equal_payload_three_entries(void)
{
    BEGIN("MA-10 2x3 with 3 nonzeros (equal 48-byte payload, not smaller)");
    /*
     * Dense:  2 * 3 = 6 cells * 8 bytes = 48 bytes
     * Sparse: 3 entries * 16 bytes = 48 bytes
     * Equal payloads -> sparse_is_smaller == 0, savings_bytes == 0, percent_saved == 0.0
     *
     * Also verifies that the theoretical payload constant (16 bytes) is decoupled from
     * the 64-bit runtime sizeof(Triplet) (which is 24 bytes on Windows x64).
     */
    MemoryAnalysis ma;
    CHECK(memory_analysis_compute(2, 3, 3, &ma) == ZF_OK);
    CHECK(ma.rows == 2);
    CHECK(ma.cols == 3);
    CHECK(ma.total_cells == 6);
    CHECK(ma.nnz == 3);
    CHECK(ma.zero_count == 3);
    CHECK(fabs(ma.density_percent - 50.0) < 1e-9);
    CHECK(fabs(ma.sparsity_percent - 50.0) < 1e-9);
    CHECK(ma.dense_payload_bytes == 48);   /* 6 * 8 */
    CHECK(ma.sparse_payload_bytes == 48);  /* 3 * 16 */
    CHECK(ma.sparse_is_smaller == 0);      /* Equal: NOT strictly smaller */
    CHECK(ma.savings_bytes == 0);
    CHECK(fabs(ma.percent_saved - 0.0) < 1e-9);

    /* Architecture verification on 64-bit Windows: sizeof(Triplet) is 24 bytes */
    CHECK(sizeof(size_t) == 8);
    CHECK(sizeof(double) == 8);
    CHECK(sizeof(Triplet) == 24);
    CHECK(THEORETICAL_SPARSE_BYTES_PER_ENTRY == 16);
    CHECK(sizeof(Triplet) != THEORETICAL_SPARSE_BYTES_PER_ENTRY);
}

/* ═══════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  ZEROFORM — Memory Analysis Tests  (Stage 3)\n");
    printf("══════════════════════════════════════════════════════════════\n\n");

    test_ma_all_zero();
    test_ma_equal_payload();
    test_ma_mostly_dense();
    test_ma_single_nonzero();
    test_ma_density_sparsity_sum();
    test_ma_invalid_dimensions();
    test_ma_nnz_exceeds_cells();
    test_ma_overflow_guard();
    test_ma_convenience_wrappers();
    test_ma_equal_payload_three_entries();

    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed  (total checks: %d)\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("══════════════════════════════════════════════════════════════\n");

    return (g_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
